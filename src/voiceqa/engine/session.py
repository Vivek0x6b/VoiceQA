"""An interactive call: say something, then wait for the agent's next turn.

CallSession places one call the way run_call does and records it the same
way (caller.wav, agent.wav, mix.wav, events.json). On top of that:

- Every frame received from the agent goes through a TurnTap (engine/turns.py)
  to a TurnDetector running in its own thread (ThreadedTurnDetector), so a
  slow VAD never holds up the event loop or the RTP sender.
- When an agent turn ends, its audio is cut from the tap's buffer and sent
  to the STT engine (in practice an SttWorker started before dialing). The
  call keeps running while it transcribes.
- say() queues audio and returns the slot times of its first and last
  frame. next_agent_turn() returns agent turns in order, each with its
  transcript.
- Turn events go to events.json: agent_speech_start, agent_turn_end,
  caller_say_start, caller_say_end, and overlap when the agent and we speak
  at the same time (only recorded, nothing acts on it). At the end,
  transcript.json holds the agent segments from the per-turn transcripts
  and one caller segment per say() with the text it was given.

All times handed out are seconds on media_clock. events.json and
transcript.json use seconds since the recorder was created, as everywhere.

    session = CallSession(target, host, port, user, password, run_dir, stt=worker)
    print(session.vad_notice or "")       # before dialing
    async with session:                   # dials; CallNotAnswered if it fails
        said = await session.say(samples, text="hi")
        turn = await session.next_agent_turn(timeout=10)
    session.result                        # CallResult, once closed
"""

import asyncio
import contextlib
import gc
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voiceqa.engine.call import CallResult, _first_audio_latency
from voiceqa.engine.transcribe import AsyncStt, shift_segment, write_transcript
from voiceqa.engine.turns import TurnTap
from voiceqa.speech.models import Segment, Transcript
from voiceqa.speech.vad import (
    SPEECH_START,
    TURN_END,
    ThreadedTurnDetector,
    TurnDetector,
    TurnEvent,
    Vad,
    auto_vad_name,
    create_vad,
    vad_notice,
)
from voiceqa.telephony.media import FRAME_SAMPLES, SAMPLE_RATE, PayloadType, media_clock
from voiceqa.telephony.recording import CallRecorder
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip import (
    SipConfig,
    SipError,
    SipStatusError,
    place_call,
    resolve_server,
)
from voiceqa.telephony.sip.transport import routing_ip

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
# Context either side of an agent turn in the audio sent to STT, so the
# first and last sounds are not cut.
STT_PAD_SECONDS = 0.2
DETECTOR_CLOSE_TIMEOUT = 5.0

AGENT_SPEECH_START = "agent_speech_start"
AGENT_TURN_END = "agent_turn_end"
CALLER_SAY_START = "caller_say_start"
CALLER_SAY_END = "caller_say_end"
OVERLAP = "overlap"
VAD_ERROR = "vad_error"
VAD_STOPPED = "vad_stopped"


_live_sessions = 0


def _freeze_heap() -> None:
    """Collect once, then leave every object that exists now out of later
    GC passes, until the last live session closes. Setup (importing
    faster-whisper and onnxruntime for Silero) leaves tens of thousands of
    objects, and a full GC pass over them in the middle of a call holds the
    GIL long enough to make the RTP sender late. The first session of
    several pays the collect, before it dials."""
    global _live_sessions
    if _live_sessions == 0:
        gc.collect()
        gc.freeze()
    _live_sessions += 1


def _unfreeze_heap() -> None:
    # gc.unfreeze() is all or nothing: objects frozen before the first
    # session are released too. Fine in a process voiceqa owns, which is
    # the only place freeze_heap is turned on.
    global _live_sessions
    _live_sessions -= 1
    if _live_sessions == 0:
        gc.unfreeze()


class CallEnded(Exception):
    """The call is over (either side hung up) and no agent turn is left."""


class CallNotAnswered(Exception):
    def __init__(self, status: int | None, reason: str | None) -> None:
        super().__init__(f"call not answered: {status or ''} {reason or ''}".strip())
        self.status = status
        self.reason = reason


class TurnDetectionError(RuntimeError):
    """The VAD failed during the call; no more agent turns will be found."""


@dataclass(frozen=True)
class Said:
    turn: str
    first_frame: float
    last_frame: float

    @property
    def end(self) -> float:
        """When the last sample went out: the last frame's slot plus 20 ms."""
        return self.last_frame + FRAME_SECONDS


@dataclass(frozen=True)
class AgentTurn:
    turn: str
    start: float
    end: float
    samples: np.ndarray = field(repr=False)
    transcript: Transcript | None
    stt_error: str | None = None

    @property
    def text(self) -> str:
        if self.transcript is None:
            return ""
        return " ".join(s.text.strip() for s in self.transcript.segments if s.text.strip())


@dataclass
class _Say:
    turn: str
    text: str | None
    first: float | None = None
    end: float | None = None


@dataclass
class _PendingTurn:
    turn: str
    start: float
    end: float
    samples: np.ndarray
    audio_start: float
    task: asyncio.Future | None


class CallSession:
    def __init__(
        self,
        target: str,
        server_host: str,
        server_port: int,
        username: str,
        password: str,
        record_dir: str | Path,
        stt: AsyncStt | None = None,
        vad: str | Vad | None = "auto",
        codec: PayloadType = PayloadType.PCMU,
        end_silence_ms: float = 700,
        min_speech_ms: float = 100,
        stt_timeout: float = 60.0,
        freeze_heap: bool = False,
    ) -> None:
        """Builds the VAD (loading Silero if that is the one) but does not
        dial; start() or `async with` does. `vad` is a name for create_vad,
        a Vad, or None for no turn detection at all.

        `freeze_heap` collects and freezes the process's GC heap before
        dialing (see _freeze_heap). It changes process-wide GC state, so
        it is off by default and only the voiceqa CLI, which owns its
        process, turns it on."""
        self.target = target
        self.record_dir = Path(record_dir)
        self._sip_config = SipConfig(
            server_host=server_host,
            server_port=server_port,
            username=username,
            password=password,
        )
        self._stt = stt
        self._codec = codec
        self._stt_timeout = stt_timeout
        self._freeze_heap = freeze_heap
        self.vad_notice: str | None = None
        if isinstance(vad, str):
            self.vad_notice = vad_notice(vad)
            vad = create_vad(auto_vad_name() if vad == "auto" else vad)
        self._vad = vad
        self.vad_name: str | None = vad.name if vad is not None else None
        self._turn_detector = (
            TurnDetector(vad, min_speech_ms=min_speech_ms, end_silence_ms=end_silence_ms)
            if vad is not None
            else None
        )

        self.answered = False
        self.status: int | None = None
        self.reason: str | None = None
        self.result: CallResult | None = None
        self.transcript: Transcript | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._rec: CallRecorder | None = None
        self._rtp: RtpSession | None = None
        self._tap: TurnTap | None = None
        self._detector: ThreadedTurnDetector | None = None
        self._call = None
        self._watcher: asyncio.Task | None = None
        self._media_stop: asyncio.Task | None = None
        self._media_done = False
        self._started = False
        self._frozen = False
        self._closed = False
        self._t_answered: float | None = None
        self._duration: float | None = None

        self._open_turn: tuple[str, float] | None = None
        self._agent_count = 0
        self._agent_turns: list[_PendingTurn] = []
        self._ready: deque[_PendingTurn] = deque()
        self._wake = asyncio.Event()
        self._vad_error: BaseException | None = None
        self._say_count = 0
        self._says: list[_Say] = []
        self._active_say: tuple[_Say, asyncio.Future] | None = None
        self._overlaps: set[tuple[str, str]] = set()

    @property
    def ended(self) -> bool:
        return self._media_done

    @property
    def dropped_frames(self) -> int:
        """Agent frames the VAD never saw because its queue was full."""
        return self._detector.dropped_frames if self._detector is not None else 0

    @property
    def vad_error(self) -> BaseException | None:
        return self._vad_error

    async def __aenter__(self) -> "CallSession":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Dial and return once the call is answered and media runs. Raises
        CallNotAnswered (with the recording closed) if it is not."""
        if self._started:
            raise RuntimeError("the session was already started")
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._rec = CallRecorder(self.record_dir)
        if self._freeze_heap:
            _freeze_heap()
            self._frozen = True
        try:
            await self._dial()
        except BaseException:
            await self.close()
            raise
        if not self.answered:
            await self.close()
            raise CallNotAnswered(self.status, self.reason)

    async def _dial(self) -> None:
        try:
            resolved = await resolve_server(
                self._sip_config.server_host, self._sip_config.server_port
            )
        except SipError as exc:
            self.reason = str(exc)
            return
        self._rtp = await RtpSession.open(
            bind_host="0.0.0.0",
            advertise_host=routing_ip(resolved),
            payload_types=(self._codec,),
        )
        feed = None
        if self._turn_detector is not None:
            self._detector = ThreadedTurnDetector(
                self._turn_detector,
                on_event=self._from_thread(self._on_turn_event),
                on_error=self._from_thread(self._on_vad_error),
            )
            self._detector.start()
            feed = self._detector.feed
        self._tap = TurnTap(self._rec, on_agent_frame=feed)
        self._rtp.record(self._tap)
        try:
            call = await place_call(
                self._sip_config,
                self.target,
                self._rtp.local,
                on_event=self._rec.event,
                server=resolved,
            )
        except SipStatusError as exc:
            self.status, self.reason = exc.status, exc.reason
            return
        except SipError as exc:
            self.reason = str(exc)
            return
        self._call = call
        self.answered = True
        self.status, self.reason = 200, "OK"
        self._t_answered = media_clock()
        await self._rtp.start(call.remote_media)
        self._watcher = asyncio.create_task(self._watch_end())

    def _from_thread(self, fn: Callable) -> Callable:
        loop = self._loop

        def hand_over(arg: object) -> None:
            loop.call_soon_threadsafe(fn, arg)

        return hand_over

    async def _watch_end(self) -> None:
        await self._call.wait_ended()
        await self._stop_media()

    def _check_live(self) -> None:
        if not self.answered:
            raise RuntimeError("the call is not up; start() the session first")
        if self._media_done or self._media_stop is not None:
            raise CallEnded("the call has ended")

    async def say(self, samples: np.ndarray, text: str | None = None) -> Said:
        """Send int16 8 kHz mono audio and return once its last frame went
        out, with the slot times of its first and last frame. `text` is what
        the audio says; it becomes the caller segment in transcript.json.
        One say at a time; raises CallEnded if the call ends first."""
        self._check_live()
        x = np.asarray(samples)
        if x.ndim != 1 or x.dtype != np.int16:
            raise TypeError(f"say() needs 1-D int16 samples, got {x.dtype} {x.shape}")
        frames = -(-len(x) // FRAME_SAMPLES)
        if frames == 0:
            raise ValueError("say() needs at least one sample")
        if self._tap.saying:
            raise RuntimeError("a say is still going out")
        self._say_count += 1
        record = _Say(f"caller-{self._say_count}", text)
        done: asyncio.Future[Said] = self._loop.create_future()

        def on_start(at: float) -> None:
            record.first = at
            self._rec.event(CALLER_SAY_START, at=at, turn=record.turn)
            self._overlap_at_say_start(record)

        def on_end(first: float, last: float) -> None:
            record.end = last + FRAME_SECONDS
            self._rec.event(CALLER_SAY_END, at=record.end, turn=record.turn)
            self._active_say = None
            if not done.done():
                done.set_result(Said(record.turn, first, last))

        self._tap.start_say(frames, on_start, on_end)
        self._says.append(record)
        self._active_say = (record, done)
        self._rtp.send(x)
        return await done

    async def next_agent_turn(self, timeout: float) -> AgentTurn:
        """The next agent turn not yet returned, in order, once it has ended
        and been transcribed.

        `timeout` limits the wait for the turn to end; TimeoutError when it
        runs out. The transcript wait after that is limited by stt_timeout
        and shows as `stt_error`, as does an STT failure. Raises
        TurnDetectionError as soon as the VAD fails, and CallEnded once the
        call is over and every turn has been returned.
        """
        if self._turn_detector is None:
            raise RuntimeError("turn detection is off (vad=None)")
        async with asyncio.timeout(timeout):
            while not self._ready:
                if self._vad_error is not None:
                    raise TurnDetectionError(
                        f"turn detection failed: {self._vad_error}"
                    ) from self._vad_error
                if self._media_done:
                    raise CallEnded("the call has ended")
                self._wake.clear()
                await self._wake.wait()
        pending = self._ready.popleft()
        try:
            return await self._finished(pending)
        except asyncio.CancelledError:
            self._ready.appendleft(pending)
            raise

    async def _finished(self, pending: _PendingTurn) -> AgentTurn:
        transcript, error = None, None
        if pending.task is not None:
            try:
                async with asyncio.timeout(self._stt_timeout):
                    transcript = await asyncio.shield(pending.task)
            except TimeoutError:
                error = f"no transcript within {self._stt_timeout} s"
            except Exception as exc:
                error = str(exc) or type(exc).__name__
        return AgentTurn(
            pending.turn, pending.start, pending.end, pending.samples, transcript, error
        )

    async def hangup(self) -> None:
        """Hang up and close everything; same as close()."""
        await self.close()

    async def close(self) -> None:
        """Hang up if the call is still up, stop media and the VAD (a turn
        still open ends at its last speech frame), close the recording,
        and write transcript.json once the transcripts are in. Safe to call
        twice."""
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if self._call is not None:
                    await self._call.hangup()
            finally:
                await self._stop_media()
        finally:
            if self._watcher is not None:
                self._watcher.cancel()
            try:
                self._close_recording()
            finally:
                if self._frozen:
                    self._frozen = False
                    _unfreeze_heap()
        if self.answered and self._stt is not None:
            await self._write_transcript()

    def _stop_media(self) -> asyncio.Task:
        if self._media_stop is None:
            self._media_stop = asyncio.ensure_future(self._do_stop_media())
        return self._media_stop

    async def _do_stop_media(self) -> None:
        try:
            if self._rtp is not None:
                await self._rtp.close()
            if self._t_answered is not None:
                self._duration = media_clock() - self._t_answered
            if self._detector is not None:
                try:
                    await asyncio.to_thread(
                        self._detector.close, DETECTOR_CLOSE_TIMEOUT, finish=True
                    )
                except Exception as exc:
                    if self._vad_error is None:
                        self._on_vad_error(exc)
                # Events the detector handed over while closing run first.
                await asyncio.sleep(0)
                self._rec.event(
                    VAD_STOPPED,
                    vad=self.vad_name,
                    dropped_frames=self._detector.dropped_frames,
                )
            self._end_active_say()
        finally:
            self._media_done = True
            self._wake.set()

    def _end_active_say(self) -> None:
        if self._active_say is None:
            return
        record, done = self._active_say
        self._active_say = None
        sent = self._tap.abort_say()
        if sent is not None:
            record.end = sent[1] + FRAME_SECONDS
            self._rec.event(CALLER_SAY_END, at=record.end, turn=record.turn, interrupted=True)
        if not done.done():
            done.set_exception(CallEnded("the call ended while saying"))

    def _close_recording(self) -> None:
        if self._rec is None:
            return
        self._rec.close()
        self.result = CallResult(
            answered=self.answered,
            status=self.status,
            reason=self.reason,
            duration=self._duration,
            frames_sent=self._rtp.packets_sent if self._rtp is not None else 0,
            frames_received=self._rtp.packets_received if self._rtp is not None else 0,
            first_audio_latency=_first_audio_latency(self.record_dir) if self.answered else None,
            record_dir=self.record_dir,
        )

    # Detector events, on the loop thread.

    def _on_turn_event(self, event: TurnEvent) -> None:
        if event.kind == SPEECH_START:
            self._agent_count += 1
            turn = f"agent-{self._agent_count}"
            self._open_turn = (turn, event.at)
            self._tap.hold_from(event.at - STT_PAD_SECONDS)
            self._rec.event(AGENT_SPEECH_START, at=event.at, turn=turn)
            self._overlap_at_agent_start(turn, event.at)
        elif event.kind == TURN_END and self._open_turn is not None:
            turn, start = self._open_turn
            self._open_turn = None
            self._tap.hold_from(None)
            self._rec.event(AGENT_TURN_END, at=event.at, turn=turn)
            audio_start = start - STT_PAD_SECONDS
            audio = self._tap.cut(audio_start, event.at + STT_PAD_SECONDS)
            pad = round(STT_PAD_SECONDS * SAMPLE_RATE)
            samples = audio[pad : pad + round((event.at - start) * SAMPLE_RATE)]
            task = None
            if self._stt is not None:
                task = asyncio.ensure_future(self._stt.transcribe(audio, SAMPLE_RATE))
            pending = _PendingTurn(turn, start, event.at, samples, audio_start, task)
            self._agent_turns.append(pending)
            self._ready.append(pending)
            self._wake.set()

    def _on_vad_error(self, exc: BaseException) -> None:
        if self._vad_error is not None:
            return
        self._vad_error = exc
        if self._rec is not None:
            self._rec.event(VAD_ERROR, error=str(exc) or type(exc).__name__)
        self._wake.set()

    def _overlap(self, agent_turn: str, say: _Say, at: float, started_by: str) -> None:
        key = (agent_turn, say.turn)
        if key in self._overlaps:
            return
        self._overlaps.add(key)
        self._rec.event(
            OVERLAP, at=at, agent_turn=agent_turn, caller_turn=say.turn, started_by=started_by
        )

    def _overlap_at_agent_start(self, turn: str, at: float) -> None:
        # The speech_start arrives a little after the speech began, so a say
        # that started in between counts as well.
        for say in self._says:
            if say.first is not None and (say.end is None or say.end > at):
                self._overlap(
                    turn, say, max(at, say.first), "agent" if at >= say.first else "caller"
                )

    def _overlap_at_say_start(self, say: _Say) -> None:
        if self._open_turn is not None and self._open_turn[1] <= say.first:
            self._overlap(self._open_turn[0], say, say.first, "caller")

    # transcript.json

    async def _write_transcript(self) -> None:
        tasks = [p.task for p in self._agent_turns if p.task is not None]
        if tasks:
            await asyncio.wait(tasks, timeout=self._stt_timeout)
        t0 = self._rec.t0
        segments: list[Segment] = []
        parts: list[Transcript] = []
        for say in self._says:
            if say.text and say.first is not None and say.end is not None:
                segments.append(
                    Segment(
                        start=say.first - t0,
                        end=say.end - t0,
                        text=say.text,
                        speaker="caller",
                        turn=say.turn,
                    )
                )
        for pending in self._agent_turns:
            task = pending.task
            if task is None or not task.done() or task.cancelled() or task.exception():
                continue
            part = task.result()
            parts.append(part)
            offset = pending.audio_start - t0
            segments.extend(shift_segment(s, offset, "agent", pending.turn) for s in part.segments)
        for task in tasks:
            if not task.done():
                task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, return_exceptions=True)
        segments.sort(key=lambda s: s.start)
        languages = sorted({p.language for p in parts if p.language})
        self.transcript = Transcript(
            segments=segments,
            engine=parts[0].engine if parts else "",
            model=parts[0].model if parts else "",
            language=",".join(languages),
        )
        write_transcript(self.record_dir, self.transcript)
