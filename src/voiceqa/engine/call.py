"""Running one outbound call: place it, send audio, record both sides.

Wires together SIP (telephony.sip), RTP (telephony.rtp) and recording
(telephony.recording) for the `voiceqa call` command. voiceqa calibrate
(a later command) reuses run_call to send a pulse train through the same
flow instead of duplicating it.
"""

import asyncio
import contextlib
import json
import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceqa.audio.analysis import first_loud_frame
from voiceqa.telephony.media import SAMPLE_RATE, PayloadType, media_clock
from voiceqa.telephony.recording import CallRecorder
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip import SipConfig, SipError, SipStatusError, place_call
from voiceqa.telephony.sip.transport import routing_ip


@dataclass(frozen=True)
class CallResult:
    answered: bool
    status: int | None
    reason: str | None
    duration: float | None
    frames_sent: int
    frames_received: int
    first_audio_latency: float | None
    record_dir: Path


async def _resolve(host: str, port: int) -> tuple[str, int]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_DGRAM)
    return infos[0][4][0], infos[0][4][1]


def _answered_at(record_dir: Path) -> float | None:
    data = json.loads((record_dir / "events.json").read_text())
    for event in data["events"]:
        if event["event"] == "answered":
            return event["t"]
    return None


def _first_audio_latency(record_dir: Path) -> float | None:
    answered_t = _answered_at(record_dir)
    if answered_t is None:
        return None
    index = first_loud_frame(record_dir / "agent.wav")
    if index is None:
        return None
    return max(0.0, index / SAMPLE_RATE - answered_t)


async def run_call(
    target: str,
    server_host: str,
    server_port: int,
    username: str,
    password: str,
    audio: np.ndarray,
    record_dir: str | Path,
    duration: float = 10.0,
    codec: PayloadType = PayloadType.PCMU,
) -> CallResult:
    """Call target through server_host:server_port and record it to record_dir.

    audio (int16, 8 kHz, mono) is queued once the call is answered; the CLI
    passes either a loaded WAV or a default tone. Hangs up duration seconds
    after the answer, or sooner if the far end hangs up first. Always closes
    the recorder, even if this is cancelled (Ctrl+C) partway through.

    A call that cannot be set up (the server host does not resolve, the far
    end answers with no usable codec, or any other SIP failure) comes back
    as a CallResult with answered=False and a short reason, not an
    exception: there is no good way to place this call, but that is not a
    bug, and the recording still needs to be closed and reported.
    """
    record_dir = Path(record_dir)
    rec = CallRecorder(record_dir)
    answered = False
    status: int | None = None
    reason: str | None = None
    call_duration: float | None = None
    frames_sent = 0
    frames_received = 0
    try:
        try:
            resolved = await _resolve(server_host, server_port)
        except OSError as exc:
            reason = f"cannot resolve {server_host}: {exc}"
        else:
            advertise_host = routing_ip(resolved)
            rtp = await RtpSession.open(
                bind_host="0.0.0.0", advertise_host=advertise_host, payload_types=(codec,)
            )
            try:
                rtp.record(rec)
                sip_config = SipConfig(
                    server_host=server_host,
                    server_port=server_port,
                    username=username,
                    password=password,
                )
                try:
                    call = await place_call(sip_config, target, rtp.local, on_event=rec.event)
                except SipStatusError as exc:
                    status, reason = exc.status, exc.reason
                except SipError as exc:
                    reason = str(exc)
                else:
                    answered = True
                    status, reason = 200, "OK"
                    t_answered = media_clock()
                    try:
                        await rtp.start(call.remote_media)
                        rtp.send(audio)
                        with contextlib.suppress(TimeoutError):
                            async with asyncio.timeout(duration):
                                await call.wait_ended()
                    finally:
                        await call.hangup()
                    call_duration = media_clock() - t_answered
                frames_sent = rtp.packets_sent
                frames_received = rtp.packets_received
            finally:
                await rtp.close()
    finally:
        rec.close()

    return CallResult(
        answered=answered,
        status=status,
        reason=reason,
        duration=call_duration,
        frames_sent=frames_sent,
        frames_received=frames_received,
        first_audio_latency=_first_audio_latency(record_dir) if answered else None,
        record_dir=record_dir,
    )
