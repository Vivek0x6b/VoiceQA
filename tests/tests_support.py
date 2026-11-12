"""Helpers shared by tests: the pytest process's child processes, a check
that a test left nothing running, a one-line description of the machine's
state for failure messages, and a SIP endpoint that rejects every call."""

import asyncio
import contextlib
import gc
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

from voiceqa.telephony.sip.message import build_response, new_tag, parse_message

# Time for children and threads that are already stopping to be reaped.
LEAK_GRACE_SECONDS = 5.0


def _windows_children(pid: int) -> list[str]:
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    found = []
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        ok = kernel32.Process32First(snapshot, ctypes.byref(entry))
        while ok:
            if entry.th32ParentProcessID == pid:
                name = entry.szExeFile.decode(errors="replace")
                found.append(f"{entry.th32ProcessID} {name}")
            ok = kernel32.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return found


def child_processes() -> list[str]:
    """Live child processes of this process, as "pid command". Leaves out
    multiprocessing's resource tracker, which lives as long as the process
    once any multiprocessing lock has been made."""
    pid = os.getpid()
    if sys.platform == "win32":
        return sorted(_windows_children(pid))
    # -ww: without a terminal, ps cuts lines at 80 columns, which hid
    # "resource_tracker" behind a long interpreter path on CI.
    out = subprocess.run(
        ["ps", "-A", "-ww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    ).stdout
    found = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != str(pid):
            continue
        if "resource_tracker" in parts[2] or parts[2].startswith("ps -A"):
            continue
        found.append(f"{parts[0]} {parts[2][:200]}")
    return sorted(found)


def _threads() -> set[str]:
    return {
        f"{t.name} (daemon={t.daemon})"
        for t in threading.enumerate()
        if t is not threading.main_thread() and (not t.daemon or t.name.startswith("voiceqa"))
    }


def _snapshot() -> dict:
    return {
        "child processes": set(child_processes()),
        "multiprocessing children": {p.name for p in multiprocessing.active_children()},
        "threads": _threads(),
        "frozen objects": gc.get_freeze_count(),
    }


def _left(before: dict, after: dict) -> dict:
    left = {}
    for key, value in after.items():
        if isinstance(value, set):
            extra = value - before[key]
            if extra:
                left[key] = sorted(extra)
        elif value > before[key]:
            # Only a rise is ours: the last session's unfreeze also clears
            # objects something else froze earlier in the process.
            left[key] = f"{before[key]} -> {value}"
    return left


@contextlib.contextmanager
def nothing_left_running(grace: float = LEAK_GRACE_SECONDS) -> Iterator[None]:
    """Fails if the body leaves a child process, a multiprocessing child, a
    non-daemon or voiceqa thread, or more frozen GC objects behind."""
    before = _snapshot()
    yield
    deadline = time.monotonic() + grace
    while True:
        left = _left(before, _snapshot())
        if not left or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    assert not left, f"the test left this behind: {left}"


def say_probe(text: str) -> str:
    """macOS only: whether `say` speaks `text` with a named voice, and what
    the default voice setting is, for a failure message. Never raises."""
    from voiceqa.speech.os_tts import SayTts

    parts = []
    try:
        prefs = subprocess.run(
            ["defaults", "read", "com.apple.speech.voice.prefs"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        parts.append(f"voice prefs: {(prefs.stdout or prefs.stderr).strip()[:300]!r}")
    except Exception as exc:
        parts.append(f"voice prefs unreadable: {exc}")
    try:
        tts = SayTts()
        voices = tts.voices()
        voice = next((v for v in ("Samantha", "Alex", "Albert") if v in voices), None)
        voice = voice or (voices[0] if voices else None)
        if voice is None:
            parts.append("no say voices listed")
        else:
            speech = tts.synthesize(text, voice=voice)
            seconds = len(speech.samples) / speech.rate
            parts.append(f"with voice {voice!r}: {seconds:.2f} s of audio")
    except Exception as exc:
        parts.append(f"with a named voice: {exc}")
    return "; ".join(parts)


SAY_AT_START = "not checked"


def check_say_at_start() -> None:
    """macOS only: try `say` once before any test runs, so a later failure
    shows whether `say` already failed then or was broken by earlier tests."""
    global SAY_AT_START
    from voiceqa.speech.os_tts import SayTts

    try:
        speech = SayTts().synthesize("Checking the speech engine.")
        SAY_AT_START = f"{len(speech.samples) / speech.rate:.2f} s of audio"
    except Exception as exc:
        SAY_AT_START = f"failed: {exc}"


def machine_state() -> str:
    """Load, child processes and threads, for a failure message."""
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    load_text = "n/a" if load is None else ", ".join(f"{x:.2f}" for x in load)
    children = child_processes()
    threads = sorted(t.name for t in threading.enumerate())
    return (
        f"say at session start: {SAY_AT_START}; "
        f"load average {load_text}; {len(children)} child processes {children}; "
        f"{len(threads)} threads {threads}"
    )


class _RejectingUas(asyncio.DatagramProtocol):
    """Answers every INVITE with 100 Trying then a final rejection. Kept small
    on purpose: just enough SIP to make place_call raise SipStatusError."""

    def __init__(self, status=486, reason="Busy Here"):
        self.status = status
        self.reason = reason

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        req = parse_message(data)
        if req.method != "INVITE":
            return
        for status, reason in ((100, "Trying"), (self.status, self.reason)):
            resp = build_response(req, status, reason)
            if status > 100:
                to = req.headers.get("To") or ""
                resp.headers.set("To", f"{to};tag={new_tag()}")
            self.transport.sendto(resp.to_bytes(), addr)

    def error_received(self, exc):
        pass


async def start_rejecting_uas(**kw):
    loop = asyncio.get_running_loop()
    _, proto = await loop.create_datagram_endpoint(
        lambda: _RejectingUas(**kw), local_addr=("127.0.0.1", 0)
    )
    return proto
