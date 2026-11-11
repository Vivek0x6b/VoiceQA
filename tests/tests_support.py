"""Helpers shared by tests: the pytest process's child processes and a
one-line description of the machine's state for failure messages."""

import os
import subprocess
import sys
import threading


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
    out = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    ).stdout
    found = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != str(pid):
            continue
        if "resource_tracker" in parts[2] or parts[2].startswith("ps -A"):
            continue
        found.append(f"{parts[0]} {parts[2][:120]}")
    return sorted(found)


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
