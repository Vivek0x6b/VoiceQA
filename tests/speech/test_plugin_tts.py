import json
import os
import stat
import sys
import time

import numpy as np
import pytest

from voiceqa.speech.plugin_tts import PluginTts
from voiceqa.speech.registry import create_tts
from voiceqa.speech.tts_process import TtsError

# The fake plugin records the request it got next to its output, then
# behaves according to MODE: write a tone, fail with a message, hang, or
# exit 0 without writing anything.
PLUGIN_SOURCE = """\
import json
import math
import os
import struct
import sys
import time
import wave

MODE = {mode!r}
args = sys.argv[1:]
out = args[args.index("--out") + 1]
request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
if os.environ.get("FAKE_PLUGIN_LOG"):
    with open(os.environ["FAKE_PLUGIN_LOG"], "w", encoding="utf-8") as f:
        json.dump({{"request": request, "args": args}}, f)
if MODE == "fail":
    sys.stderr.write("no such voice: " + str(request["voice"]) + "\\n")
    sys.exit(3)
if MODE == "slow":
    time.sleep(4)
    open(os.environ["FAKE_PLUGIN_SURVIVED"], "w").close()
if MODE == "silent-exit":
    sys.exit(0)
rate = 16000
n = int(rate * 0.01 * len(request["text"]))
frames = b"".join(
    struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)
)
with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(frames)
"""


def install_plugin(folder, name, mode):
    folder.mkdir(exist_ok=True)
    script = folder / f"{name}_plugin.py"
    script.write_text(PLUGIN_SOURCE.format(mode=mode), encoding="utf-8")
    if sys.platform == "win32":
        launcher = folder / f"voiceqa-tts-{name}.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = folder / f"voiceqa-tts-{name}"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher


@pytest.fixture
def plugin_dir(tmp_path, monkeypatch):
    folder = tmp_path / "bin"
    folder.mkdir()
    monkeypatch.setenv("PATH", str(folder) + os.pathsep + os.environ.get("PATH", ""))
    return folder


@pytest.mark.realtime
def test_plugin_success_returns_its_wav_as_speech(plugin_dir):
    install_plugin(plugin_dir, "tone", "ok")
    tts = PluginTts(plugin="tone")

    speech = tts.synthesize("hello there", voice="anna")

    assert tts.name == "plugin:tone"
    assert speech.rate == 16000
    assert len(speech.samples) == int(16000 * 0.01 * len("hello there"))
    assert np.abs(speech.samples).max() > 5000


@pytest.mark.realtime
def test_plugin_gets_text_and_voice_as_json_on_stdin(plugin_dir, monkeypatch, tmp_path):
    install_plugin(plugin_dir, "tone", "ok")
    log = tmp_path / "request.json"
    monkeypatch.setenv("FAKE_PLUGIN_LOG", str(log))
    text = 'say "hi" $HOME `whoami` %PATH% & echo\nline two caf' + chr(0xE9)

    PluginTts(plugin="tone").synthesize(text, voice=None)

    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["request"] == {"text": text, "voice": None}
    assert data["args"][0] == "--out"
    assert data["args"][1].endswith(".wav")


@pytest.mark.realtime
def test_plugin_failure_shows_exit_code_and_stderr(plugin_dir):
    install_plugin(plugin_dir, "broken", "fail")
    with pytest.raises(TtsError) as exc_info:
        PluginTts(plugin="broken").synthesize("hello", voice="nobody")
    message = str(exc_info.value)
    assert "voiceqa-tts-broken" in message
    assert "exit code 3" in message
    assert "no such voice: nobody" in message


@pytest.mark.realtime
def test_plugin_timeout_kills_it_and_says_so(plugin_dir, tmp_path, monkeypatch):
    install_plugin(plugin_dir, "slow", "slow")
    survived = tmp_path / "survived"
    monkeypatch.setenv("FAKE_PLUGIN_SURVIVED", str(survived))
    start = time.monotonic()
    with pytest.raises(TtsError, match="timed out after 1 s"):
        PluginTts(plugin="slow", timeout=1.0).synthesize("hello")
    # The plugin writes `survived` after sleeping 4 s. Waiting well past
    # that and finding no file shows the whole plugin was killed, including
    # the python child of the .cmd wrapper on Windows.
    time.sleep(max(0.0, start + 8 - time.monotonic()))
    assert not survived.exists()


@pytest.mark.realtime
def test_plugin_that_writes_no_wav_is_an_error(plugin_dir):
    install_plugin(plugin_dir, "lazy", "silent-exit")
    with pytest.raises(TtsError, match="without writing"):
        PluginTts(plugin="lazy").synthesize("hello")


@pytest.mark.realtime
def test_missing_plugin_names_the_executable(plugin_dir):
    with pytest.raises(TtsError, match="voiceqa-tts-nothere"):
        PluginTts(plugin="nothere").synthesize("hello")


@pytest.mark.parametrize("bad", ["", "a/b", "a b", "..\\x", "x;y"])
def test_plugin_names_are_checked(bad):
    with pytest.raises(ValueError, match="bad plugin name"):
        PluginTts(plugin=bad)


def test_registry_builds_plugin_engines(plugin_dir):
    install_plugin(plugin_dir, "tone", "ok")
    tts = create_tts("plugin:tone", timeout=5.0)
    assert isinstance(tts, PluginTts)
    assert tts.synthesize("abc").rate == 16000


def test_plugin_has_no_voice_list():
    assert PluginTts(plugin="any").voices() == []
