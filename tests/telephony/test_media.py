import pytest

from voiceqa.telephony.media import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    LocalMedia,
    PayloadType,
    RemoteMedia,
)


def test_payload_type_numbers():
    assert PayloadType(0) is PayloadType.PCMU
    assert PayloadType(8) is PayloadType.PCMA


def test_frame_constants():
    assert SAMPLE_RATE == 8000
    assert FRAME_SAMPLES == 160


def test_local_media_defaults():
    local = LocalMedia(host="192.168.1.10", port=40000)
    assert local.payload_types == (PayloadType.PCMU, PayloadType.PCMA)
    assert local.ptime_ms == 20


def test_remote_media_fields():
    remote = RemoteMedia(host="203.0.113.5", port=18000, payload_type=PayloadType.PCMA)
    assert remote.ptime_ms == 20
    assert remote.payload_type == 8


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_bad_port_rejected(port):
    with pytest.raises(ValueError, match="port"):
        LocalMedia(host="127.0.0.1", port=port)
    with pytest.raises(ValueError, match="port"):
        RemoteMedia(host="127.0.0.1", port=port, payload_type=PayloadType.PCMU)


def test_empty_payload_types_rejected():
    with pytest.raises(ValueError, match="payload"):
        LocalMedia(host="127.0.0.1", port=40000, payload_types=())


def test_only_20ms_ptime():
    with pytest.raises(ValueError, match="ptime"):
        RemoteMedia(host="127.0.0.1", port=40000, payload_type=PayloadType.PCMU, ptime_ms=30)


def test_media_is_immutable():
    local = LocalMedia(host="127.0.0.1", port=40000)
    with pytest.raises(AttributeError):
        local.port = 1
