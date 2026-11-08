"""G.711 PCMU (mu-law) and PCMA (A-law) codecs.

Bit-exact with the Sun reference implementation (g711.c) that the old
stdlib audioop module used. Encoding and decoding are table lookups; the
tables are built once at import.
"""

import numpy as np
import numpy.typing as npt

_ALL_INT16 = np.arange(-32768, 32768, dtype=np.int32)
_ALL_CODES = np.arange(256, dtype=np.int32)

_PCMU_BIAS = 0x84
_PCMU_CLIP = 8159
_PCMU_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def _build_pcmu_encode() -> np.ndarray:
    x = _ALL_INT16 >> 2
    mask = np.where(x < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(x), _PCMU_CLIP) + (_PCMU_BIAS >> 2)
    seg = np.searchsorted(_PCMU_SEG_END, mag)
    code = np.where(seg >= 8, 0x7F, (seg << 4) | ((mag >> (seg + 1)) & 0x0F))
    return (code ^ mask).astype(np.uint8)


def _build_pcmu_decode() -> np.ndarray:
    u = ~_ALL_CODES & 0xFF
    t = (((u & 0x0F) << 3) + _PCMU_BIAS) << ((u & 0x70) >> 4)
    return np.where(u & 0x80, _PCMU_BIAS - t, t - _PCMU_BIAS).astype(np.int16)


_PCMA_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])


def _build_pcma_encode() -> np.ndarray:
    x = _ALL_INT16 >> 3
    mask = np.where(x >= 0, 0xD5, 0x55)
    mag = np.where(x >= 0, x, -x - 1)
    seg = np.searchsorted(_PCMA_SEG_END, mag)
    shift = np.maximum(seg, 1)
    code = np.where(seg >= 8, 0x7F, (seg << 4) | ((mag >> shift) & 0x0F))
    return (code ^ mask).astype(np.uint8)


def _build_pcma_decode() -> np.ndarray:
    a = _ALL_CODES ^ 0x55
    q = a & 0x0F
    seg = (a & 0x70) >> 4
    t = np.where(seg > 0, (2 * q + 33) << (seg + 2), (2 * q + 1) << 3)
    return np.where(a & 0x80, t, -t).astype(np.int16)


_PCMU_ENC = _build_pcmu_encode()
_PCMU_DEC = _build_pcmu_decode()
_PCMA_ENC = _build_pcma_encode()
_PCMA_DEC = _build_pcma_decode()


def _table_index(samples: npt.ArrayLike) -> np.ndarray:
    x = np.asarray(samples)
    if x.size == 0:
        return np.zeros(0, dtype=np.int32)
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"G.711 encoding needs integer samples, got {x.dtype}")
    return np.clip(x.ravel().astype(np.int64), -32768, 32767) + 32768


def encode_pcmu(samples: npt.ArrayLike) -> bytes:
    return _PCMU_ENC[_table_index(samples)].tobytes()


def decode_pcmu(data: bytes | bytearray | memoryview) -> np.ndarray:
    return _PCMU_DEC[np.frombuffer(data, dtype=np.uint8)]


def encode_pcma(samples: npt.ArrayLike) -> bytes:
    return _PCMA_ENC[_table_index(samples)].tobytes()


def decode_pcma(data: bytes | bytearray | memoryview) -> np.ndarray:
    return _PCMA_DEC[np.frombuffer(data, dtype=np.uint8)]
