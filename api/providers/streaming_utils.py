from collections.abc import Iterable
from urllib.parse import urlencode

from . import PermanentError


DEFAULT_STREAMING_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 100


def pcm16_chunks(
    audio_file_path: str,
    sample_rate: int = DEFAULT_STREAMING_SAMPLE_RATE,
    chunk_ms: int = DEFAULT_CHUNK_MS,
) -> Iterable[bytes]:
    import numpy as np
    import soundfile as sf

    audio, actual_sample_rate = sf.read(
        audio_file_path, always_2d=True, dtype="float32"
    )
    if actual_sample_rate != sample_rate:
        raise PermanentError(
            f"Streaming ASR requires {sample_rate} Hz audio; got {actual_sample_rate} Hz"
        )

    mono = audio.mean(axis=1)
    pcm16 = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm16 * np.iinfo(np.int16).max).astype("<i2")

    chunk_frames = max(1, int(sample_rate * chunk_ms / 1000))
    chunk_bytes = chunk_frames * 2
    payload = pcm16.tobytes()
    for start in range(0, len(payload), chunk_bytes):
        chunk = payload[start : start + chunk_bytes]
        if chunk:
            yield chunk


def compact_text(parts: Iterable[str]) -> str:
    return " ".join(" ".join(part for part in parts if part).split())


def build_query_url(base_url: str, params: dict) -> str:
    return f"{base_url}?{urlencode(params)}"


def connect_websocket(url: str, headers: dict | None = None):
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        try:
            from websockets import connect
        except ImportError as exc:
            raise ImportError(
                "websockets is required for streaming ASR. Install with 'pip install websockets'."
            ) from exc

    try:
        return connect(url, additional_headers=headers)
    except TypeError:
        return connect(url, extra_headers=headers)
