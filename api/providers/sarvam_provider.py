import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Optional

import requests

from . import APIProvider, PermanentError, register
from .streaming_utils import build_query_url, connect_websocket


SARVAM_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
SARVAM_STREAMING_ENDPOINT = "wss://api.sarvam.ai/speech-to-text/ws"
DEFAULT_MODEL = "saaras:v3"
SAMPLE_RATE = 16000
STREAMING_CHUNK_BYTES = 4096
STREAMING_TIMEOUT_S = 30
STATIC_MAX_DURATION_S = 30


def _language_code(language: str) -> str:
    return "en-IN" if language == "en" else language


def _validate_model(model_variant: str) -> str:
    model = model_variant or DEFAULT_MODEL
    if model != DEFAULT_MODEL:
        raise PermanentError(
            f"Unsupported Sarvam model '{model}'. Expected '{DEFAULT_MODEL}'."
        )
    return model


def _audio_info(audio_file_path: str):
    import soundfile as sf

    return sf.info(audio_file_path)


def _raise_for_response(response: requests.Response) -> None:
    if response.status_code != 429 and 400 <= response.status_code < 500:
        raise PermanentError(
            f"Sarvam API returned {response.status_code}: {response.text}"
        )
    response.raise_for_status()


async def _transcribe_streaming(
    audio_file_path: str,
    api_key: str,
    model: str,
    language: str,
) -> str:
    params = {
        "model": model,
        "language-code": _language_code(language),
        "mode": "transcribe",
        "sample_rate": str(SAMPLE_RATE),
    }
    transcript = ""

    async with connect_websocket(
        build_query_url(SARVAM_STREAMING_ENDPOINT, params),
        headers={"Api-Subscription-Key": api_key},
    ) as ws:
        with open(audio_file_path, "rb") as audio_file:
            while chunk := audio_file.read(STREAMING_CHUNK_BYTES):
                await ws.send(
                    json.dumps(
                        {
                            "audio": {
                                "data": base64.b64encode(chunk).decode("ascii"),
                                "sample_rate": str(SAMPLE_RATE),
                                "encoding": "audio/wav",
                            }
                        }
                    )
                )

        await ws.send(json.dumps({"type": "flush"}))

        async def receive_messages() -> None:
            nonlocal transcript
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                message_type = message.get("type")
                data = message.get("data") or {}
                if message_type == "data":
                    next_transcript = data.get("transcript", "")
                    if next_transcript:
                        transcript = next_transcript
                        return
                elif message_type == "error":
                    raise PermanentError(
                        f"Sarvam streaming error: {data.get('error', 'unknown')}"
                    )

        try:
            await asyncio.wait_for(receive_messages(), timeout=STREAMING_TIMEOUT_S)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Sarvam streaming response timed out after {STREAMING_TIMEOUT_S}s"
            ) from exc

    return " ".join(transcript.split())


@register("sarvam")
class SarvamProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if use_url:
            raise PermanentError(
                "Sarvam provider requires local audio; do not use --use_url"
            )
        if audio_file_path is None:
            raise PermanentError("Sarvam provider requires an audio file path")

        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("SARVAM_API_KEY environment variable not set")

        info = _audio_info(audio_file_path)
        if info.duration > STATIC_MAX_DURATION_S:
            raise PermanentError(
                "Sarvam static API supports audio up to 30 seconds; use --streaming"
            )

        model = _validate_model(model_variant)
        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(
                SARVAM_ENDPOINT,
                headers={"Api-Subscription-Key": api_key},
                data={
                    "model": model,
                    "mode": "transcribe",
                    "language_code": _language_code(language),
                },
                files={
                    "file": (Path(audio_file_path).name, audio_file, "audio/wav")
                },
                timeout=300,
            )

        _raise_for_response(response)
        return response.json().get("transcript", "") or "."

    def transcribe_streaming(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if use_url:
            raise PermanentError(
                "Sarvam streaming provider requires local audio; do not use --use_url"
            )
        if audio_file_path is None:
            raise PermanentError(
                "Sarvam streaming provider requires an audio file path"
            )

        info = _audio_info(audio_file_path)
        if info.samplerate != SAMPLE_RATE:
            raise PermanentError(
                f"Sarvam streaming requires {SAMPLE_RATE} Hz audio; "
                f"got {info.samplerate} Hz"
            )

        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("SARVAM_API_KEY environment variable not set")

        model = _validate_model(model_variant)
        return (
            asyncio.run(
                _transcribe_streaming(
                    audio_file_path=audio_file_path,
                    api_key=api_key,
                    model=model,
                    language=language,
                )
            )
            or "."
        )
