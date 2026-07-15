import asyncio
import json
import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register
from .streaming_utils import (
    build_query_url,
    compact_text,
    connect_websocket,
    pcm16_chunks,
)


CARTESIA_STT_ENDPOINT = "https://api.cartesia.ai/stt"
CARTESIA_WS_ENDPOINT = "wss://api.cartesia.ai/stt/turns/websocket"
CARTESIA_API_VERSION = "2026-03-01"
DEFAULT_STATIC_MODEL = "ink-whisper"
DEFAULT_STREAMING_MODEL = "ink-2"


async def _transcribe_streaming(
    audio_file_path: str,
    api_key: str,
    model: str,
) -> str:
    params = {
        "model": model,
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "cartesia_version": CARTESIA_API_VERSION,
    }

    transcripts = []
    async with connect_websocket(
        build_query_url(CARTESIA_WS_ENDPOINT, params),
        headers={"X-API-Key": api_key},
    ) as ws:
        for chunk in pcm16_chunks(audio_file_path):
            await ws.send(chunk)
        await ws.send(json.dumps({"type": "close"}))

        async for message in ws:
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                continue

            event_type = data.get("type", "")
            if event_type == "turn.end":
                transcript = data.get("transcript", "")
                if transcript:
                    transcripts.append(transcript)
            elif event_type == "error":
                raise RuntimeError(f"Cartesia streaming error: {data.get('message', data)}")

    return compact_text(transcripts)


@register("cartesia")
class CartesiaProvider(APIProvider):
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
                "Cartesia static provider requires a local audio file; do not use --use_url"
            )
        if audio_file_path is None:
            raise PermanentError("Cartesia static provider requires an audio file path")

        api_key = os.getenv("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError(
                "CARTESIA_API_KEY environment variable not set, get your key at https://play.cartesia.ai"
            )

        model = model_variant or DEFAULT_STATIC_MODEL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Cartesia-Version": CARTESIA_API_VERSION,
        }

        form_data = {"model": model}
        if language != "unknown":
            form_data["language"] = language

        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(
                CARTESIA_STT_ENDPOINT,
                headers=headers,
                data=form_data,
                files={"file": audio_file},
                timeout=300,
            )

        if response.status_code != 429 and 400 <= response.status_code < 500:
            raise PermanentError(
                f"Cartesia API returned {response.status_code}: {response.text}"
            )
        response.raise_for_status()

        return response.json().get("text", "") or "."

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
                "Cartesia streaming provider requires local audio; do not use --use_url"
            )
        if audio_file_path is None:
            raise PermanentError(
                "Cartesia streaming provider requires an audio file path"
            )

        api_key = os.getenv("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError(
                "CARTESIA_API_KEY environment variable not set, get your key at https://play.cartesia.ai"
            )

        model = model_variant or DEFAULT_STREAMING_MODEL
        return (
            asyncio.run(
                _transcribe_streaming(
                    audio_file_path=audio_file_path,
                    api_key=api_key,
                    model=model,
                )
            )
            or "."
        )
