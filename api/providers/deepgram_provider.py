import asyncio
import json
import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register
from .streaming_utils import build_query_url, compact_text, connect_websocket, pcm16_chunks


DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"
DEEPGRAM_STREAMING_ENDPOINT = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"


async def _transcribe_streaming(
    audio_file_path: str,
    api_key: str,
    model: str,
    language: str,
) -> str:
    params = {
        "model": model,
        "smart_format": "true",
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
    }
    if language:
        params["language"] = language

    transcripts = []
    async with connect_websocket(
        build_query_url(DEEPGRAM_STREAMING_ENDPOINT, params),
        headers={"Authorization": f"Token {api_key}"},
    ) as ws:
        for chunk in pcm16_chunks(audio_file_path):
            await ws.send(chunk)
        await ws.send(json.dumps({"type": "CloseStream"}))

        async for message in ws:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "Metadata":
                break
            if not data.get("is_final"):
                continue

            alternatives = data.get("channel", {}).get("alternatives", [])
            if alternatives:
                transcripts.append(alternatives[0].get("transcript", ""))

    return compact_text(transcripts)


@register("deepgram")
class DeepgramProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError(
                "DEEPGRAM_API_KEY environment variable not set, get your key at https://console.deepgram.com"
            )

        model = model_variant or DEFAULT_MODEL
        params = {
            "model": model,
            "smart_format": "true",
        }
        if language:
            params["language"] = language

        headers = {"Authorization": f"Token {api_key}"}
        if use_url:
            audio_url = sample["row"]["audio"][0]["src"]
            response = requests.post(
                DEEPGRAM_ENDPOINT,
                headers={**headers, "Content-Type": "application/json"},
                params=params,
                json={"url": audio_url},
                timeout=300,
            )
        else:
            if audio_file_path is None:
                raise PermanentError("Deepgram provider requires an audio file path")
            with open(audio_file_path, "rb") as audio_file:
                response = requests.post(
                    DEEPGRAM_ENDPOINT,
                    headers={**headers, "Content-Type": "audio/wav"},
                    params=params,
                    data=audio_file,
                    timeout=300,
                )

        if response.status_code != 429 and 400 <= response.status_code < 500:
            raise PermanentError(
                f"Deepgram API returned {response.status_code}: {response.text}"
            )
        response.raise_for_status()

        channels = response.json().get("results", {}).get("channels", [])
        if not channels:
            return "."
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return "."
        return alternatives[0].get("transcript", "") or "."

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
                "Deepgram streaming provider requires local audio; do not use --use_url"
            )
        if audio_file_path is None:
            raise PermanentError("Deepgram streaming provider requires an audio file path")

        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError(
                "DEEPGRAM_API_KEY environment variable not set, get your key at https://console.deepgram.com"
            )

        model = model_variant or DEFAULT_MODEL
        return asyncio.run(
            _transcribe_streaming(
                audio_file_path=audio_file_path,
                api_key=api_key,
                model=model,
                language=language,
            )
        ) or "."
