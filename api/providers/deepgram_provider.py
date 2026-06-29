import os
from typing import Optional

import requests

from . import APIProvider, PermanentError, register


DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"


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
