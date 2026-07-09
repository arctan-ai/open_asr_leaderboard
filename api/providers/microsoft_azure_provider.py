import io
import json
import os
import threading
import warnings
from typing import Optional

import requests

from . import APIProvider, PermanentError, register
from .streaming_utils import compact_text, pcm16_chunks

MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}


@register("microsoft")
class MicrosoftAzureProvider(APIProvider):
    ENDPOINT = "https://northeurope.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    STREAMING_MODEL = "MAI-Transcribe-1.5"
    STREAMING_TIMEOUT_SECONDS = 300

    # support 26 languages, list of locales in ML benchmark
    # It is Multi-lingual model, can use without specifying the language.
    LOCALE_DICT = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-PT",
    }

    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        api_key = os.getenv("AZURE_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("AZURE_API_KEY environment variable not set")

        locale = self.LOCALE_DICT.get(language, "")
        definition = {
            "locales": [locale],
            "profanityFilterMode": "None",
            "enhancedMode": {
                "enabled": True,
                "task": "transcribe",
            },
        }
        if prompt is not None:
            # E.g., prompt = "Output must be in lexical format."
            definition["enhancedMode"]["prompt"] = [prompt]

        if use_url:
            file_url = sample["row"]["audio"][0]["src"]
            audio_resp = requests.get(file_url, timeout=120)
            audio_resp.raise_for_status()
            audio_data = io.BytesIO(audio_resp.content)
            files = [
                ("definition", (None, json.dumps(definition))),
                ("audio", ("audio.wav", audio_data, "audio/wav")),
            ]
        else:
            mime = MIME_MAP.get(
                os.path.splitext(audio_file_path)[1].lower(), "audio/wav"
            )
            files = [
                ("definition", (None, json.dumps(definition))),
                ("audio", (audio_file_path, open(audio_file_path, "rb"), mime)),
            ]
        resp = requests.post(
            self.ENDPOINT,
            headers={"Ocp-Apim-Subscription-Key": api_key},
            files=files,
            timeout=300,
        )
        if not resp.ok:
            print(f"Azure API error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json().get("combinedPhrases", [{}])[0].get("text", "") or "."

    def transcribe_streaming(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        if model_variant != self.STREAMING_MODEL:
            raise PermanentError(
                "Microsoft streaming only supports "
                f"microsoft/{self.STREAMING_MODEL}; got microsoft/{model_variant}"
            )
        if use_url:
            raise PermanentError(
                "Microsoft streaming provider requires local audio; do not use --use_url"
            )
        if not audio_file_path:
            raise PermanentError(
                "Microsoft streaming provider requires an audio file path"
            )
        if not os.path.isfile(audio_file_path):
            raise PermanentError(f"Audio file not found: {audio_file_path}")
        if prompt is not None:
            warnings.warn(
                "Microsoft MAI-Transcribe-1.5 streaming does not support prompts; "
                "returning Azure display text",
                RuntimeWarning,
                stacklevel=2,
            )

        endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")
        speech_key = os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_API_KEY")
        if not endpoint:
            raise ValueError("AZURE_SPEECH_ENDPOINT environment variable not set")
        if not speech_key or speech_key == "your_api_key":
            raise ValueError(
                "AZURE_SPEECH_KEY environment variable not set "
                "(AZURE_API_KEY is accepted as a fallback)"
            )

        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise ImportError(
                "azure-cognitiveservices-speech is required for Microsoft streaming ASR"
            ) from exc

        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            endpoint=endpoint,
        )
        locale = self.LOCALE_DICT.get(language, language)
        if locale:
            speech_config.speech_recognition_language = locale

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        transcripts = []
        finished = threading.Event()
        cancellation = []

        def on_recognized(event):
            text = getattr(event.result, "text", "")
            if text:
                transcripts.append(text)

        def on_canceled(event):
            details = getattr(event, "error_details", "") or getattr(
                event.result, "text", ""
            )
            cancellation.append(details or "Azure Speech recognition canceled")
            finished.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(lambda event: finished.set())

        started = False
        try:
            recognizer.start_continuous_recognition_async().get()
            started = True
            for chunk in pcm16_chunks(audio_file_path):
                push_stream.write(chunk)
            push_stream.close()

            if not finished.wait(self.STREAMING_TIMEOUT_SECONDS):
                raise PermanentError(
                    "Microsoft streaming recognition timed out after "
                    f"{self.STREAMING_TIMEOUT_SECONDS} seconds"
                )
            if cancellation:
                raise PermanentError(
                    f"Microsoft streaming recognition canceled: {cancellation[0]}"
                )
        finally:
            push_stream.close()
            if started:
                recognizer.stop_continuous_recognition_async().get()

        return compact_text(transcripts) or "."
