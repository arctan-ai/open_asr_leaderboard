import asyncio
import json
import os
from typing import Optional

import assemblyai as aai

from . import APIProvider, PermanentError, ProviderTranscription, register
from .streaming_utils import (
    build_query_url,
    compact_text,
    connect_websocket,
    pcm16_chunks,
)


ASSEMBLY_STREAMING_ENDPOINT = "wss://streaming.assemblyai.com/v3/ws"
ASSEMBLY_ALIAS = "universal-stt"
LEGACY_ASSEMBLY_ALIAS = "universal-3-pro"
PRIMARY_MODEL = "universal-3-5-pro"
FALLBACK_MODEL = "universal-2"
STREAMING_MODEL_MAP = {
    ASSEMBLY_ALIAS: PRIMARY_MODEL,
    LEGACY_ASSEMBLY_ALIAS: PRIMARY_MODEL,
}
STREAMING_CHUNK_MS = 50
TERMINATION_TIMEOUT_S = 30


async def _transcribe_streaming(
    audio_file_path: str,
    api_key: str,
    model: str,
) -> ProviderTranscription:
    transcripts = []
    actual_model = model
    detected_languages: set[str] = set()
    url = build_query_url(
        ASSEMBLY_STREAMING_ENDPOINT,
        {
            "sample_rate": "16000",
            "speech_model": model,
            "language_detection": "true",
        },
    )

    async with connect_websocket(url, headers={"Authorization": api_key}) as ws:

        async def receive_messages():
            async for message in ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                message_type = data.get("type")
                if message_type == "Error":
                    raise PermanentError(
                        f"AssemblyAI streaming error {data.get('error_code', 'unknown')}: "
                        f"{data.get('error', 'unknown')}"
                    )
                if message_type == "Begin":
                    nonlocal actual_model
                    actual_model = (
                        (data.get("configuration") or {}).get("model") or actual_model
                    )
                    continue
                if message_type == "Termination":
                    return
                if message_type != "Turn" or not data.get("end_of_turn"):
                    continue

                language_code = data.get("language_code")
                if language_code:
                    detected_languages.add(str(language_code))
                for word in data.get("words", []):
                    word_language = word.get("language_code") or word.get("language")
                    if word_language:
                        detected_languages.add(str(word_language))

                transcript = data.get("transcript", "")
                if transcript:
                    transcripts.append(transcript)

        receiver = asyncio.create_task(receive_messages())
        send_error: BaseException | None = None
        try:
            for chunk in pcm16_chunks(audio_file_path, chunk_ms=STREAMING_CHUNK_MS):
                await ws.send(chunk)
                await asyncio.sleep(STREAMING_CHUNK_MS / 1000)
            await ws.send(json.dumps({"type": "Terminate"}))
            try:
                await asyncio.wait_for(receiver, timeout=TERMINATION_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"AssemblyAI streaming termination timed out after "
                    f"{TERMINATION_TIMEOUT_S}s"
                ) from exc
        except BaseException as exc:
            send_error = exc
        finally:
            if not receiver.done():
                receiver.cancel()
            receiver_error: BaseException | None = None
            try:
                await receiver
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                receiver_error = exc
            await ws.close()

        if receiver_error is not None:
            raise receiver_error
        if send_error is not None:
            raise send_error

    return ProviderTranscription(
        text=compact_text(transcripts),
        actual_model=f"assembly/{actual_model}",
        detected_languages=tuple(sorted(detected_languages)),
    )


@register("assembly")
class AssemblyAIProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> ProviderTranscription:
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        transcriber = aai.Transcriber()

        if model_variant in {ASSEMBLY_ALIAS, LEGACY_ASSEMBLY_ALIAS}:
            speech_models = [PRIMARY_MODEL, FALLBACK_MODEL]
        else:
            speech_models = [model_variant]
        language_config = (
            {"language_detection": True}
            if language == "unknown"
            else {"language_code": language}
        )
        config = aai.TranscriptionConfig(
            speech_models=speech_models,
            **language_config,
        )

        if use_url:
            audio_url = sample["row"]["audio"][0]["src"]
            audio_duration = sample["row"]["audio_length_s"]
            if audio_duration < 0.160:
                print(f"Skipping audio duration {audio_duration}s")
                return ProviderTranscription(
                    text="",
                    actual_model=f"assembly/{speech_models[0]}",
                )
            transcript = transcriber.transcribe(audio_url, config=config)
        else:
            audio_duration = (
                len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            )
            if audio_duration < 0.160:
                print(f"Skipping audio duration {audio_duration}s")
                return ProviderTranscription(
                    text="",
                    actual_model=f"assembly/{speech_models[0]}",
                )
            transcript = transcriber.transcribe(audio_file_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            raise PermanentError(f"AssemblyAI transcription error: {transcript.error}")
        response = getattr(transcript, "json_response", None) or {}
        actual_model = response.get("speech_model_used") or speech_models[0]
        detected_language = (
            response.get("language_code") or getattr(transcript, "language_code", None)
        )
        return ProviderTranscription(
            text=transcript.text or "",
            actual_model=f"assembly/{actual_model}",
            detected_languages=(str(detected_language),) if detected_language else (),
        )

    def transcribe_streaming(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> ProviderTranscription:
        if use_url:
            raise PermanentError(
                "AssemblyAI streaming provider requires local audio; do not use --use_url"
            )
        if audio_file_path is None:
            raise PermanentError(
                "AssemblyAI streaming provider requires an audio file path"
            )

        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")

        model = STREAMING_MODEL_MAP.get(model_variant, model_variant)
        return asyncio.run(
            _transcribe_streaming(
                audio_file_path=audio_file_path,
                api_key=api_key,
                model=model,
            )
        )
