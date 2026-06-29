import asyncio
import json
import os
from typing import Optional

import assemblyai as aai

from . import APIProvider, PermanentError, register
from .streaming_utils import build_query_url, compact_text, connect_websocket, pcm16_chunks


ASSEMBLY_STREAMING_ENDPOINT = "wss://streaming.assemblyai.com/v3/ws"
STREAMING_MODEL_MAP = {"universal-3-pro": "universal-3-5-pro"}
STREAMING_CHUNK_MS = 50
TERMINATION_TIMEOUT_S = 30


async def _transcribe_streaming(
    audio_file_path: str,
    api_key: str,
    model: str,
) -> str:
    transcripts = []
    url = build_query_url(
        ASSEMBLY_STREAMING_ENDPOINT,
        {"sample_rate": "16000", "speech_model": model},
    )

    async with connect_websocket(url, headers={"Authorization": api_key}) as ws:
        async def receive_messages():
            async for message in ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "Termination":
                    return
                if data.get("type") != "Turn" or not data.get("end_of_turn"):
                    continue

                transcript = data.get("transcript", "")
                if transcript:
                    transcripts.append(transcript)

        receiver = asyncio.create_task(receive_messages())
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
        finally:
            if not receiver.done():
                receiver.cancel()
            await ws.close()

    return compact_text(transcripts)


@register("assembly")
class AssemblyAIProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        transcriber = aai.Transcriber()

        # Models like "universal-3-pro" use the newer speech_models (list) API
        MULTI_MODEL_VARIANTS = {"universal-3-pro"}
        if model_variant in MULTI_MODEL_VARIANTS:
            config = aai.TranscriptionConfig(
                speech_models=[model_variant],
                language_code=language,
            )
        else:
            config = aai.TranscriptionConfig(
                speech_model=model_variant,
                language_code=language,
            )

        if use_url:
            audio_url = sample["row"]["audio"][0]["src"]
            audio_duration = sample["row"]["audio_length_s"]
            if audio_duration < 0.160:
                print(f"Skipping audio duration {audio_duration}s")
                return "."
            transcript = transcriber.transcribe(audio_url, config=config)
        else:
            audio_duration = (
                len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            )
            if audio_duration < 0.160:
                print(f"Skipping audio duration {audio_duration}s")
                return "."
            transcript = transcriber.transcribe(audio_file_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            raise PermanentError(f"AssemblyAI transcription error: {transcript.error}")
        return transcript.text or ""

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
        ) or "."
