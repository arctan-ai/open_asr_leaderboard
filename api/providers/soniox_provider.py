import os
import time
from typing import Optional

import requests

from . import APIProvider, PermanentError, register


SONIOX_API_BASE_URL = "https://api.soniox.com"
DEFAULT_MODEL = "stt-async-v5"
POLL_INTERVAL_S = 1
POLL_TIMEOUT_S = 600


def _render_tokens(tokens: list[dict]) -> str:
    text = "".join(
        str(token.get("text", ""))
        for token in tokens
        if str(token.get("text", "")) and str(token.get("text", "")) != "<end>"
    )
    return " ".join(text.split())


def _raise_for_permanent_client_error(response: requests.Response) -> None:
    if response.status_code != 429 and 400 <= response.status_code < 500:
        raise PermanentError(
            f"Soniox API returned {response.status_code}: {response.text}"
        )
    response.raise_for_status()


def _upload_file(session: requests.Session, audio_file_path: str) -> str:
    with open(audio_file_path, "rb") as audio_file:
        response = session.post(
            f"{SONIOX_API_BASE_URL}/v1/files",
            files={"file": audio_file},
            timeout=300,
        )
    _raise_for_permanent_client_error(response)
    return response.json()["id"]


def _create_transcription(
    session: requests.Session,
    model: str,
    file_id: str,
    language: str,
) -> str:
    response = session.post(
        f"{SONIOX_API_BASE_URL}/v1/transcriptions",
        json={
            "model": model,
            "language_hints": [language],
            "file_id": file_id,
        },
        timeout=60,
    )
    _raise_for_permanent_client_error(response)
    return response.json()["id"]


def _wait_until_completed(session: requests.Session, transcription_id: str) -> None:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        response = session.get(
            f"{SONIOX_API_BASE_URL}/v1/transcriptions/{transcription_id}",
            timeout=60,
        )
        _raise_for_permanent_client_error(response)
        data = response.json()
        status = data.get("status")
        if status == "completed":
            return
        if status == "error":
            raise PermanentError(
                f"Soniox transcription error: {data.get('error_message', 'unknown')}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Soniox transcription timed out after {POLL_TIMEOUT_S}s"
            )
        time.sleep(POLL_INTERVAL_S)


def _get_transcript(session: requests.Session, transcription_id: str) -> str:
    response = session.get(
        f"{SONIOX_API_BASE_URL}/v1/transcriptions/{transcription_id}/transcript",
        timeout=60,
    )
    _raise_for_permanent_client_error(response)
    return _render_tokens(response.json().get("tokens", []))


def _delete_transcription(session: requests.Session, transcription_id: str) -> None:
    response = session.delete(
        f"{SONIOX_API_BASE_URL}/v1/transcriptions/{transcription_id}",
        timeout=60,
    )
    _raise_for_permanent_client_error(response)


def _delete_file(session: requests.Session, file_id: str) -> None:
    response = session.delete(
        f"{SONIOX_API_BASE_URL}/v1/files/{file_id}",
        timeout=60,
    )
    _raise_for_permanent_client_error(response)


@register("soniox")
class SonioxProvider(APIProvider):
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
                "Soniox provider requires local audio; do not use --use_url"
            )
        if audio_file_path is None:
            raise PermanentError("Soniox provider requires an audio file path")

        api_key = os.getenv("SONIOX_API_KEY")
        if not api_key or api_key == "your_api_key":
            raise ValueError("SONIOX_API_KEY environment variable not set")

        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {api_key}"

        model = model_variant or DEFAULT_MODEL
        file_id = None
        transcription_id = None
        try:
            file_id = _upload_file(session, audio_file_path)
            transcription_id = _create_transcription(
                session=session,
                model=model,
                file_id=file_id,
                language=language,
            )
            _wait_until_completed(session, transcription_id)
            return _get_transcript(session, transcription_id) or "."
        finally:
            if transcription_id is not None:
                try:
                    _delete_transcription(session, transcription_id)
                except Exception as exc:
                    print(f"Warning: failed to delete Soniox transcription: {exc}")
            if file_id is not None:
                try:
                    _delete_file(session, file_id)
                except Exception as exc:
                    print(f"Warning: failed to delete Soniox file: {exc}")
