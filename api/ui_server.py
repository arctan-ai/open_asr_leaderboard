from __future__ import annotations
import asyncio
import json
import os
import secrets
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.sessions import SessionMiddleware

from api.dataset_catalog import (
    DatasetValidationError,
    dataset_catalog,
    source_options,
    validate_dataset_selection,
)
from api.language_catalog import MODEL_LANGUAGES, effective_mode, language_options

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "ui-runs"
DEFAULT_WEB_DIST = REPO_ROOT / "web" / "dist"
RUNNER_PATH = REPO_ROOT / "api" / "run_eval.py"
TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
SESSION_MAX_AGE_SECONDS = 8 * 60 * 60

PROVIDERS = {
    "assembly": {
        "label": "AssemblyAI",
        "models": ["universal-stt"],
        "credentials": ["ASSEMBLYAI_API_KEY"],
    },
    "cartesia": {
        "label": "Cartesia",
        "models": ["ink-whisper", "ink-2"],
        "credentials": ["CARTESIA_API_KEY"],
    },
    "deepgram": {
        "label": "Deepgram",
        "models": ["nova-3"],
        "credentials": ["DEEPGRAM_API_KEY"],
    },
    "microsoft": {
        "label": "Microsoft Azure",
        "models": ["azure-speech-05-2026", "MAI-Transcribe-1.5"],
        "credentials": ["AZURE_API_KEY"],
    },
    "sarvam": {
        "label": "Sarvam",
        "models": ["saaras:v3"],
        "credentials": ["SARVAM_API_KEY"],
    },
    "smallestai": {
        "label": "Smallest AI",
        "models": ["pulse"],
        "credentials": ["SMALLESTAI_API_KEY"],
    },
    "soniox": {
        "label": "Soniox",
        "models": ["stt-async-v5", "stt-rt-v5"],
        "credentials": ["SONIOX_API_KEY"],
    },
}

PREPROCESSORS = ["none", "arctan", "ai_coustics_vfl_2_1", "krisp_bvc", "rnnoise"]
VAD_POSITIONS = ["none", "pre", "post"]

load_dotenv(REPO_ROOT / ".env")
PREPROCESSOR_CREDENTIALS = {
    "arctan": ["ARCTAN_SDK_KEY"],
    "ai_coustics_vfl_2_1": [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ],
    "krisp_bvc": ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"],
}


@dataclass(frozen=True)
class AuthSettings:
    client_id: str = ""
    client_secret: str = ""
    session_secret: str = ""
    public_url: str = ""
    google_domain: str = "arctan.ai"
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "AuthSettings":
        return cls(
            client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip(),
            client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip(),
            session_secret=os.environ.get("OPEN_ASR_SESSION_SECRET", "").strip(),
            public_url=os.environ.get("OPEN_ASR_PUBLIC_URL", "").strip().rstrip("/"),
            google_domain=(
                os.environ.get("OPEN_ASR_GOOGLE_DOMAIN", "arctan.ai").strip().lower()
                or "arctan.ai"
            ),
        )

    @classmethod
    def disabled_for_tests(cls) -> "AuthSettings":
        return cls(enabled=False)

    @property
    def missing(self) -> list[str]:
        values = {
            "GOOGLE_OAUTH_CLIENT_ID": self.client_id,
            "GOOGLE_OAUTH_CLIENT_SECRET": self.client_secret,
            "OPEN_ASR_SESSION_SECRET": self.session_secret,
            "OPEN_ASR_PUBLIC_URL": self.public_url,
        }
        missing = [name for name, value in values.items() if not value]
        if self.session_secret and len(self.session_secret) < 32:
            missing.append("OPEN_ASR_SESSION_SECRET (must be at least 32 characters)")
        if self.public_url:
            parsed = urlsplit(self.public_url)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                missing.append("OPEN_ASR_PUBLIC_URL (must be an HTTPS origin)")
        return missing

    @property
    def configured(self) -> bool:
        return not self.missing

    @property
    def callback_url(self) -> str:
        return f"{self.public_url}/auth/google/callback"


class ConsoleAuthMiddleware:
    def __init__(self, app, settings: AuthSettings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        path = request.url.path
        if not self.settings.enabled or path in {
            "/healthz",
            "/auth/google/login",
            "/auth/google/callback",
        }:
            await self.app(scope, receive, send)
            return
        if not self.settings.configured:
            response = JSONResponse(
                {
                    "detail": "Google authentication is not configured",
                    "missing": self.settings.missing,
                },
                status_code=503,
            )
        elif request.session.get("user"):
            await self.app(scope, receive, send)
            return
        elif path == "/api" or path.startswith("/api/"):
            response = JSONResponse(
                {"detail": "Authentication required"}, status_code=401
            )
        else:
            response = RedirectResponse("/auth/google/login", status_code=303)
        await response(scope, receive, send)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_configured(name: str) -> bool:
    value = os.environ.get(name, "")
    return bool(value and value != "your_api_key")


class RunCreate(BaseModel):
    dataset_source: Literal["huggingface", "local"] = "huggingface"
    dataset_path: str = Field(min_length=1, max_length=300)
    dataset: str = Field(default="default", min_length=1, max_length=200)
    split: str = Field(default="test", min_length=1, max_length=200)
    model_name: str = Field(default="deepgram/nova-3", min_length=3, max_length=200)
    language: str = Field(default="en", min_length=2, max_length=32)
    max_samples: int | None = Field(default=None, ge=1)
    max_workers: int = Field(default=4, ge=1, le=300)
    use_url: bool = False
    streaming: bool = False
    prompt: str | None = Field(default=None, max_length=2000)
    audio_preprocessor: Literal[
        "none", "arctan", "ai_coustics_vfl_2_1", "krisp_bvc", "rnnoise"
    ] = "none"
    vad_position: Literal["none", "pre", "post"] = "none"
    arctan_chunk_ms: int = Field(default=10, ge=1, le=1000)
    audio_preprocessor_batch_size: int = Field(default=1, ge=1, le=1000)
    ai_coustics_project_dir: str | None = None
    ai_coustics_sample_rate: int = Field(default=16000, ge=1)
    ai_coustics_pad_seconds: float = Field(default=2.0, ge=0)
    ai_coustics_enhancement_level: float = Field(default=1.0, ge=0, le=1)
    krisp_bvc_project_dir: str | None = None
    krisp_bvc_sample_rate: int = Field(default=16000, ge=1)
    krisp_bvc_pad_seconds: float = Field(default=2.0, ge=0)

    @model_validator(mode="after")
    def validate_combination(self):
        prefix, separator, variant = self.model_name.partition("/")
        if not separator or prefix not in PROVIDERS:
            raise ValueError(
                "model_name must use a registered provider prefix: "
                + ", ".join(PROVIDERS)
            )
        if variant not in PROVIDERS[prefix]["models"]:
            allowed = ", ".join(
                f"{prefix}/{model}" for model in PROVIDERS[prefix]["models"]
            )
            raise ValueError(f"Unsupported model_name. Allowed values: {allowed}")
        mode = effective_mode(self.model_name, self.streaming)
        supported_languages = language_options(self.model_name, self.streaming)
        if not supported_languages:
            raise ValueError(
                f"{self.model_name} does not support {mode} evaluation in the console"
            )
        supported_codes = {option["code"] for option in supported_languages}
        if self.language not in supported_codes:
            raise ValueError(
                f"Unsupported language '{self.language}' for {self.model_name} "
                f"in {mode} mode. Allowed values: {', '.join(sorted(supported_codes))}"
            )
        if self.dataset_source == "local":
            relative = Path(self.dataset_path)
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or relative.name.startswith(".")
            ):
                raise ValueError("Local dataset path must name an immediate child directory")
            if self.dataset != "default" or self.split != "test":
                raise ValueError(
                    "Local datasets expose only config 'default' and split 'test'"
                )
        forced_streaming = self.model_name == "soniox/stt-rt-v5"
        if self.dataset_source == "local" and self.use_url:
            raise ValueError("Local datasets cannot be combined with URL mode")
        if self.use_url and (self.streaming or forced_streaming):
            raise ValueError("URL mode cannot be combined with streaming")
        if self.use_url and prefix == "sarvam":
            raise ValueError("Sarvam requires local audio; URL mode is not supported")
        if self.use_url and self.audio_preprocessor != "none":
            raise ValueError("URL mode cannot be combined with audio preprocessing")
        if self.use_url and self.vad_position != "none":
            raise ValueError("URL mode cannot be combined with VAD")
        return self


class DatasetInspect(BaseModel):
    dataset_path: str = Field(min_length=1, max_length=300)


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.db"
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    pid INTEGER,
                    output_dir TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    summary_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                UPDATE runs
                SET status = 'interrupted', finished_at = ?,
                    error = COALESCE(error, 'UI server restarted while run was active')
                WHERE status IN ('queued', 'running', 'cancelling')
                """,
                (now_iso(),),
            )

    def create(self, run_id: str, config: dict, output_dir: Path):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, status, created_at, output_dir, config_json)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (run_id, now_iso(), str(output_dir), json.dumps(config)),
            )

    def update(self, run_id: str, **values):
        if not values:
            return
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE runs SET {assignments} WHERE id = ?",
                (*values.values(), run_id),
            )

    def get(self, run_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._serialize(row) if row else None

    def list(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [self._serialize(row) for row in rows]

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        summary_json = result.pop("summary_json")
        result["summary"] = json.loads(summary_json) if summary_json else None
        output_dir = Path(result["output_dir"])
        progress_path = output_dir / "progress.json"
        result["progress"] = None
        if progress_path.is_file():
            try:
                result["progress"] = json.loads(
                    progress_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pass
        artifacts = []
        if output_dir.exists():
            artifacts = sorted(
                path.name
                for path in output_dir.iterdir()
                if path.is_file()
                and (
                    path.name in {"run.log", "summary.json"} or path.suffix == ".jsonl"
                )
            )
        result["artifacts"] = artifacts
        return result


class RunManager:
    def __init__(self, store: RunStore, runner_path: Path = RUNNER_PATH):
        self.store = store
        self.runner_path = runner_path
        self.processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(self, request: RunCreate) -> dict:
        self._validate_credentials(request)
        validate_dataset_selection(
            request.dataset_source,
            request.dataset_path,
            request.dataset,
            request.split,
        )
        run_id = uuid.uuid4().hex[:12]
        output_dir = self.store.root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        config = request.model_dump()
        self.store.create(run_id, config, output_dir)
        thread = threading.Thread(
            target=self._execute,
            args=(run_id, request, output_dir),
            daemon=True,
            name=f"eval-{run_id}",
        )
        thread.start()
        return self.store.get(run_id)

    def retry(self, run_id: str) -> dict:
        original = self.store.get(run_id)
        if original is None:
            raise KeyError(run_id)
        if original["status"] not in TERMINAL_STATES:
            raise RuntimeError("Only finished runs can be retried")
        request = RunCreate.model_validate(original["config"])
        return self.start(request)

    def _validate_credentials(self, request: RunCreate):
        prefix = request.model_name.split("/", 1)[0]
        missing = [
            name
            for name in PROVIDERS[prefix]["credentials"]
            if not env_configured(name)
        ]
        if request.model_name == "microsoft/MAI-Transcribe-1.5" and request.streaming:
            missing = [
                name for name in ("AZURE_SPEECH_ENDPOINT",) if not env_configured(name)
            ]
            if not (
                env_configured("AZURE_SPEECH_KEY") or env_configured("AZURE_API_KEY")
            ):
                missing.append("AZURE_SPEECH_KEY")
        missing.extend(
            name
            for name in PREPROCESSOR_CREDENTIALS.get(request.audio_preprocessor, [])
            if not env_configured(name)
        )
        if missing:
            raise ValueError(
                "Missing server credentials: " + ", ".join(sorted(set(missing)))
            )

    def build_command(self, request: RunCreate, output_dir: Path) -> list[str]:
        command = [
            sys.executable,
            "-u",
            str(self.runner_path),
            "--dataset_source",
            request.dataset_source,
            "--dataset_path",
            request.dataset_path,
            "--dataset",
            request.dataset,
            "--split",
            request.split,
            "--model_name",
            request.model_name,
            "--language",
            request.language,
            "--max_workers",
            str(request.max_workers),
            "--output_dir",
            str(output_dir),
            "--audio_preprocessor",
            request.audio_preprocessor,
            "--vad_position",
            request.vad_position,
            "--arctan_chunk_ms",
            str(request.arctan_chunk_ms),
            "--audio_preprocessor_batch_size",
            str(request.audio_preprocessor_batch_size),
        ]
        if request.max_samples is not None:
            command.extend(["--max_samples", str(request.max_samples)])
        if request.use_url:
            command.append("--use_url")
        if request.streaming:
            command.append("--streaming")
        if request.prompt:
            command.extend(["--prompt", request.prompt])
        if request.ai_coustics_project_dir:
            command.extend(
                ["--ai_coustics_project_dir", request.ai_coustics_project_dir]
            )
        if request.audio_preprocessor == "ai_coustics_vfl_2_1":
            command.extend(
                [
                    "--ai_coustics_sample_rate",
                    str(request.ai_coustics_sample_rate),
                    "--ai_coustics_pad_seconds",
                    str(request.ai_coustics_pad_seconds),
                    "--ai_coustics_enhancement_level",
                    str(request.ai_coustics_enhancement_level),
                ]
            )
        if request.krisp_bvc_project_dir:
            command.extend(["--krisp_bvc_project_dir", request.krisp_bvc_project_dir])
        if request.audio_preprocessor == "krisp_bvc":
            command.extend(
                [
                    "--krisp_bvc_sample_rate",
                    str(request.krisp_bvc_sample_rate),
                    "--krisp_bvc_pad_seconds",
                    str(request.krisp_bvc_pad_seconds),
                ]
            )
        return command

    def _execute(self, run_id: str, request: RunCreate, output_dir: Path):
        log_path = output_dir / "run.log"
        command = self.build_command(request, output_dir)
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with self._lock:
                    self.processes[run_id] = process
                self.store.update(
                    run_id,
                    status="running",
                    started_at=now_iso(),
                    pid=process.pid,
                )
                return_code = process.wait()

            current = self.store.get(run_id)
            summary_path = output_dir / "summary.json"
            summary = None
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if current and current["status"] == "cancelling":
                status = "cancelled"
                error = "Cancelled by operator"
            elif return_code == 0:
                status = "completed"
                error = None
            else:
                status = "failed"
                error = (summary or {}).get(
                    "error"
                ) or f"Evaluator exited with code {return_code}"
            self.store.update(
                run_id,
                status=status,
                finished_at=now_iso(),
                summary_json=json.dumps(summary) if summary else None,
                error=error,
            )
        except Exception as exc:
            self.store.update(
                run_id,
                status="failed",
                finished_at=now_iso(),
                error=str(exc),
            )
        finally:
            with self._lock:
                self.processes.pop(run_id, None)

    def cancel(self, run_id: str) -> dict:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run["status"] in TERMINAL_STATES:
            return run
        with self._lock:
            process = self.processes.get(run_id)
        self.store.update(run_id, status="cancelling")
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            def force_kill():
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            timer = threading.Timer(5, force_kill)
            timer.daemon = True
            timer.start()
        return self.store.get(run_id)


def options_payload() -> dict:
    providers = []
    credential_names = {"HF_TOKEN"}
    for prefix, provider in PROVIDERS.items():
        credential_names.update(provider["credentials"])
        providers.append(
            {
                "prefix": prefix,
                "label": provider["label"],
                "models": provider["models"],
                "language_options": {
                    model: MODEL_LANGUAGES[f"{prefix}/{model}"]
                    for model in provider["models"]
                },
                "configured": all(
                    env_configured(name) for name in provider["credentials"]
                ),
            }
        )
    for names in PREPROCESSOR_CREDENTIALS.values():
        credential_names.update(names)
    credential_names.update({"AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT"})
    return {
        "providers": providers,
        "preprocessors": PREPROCESSORS,
        "vad_positions": VAD_POSITIONS,
        "dataset_sources": source_options(),
        "credentials": {
            name: env_configured(name) for name in sorted(credential_names)
        },
        "defaults": {
            "dataset_source": "huggingface",
            "model_name": "deepgram/nova-3",
            "dataset": "default",
            "split": "test",
            "max_workers": 4,
        },
    }


def create_app(
    results_root: Path | None = None,
    runner_path: Path = RUNNER_PATH,
    web_dist: Path = DEFAULT_WEB_DIST,
    auth_settings: AuthSettings | None = None,
) -> FastAPI:
    app = FastAPI(title="Open ASR Evaluation Console", version="0.1.0")
    auth = auth_settings or AuthSettings.from_env()
    oauth = OAuth()
    google_oauth = None
    if auth.enabled and auth.configured:
        google_oauth = oauth.register(
            name="google",
            client_id=auth.client_id,
            client_secret=auth.client_secret,
            server_metadata_url=GOOGLE_DISCOVERY_URL,
            client_kwargs={"scope": "openid email profile"},
        )
    store = RunStore(
        results_root
        or Path(os.environ.get("OPEN_ASR_UI_RESULTS_DIR", DEFAULT_RESULTS_ROOT))
    )
    manager = RunManager(store, runner_path=runner_path)
    app.state.store = store
    app.state.manager = manager
    app.state.auth_settings = auth
    app.state.oauth = oauth
    app.state.google_oauth = google_oauth

    @app.get("/healthz", include_in_schema=False)
    async def liveness():
        return {"status": "ok"}

    @app.get("/auth/google/login", include_in_schema=False)
    async def google_login(request: Request):
        if not auth.enabled:
            return RedirectResponse("/", status_code=303)
        if not auth.configured or app.state.google_oauth is None:
            return JSONResponse(
                {
                    "detail": "Google authentication is not configured",
                    "missing": auth.missing,
                },
                status_code=503,
            )
        return await app.state.google_oauth.authorize_redirect(
            request,
            auth.callback_url,
            hd=auth.google_domain,
        )

    @app.get("/auth/google/callback", include_in_schema=False)
    async def google_callback(request: Request):
        if not auth.enabled:
            return RedirectResponse("/", status_code=303)
        if not auth.configured or app.state.google_oauth is None:
            return JSONResponse(
                {
                    "detail": "Google authentication is not configured",
                    "missing": auth.missing,
                },
                status_code=503,
            )
        try:
            token = await app.state.google_oauth.authorize_access_token(request)
        except Exception:
            request.session.clear()
            return JSONResponse({"detail": "Google sign-in failed"}, status_code=400)

        identity = token.get("userinfo") or {}
        if (
            identity.get("email_verified") is not True
            or str(identity.get("hd", "")).lower() != auth.google_domain
        ):
            request.session.clear()
            return JSONResponse(
                {"detail": f"Access is restricted to {auth.google_domain}"},
                status_code=403,
            )
        if not identity.get("sub") or not identity.get("email"):
            request.session.clear()
            return JSONResponse({"detail": "Google identity is incomplete"}, status_code=400)

        request.session.clear()
        request.session["user"] = {
            "id": str(identity["sub"]),
            "email": str(identity["email"]),
            "name": str(identity.get("name") or identity["email"]),
        }
        return RedirectResponse("/", status_code=303)

    @app.get("/api/auth/me")
    async def current_user(request: Request):
        return request.session["user"]

    @app.post("/api/auth/logout", status_code=204)
    async def logout(request: Request):
        request.session.clear()

    @app.get("/api/health")
    async def health():
        runs = await asyncio.to_thread(store.list)
        active = sum(
            run["status"] in {"queued", "running", "cancelling"} for run in runs
        )
        return {"status": "ok", "active_runs": active}

    @app.get("/api/options")
    async def options():
        return await asyncio.to_thread(options_payload)

    @app.get("/api/datasets")
    async def list_datasets(source_id: str):
        try:
            return await asyncio.to_thread(dataset_catalog, source_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Dataset source not found"
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.post("/api/datasets/inspect")
    async def inspect_dataset(request: DatasetInspect):
        def inspect():
            import datasets

            configs = datasets.get_dataset_config_names(request.dataset_path)
            details = []
            for config in configs:
                splits = datasets.get_dataset_split_names(request.dataset_path, config)
                builder = datasets.load_dataset_builder(request.dataset_path, config)
                details.append(
                    {
                        "name": config,
                        "splits": splits,
                        "features": list(builder.info.features or {}),
                        "schema": (
                            builder.info.features.to_dict()
                            if builder.info.features is not None
                            else {}
                        ),
                    }
                )
            return {"dataset_path": request.dataset_path, "configs": details}

        try:
            return await asyncio.to_thread(inspect)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs", status_code=201)
    async def create_run(request: RunCreate):
        try:
            return await asyncio.to_thread(manager.start, request)
        except DatasetValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "dataset_incompatible", "message": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    async def list_runs():
        return await asyncio.to_thread(store.list)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run = await asyncio.to_thread(store.get, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        try:
            return await asyncio.to_thread(manager.cancel, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.post("/api/runs/{run_id}/retry", status_code=201)
    async def retry_run(run_id: str):
        try:
            return await asyncio.to_thread(manager.retry, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DatasetValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "dataset_incompatible", "message": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):

        if store.get(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")

        async def events():
            offset = 0
            last_status = None
            last_progress = None
            while True:
                run = store.get(run_id)
                if run is None:
                    break
                log_path = Path(run["output_dir"]) / "run.log"
                if log_path.exists():
                    with log_path.open(
                        "r", encoding="utf-8", errors="replace"
                    ) as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                        offset = handle.tell()
                    if chunk:
                        yield f"event: log\ndata: {json.dumps({'chunk': chunk})}\n\n"
                if run["status"] != last_status:
                    last_status = run["status"]
                    yield f"event: status\ndata: {json.dumps(run)}\n\n"
                progress = run.get("progress")
                if progress != last_progress:
                    last_progress = progress
                    yield f"event: progress\ndata: {json.dumps(run)}\n\n"
                if run["status"] in TERMINAL_STATES:
                    break
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.75)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    async def artifact(run_id: str, name: str):
        run = await asyncio.to_thread(store.get, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if Path(name).name != name or not (
            name in {"run.log", "summary.json"} or name.endswith(".jsonl")
        ):
            raise HTTPException(status_code=400, detail="Unsupported artifact")
        path = Path(run["output_dir"]) / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path, filename=name)

    assets_dir = web_dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        index_path = web_dist / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        return JSONResponse(
            {
                "message": "Open ASR Evaluation Console API",
                "frontend": "not built",
            }
        )

    app.add_middleware(ConsoleAuthMiddleware, settings=auth)
    app.add_middleware(
        SessionMiddleware,
        secret_key=auth.session_secret or secrets.token_urlsafe(32),
        session_cookie="open_asr_session",
        max_age=SESSION_MAX_AGE_SECONDS,
        same_site="lax",
        https_only=auth.enabled,
    )

    return app


app = create_app()
