import asyncio
from base64 import b64decode, b64encode
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import httpx
from itsdangerous import TimestampSigner


try:
    from api.dataset_catalog import DatasetValidationError
    from fastapi.responses import JSONResponse
    from api.ui_server import (
        AuthSettings,
        RunCreate,
        RunManager,
        RunStore,
        SESSION_MAX_AGE_SECONDS,
        create_app,
        options_payload,
    )
except ImportError as exc:  # pragma: no cover - reported clearly in minimal envs
    raise unittest.SkipTest(f"UI server dependencies unavailable: {exc}")


FAKE_RUNNER = """
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--dataset_path')
parser.add_argument('--model_name')
parser.add_argument('--sleep', action='store_true')
args, _ = parser.parse_known_args()
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
print('loading dataset', flush=True)
progress = {
    'completed_samples': 1,
    'total_samples': 1,
    'actual_models': {'deepgram/nova-3': 1},
    'detected_languages': {'en': 1},
}
(output / 'progress.json').write_text(json.dumps(progress))
if 'slow' in (args.dataset_path or ''):
    time.sleep(30)
(output / 'manifest.jsonl').write_text('{"text": "hello"}\\n')
summary = {
    'status': 'completed',
    'wer_percent': 12.5,
    'rtfx': 20.0,
    'num_samples': 1,
}
(output / 'summary.json').write_text(json.dumps(summary))
print('WER: 12.5 %', flush=True)
"""


class TestClient:
    """Small synchronous wrapper around HTTPX's non-blocking ASGI transport."""

    def __init__(self, app, base_url="http://testserver"):
        self.app = app
        self.base_url = base_url
        self.cookies = httpx.Cookies()
        self.loop = asyncio.new_event_loop()

    def __del__(self):
        if not self.loop.is_closed():
            self.loop.close()

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=self.base_url,
                cookies=self.cookies,
            ) as client:
                response = await client.request(method, path, **kwargs)
                self.cookies = httpx.Cookies(client.cookies)
                return response

        return self.loop.run_until_complete(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


def wait_for(store, run_id, states, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = store.get(run_id)
        if run["status"] in states:
            return run
        time.sleep(0.03)
    raise AssertionError(f"run {run_id} did not reach {states}: {store.get(run_id)}")


class UiServerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runner = self.root / "fake_runner.py"
        self.runner.write_text(FAKE_RUNNER, encoding="utf-8")
        self.store = RunStore(self.root / "runs")
        self.dataset_validation_patcher = mock.patch(
            "api.ui_server.validate_dataset_selection", return_value={"features": ["audio", "text"], "splits": ["test"]}
        )
        self.dataset_validation = self.dataset_validation_patcher.start()
        async def run_inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        self.to_thread_patcher = mock.patch(
            "api.ui_server.asyncio.to_thread", side_effect=run_inline
        )
        self.to_thread_patcher.start()
        self.manager = RunManager(self.store, runner_path=self.runner)

    def tearDown(self):
        self.tempdir.cleanup()
        self.dataset_validation_patcher.stop()
        self.to_thread_patcher.stop()

    def request(self, **overrides):
        values = {
            "dataset_path": "bettercallaaryan/nc_agent_clips_openasr",
            "dataset": "default",
            "split": "test",
            "model_name": "deepgram/nova-3",
        }
        values.update(overrides)
        return RunCreate(**values)

    def test_options_only_exposes_credential_status(self):
        with mock.patch.dict(
            os.environ,
            {"DEEPGRAM_API_KEY": "secret-value"},
            clear=True,
        ):
            payload = options_payload()

        rendered = json.dumps(payload)
        self.assertNotIn("secret-value", rendered)
        self.assertTrue(payload["credentials"]["DEEPGRAM_API_KEY"])
        self.assertEqual(payload["defaults"]["dataset_source"], "huggingface")
        self.assertEqual(
            [source["id"] for source in payload["dataset_sources"]],
            ["huggingface", "local"],
        )

        self.assertEqual(payload["dataset_sources"][0]["description"], "2 configured repositories")
    def test_validation_rejects_incompatible_url_mode(self):
        with self.assertRaisesRegex(ValueError, "URL mode"):
            self.request(use_url=True, audio_preprocessor="arctan")
        with self.assertRaisesRegex(ValueError, "Local datasets"):
            self.request(dataset_source="local", dataset_path="Valid", use_url=True)
        with self.assertRaisesRegex(ValueError, "immediate child"):
            self.request(dataset_source="local", dataset_path="../outside")
        with self.assertRaisesRegex(ValueError, "URL mode"):
            self.request(
                model_name="soniox/stt-rt-v5",
                use_url=True,
            )
        with self.assertRaisesRegex(ValueError, "Sarvam requires local audio"):
            self.request(
                model_name="sarvam/saaras:v3",
                language="unknown",
                use_url=True,
                audio_preprocessor="none",
            )

    def test_validation_rejects_unregistered_model(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model_name"):
            self.request(model_name="deepgram/not-a-real-model")

    def test_command_is_an_argument_list_and_uses_unique_output(self):
        request = self.request(
            dataset_path="dataset; touch /tmp/not-run", language="multi"
        )
        output = self.root / "isolated"
        command = self.manager.build_command(request, output)

        self.assertIn("dataset; touch /tmp/not-run", command)
        self.assertEqual(command[command.index("--output_dir") + 1], str(output))
        self.assertEqual(command[command.index("--language") + 1], "multi")
        self.assertNotIn("sh", command[:2])

        self.assertEqual(
            command[command.index("--dataset_source") + 1], "huggingface"
        )

    def test_run_request_validates_only_the_selected_dataset(self):
        request = self.request(dataset_path="owner/selected", dataset="config", split="validation")
        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            run = self.manager.start(request)
        wait_for(self.store, run["id"], {"completed"})

        self.dataset_validation.assert_called_with(
            "huggingface",
            "owner/selected",
            "config",
            "validation",
        )

    def test_sarvam_is_exposed_with_credential_status(self):
        with mock.patch.dict(os.environ, {"SARVAM_API_KEY": "configured"}, clear=True):
            payload = options_payload()

        sarvam = next(
            provider for provider in payload["providers"] if provider["prefix"] == "sarvam"
        )
        self.assertEqual(sarvam["models"], ["saaras:v3"])
        self.assertIn(
            {"code": "unknown", "label": "Automatic detection"},
            sarvam["language_options"]["saaras:v3"]["streaming"],
        )
        self.assertTrue(sarvam["configured"])
        self.assertTrue(payload["credentials"]["SARVAM_API_KEY"])
        deepgram = next(
            provider for provider in payload["providers"] if provider["prefix"] == "deepgram"
        )
        deepgram_codes = {
            option["code"]
            for option in deepgram["language_options"]["nova-3"]["streaming"]
        }
        self.assertIn("multi", deepgram_codes)
        self.assertIn("en-IN", deepgram_codes)

    def test_assembly_uses_stable_alias_and_documented_languages(self):
        payload = options_payload()
        assembly = next(
            provider
            for provider in payload["providers"]
            if provider["prefix"] == "assembly"
        )
        self.assertEqual(assembly["models"], ["universal-stt"])
        codes = {
            option["code"]
            for option in assembly["language_options"]["universal-stt"]["batch"]
        }
        self.assertEqual(len(codes - {"unknown"}), 18)
        self.assertIn("hi", codes)
        self.assertIn("vi", codes)

    def test_validation_rejects_unsupported_language_and_model_mode(self):
        with self.assertRaisesRegex(ValueError, "Unsupported language 'xx'"):
            self.request(language="xx")
        with self.assertRaisesRegex(ValueError, "does not support streaming"):
            self.request(model_name="cartesia/ink-whisper", streaming=True)
        with self.assertRaisesRegex(ValueError, "does not support batch"):
            self.request(model_name="cartesia/ink-2")

    def test_validation_uses_forced_streaming_language_options(self):
        request = self.request(
            model_name="soniox/stt-rt-v5",
            language="unknown",
        )
        self.assertEqual(request.language, "unknown")

        with self.assertRaisesRegex(ValueError, "Unsupported language 'xx'"):
            self.request(model_name="soniox/stt-rt-v5", language="xx")

    def test_completed_run_persists_summary_and_artifacts(self):
        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            run = self.manager.start(self.request())
        completed = wait_for(self.store, run["id"], {"completed"})

        self.assertEqual(completed["summary"]["wer_percent"], 12.5)
        self.assertEqual(completed["progress"]["actual_models"], {"deepgram/nova-3": 1})
        self.assertIn("manifest.jsonl", completed["artifacts"])
        self.assertIn("run.log", completed["artifacts"])

    def test_parallel_runs_use_different_directories(self):
        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            first = self.manager.start(self.request(vad_position="pre"))
            second = self.manager.start(self.request(vad_position="post"))
        first_done = wait_for(self.store, first["id"], {"completed"})
        second_done = wait_for(self.store, second["id"], {"completed"})

        self.assertNotEqual(first_done["output_dir"], second_done["output_dir"])
        self.assertEqual(first_done["config"]["vad_position"], "pre")
        self.assertEqual(second_done["config"]["vad_position"], "post")

    def test_retry_endpoint_creates_a_new_run_with_the_same_configuration(self):
        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            original = self.manager.start(self.request(vad_position="pre"))
        original = wait_for(self.store, original["id"], {"completed"})
        app = create_app(
            results_root=self.root / "runs",
            runner_path=self.runner,
            web_dist=self.root / "missing-dist",
            auth_settings=AuthSettings.disabled_for_tests(),
        )

        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            response = TestClient(app).post(f"/api/runs/{original['id']}/retry")

        self.assertEqual(response.status_code, 201)
        retried = response.json()
        self.assertNotEqual(retried["id"], original["id"])
        self.assertEqual(retried["config"], original["config"])
        completed = wait_for(app.state.store, retried["id"], {"completed"})
        self.assertNotEqual(completed["output_dir"], original["output_dir"])
        self.assertEqual(len(app.state.store.list()), 2)

    def test_retry_rejects_an_active_run(self):
        self.store.create("active", self.request().model_dump(), self.root / "active")
        with self.assertRaisesRegex(RuntimeError, "Only finished runs"):
            self.manager.retry("active")

    def test_cancel_stops_active_process(self):
        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            run = self.manager.start(self.request(dataset_path="slow/dataset"))
        wait_for(self.store, run["id"], {"running"})
        self.manager.cancel(run["id"])
        cancelled = wait_for(self.store, run["id"], {"cancelled"})

        self.assertEqual(cancelled["error"], "Cancelled by operator")

    def test_restart_marks_active_run_interrupted(self):
        output = self.root / "runs" / "stale"
        output.mkdir()
        self.store.create("stale", self.request().model_dump(), output)
        self.store.update("stale", status="running")

        restarted = RunStore(self.root / "runs")
        self.assertEqual(restarted.get("stale")["status"], "interrupted")

    def test_dataset_catalog_endpoint_returns_source_payload(self):
        app = create_app(
            results_root=self.root / "catalog-runs",
            runner_path=self.runner,
            web_dist=self.root / "missing-dist",
            auth_settings=AuthSettings.disabled_for_tests(),
        )
        expected = {"source_id": "local", "datasets": []}
        with mock.patch("api.ui_server.dataset_catalog", return_value=expected):
            response = TestClient(app).get("/api/datasets?source_id=local")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)

    def test_dataset_catalog_endpoint_rejects_unknown_source(self):
        app = create_app(
            results_root=self.root / "unknown-source-runs",
            runner_path=self.runner,
            web_dist=self.root / "missing-dist",
            auth_settings=AuthSettings.disabled_for_tests(),
        )
        response = TestClient(app).get("/api/datasets?source_id=unknown")

        self.assertEqual(response.status_code, 404)

    def test_api_reports_dataset_incompatibility_before_creating_run(self):
        app = create_app(
            results_root=self.root / "incompatible-runs",
            runner_path=self.runner,
            web_dist=self.root / "missing-dist",
            auth_settings=AuthSettings.disabled_for_tests(),
        )
        message = (
            "This dataset does not follow the required format expected by the "
            "evaluator and cannot be selected: missing required 'audio' column."
        )
        with mock.patch(
            "api.ui_server.validate_dataset_selection",
            side_effect=DatasetValidationError(message),
        ), mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            response = TestClient(app).post(
                "/api/runs", json=self.request().model_dump()
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            {"code": "dataset_incompatible", "message": message},
        )
        self.assertEqual(app.state.store.list(), [])

    def test_api_rejects_artifact_path_traversal(self):
        app = create_app(
            results_root=self.root / "api-runs",
            runner_path=self.runner,
            web_dist=self.root / "missing-dist",
            auth_settings=AuthSettings.disabled_for_tests(),
        )
        client = TestClient(app)

        response = client.get("/api/runs/unknown/artifacts/..%2F.env")
        self.assertIn(response.status_code, {400, 404})

    def test_sse_replays_logs_and_terminal_status(self):
        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            run = self.manager.start(self.request())
        wait_for(self.store, run["id"], {"completed"})
        app = create_app(
            results_root=self.root / "runs",
            runner_path=self.runner,
            web_dist=self.root / "missing-dist",
            auth_settings=AuthSettings.disabled_for_tests(),
        )
        response = TestClient(app).get(f"/api/runs/{run['id']}/events")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: log", response.text)
        self.assertIn("event: progress", response.text)
        self.assertIn('"deepgram/nova-3": 1', response.text)
        self.assertIn("loading dataset", response.text)
        self.assertIn('"status": "completed"', response.text)


class GoogleAuthenticationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.settings = AuthSettings(
            client_id="client-id",
            client_secret="client-secret",
            session_secret="session-secret-with-at-least-32-bytes",
            public_url="https://console.arctan.ai",
            google_domain="arctan.ai",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def app(self, settings=None):
        return create_app(
            results_root=self.root / self._testMethodName,
            runner_path=self.root / "unused-runner.py",
            web_dist=self.root / "missing-dist",
            auth_settings=settings or self.settings,
        )

    def client(self, app):
        return TestClient(app, base_url="https://console.arctan.ai")

    def identity(self, **overrides):
        identity = {
            "sub": "google-user-123",
            "email": "operator@arctan.ai",
            "name": "Arctan Operator",
            "email_verified": True,
            "hd": "arctan.ai",
        }
        identity.update(overrides)
        return identity

    def sign_in(self, app, client, identity=None):
        app.state.google_oauth.authorize_access_token = mock.AsyncMock(
            return_value={
                "access_token": "google-access-token",
                "refresh_token": "google-refresh-token",
                "userinfo": identity or self.identity(),
            }
        )
        return client.get("/auth/google/callback", follow_redirects=False)

    def test_login_uses_exact_callback_and_hosted_domain_hint(self):
        app = self.app()
        app.state.google_oauth.authorize_redirect = mock.AsyncMock(
            return_value=JSONResponse({"redirect": "google"})
        )

        response = self.client(app).get("/auth/google/login")

        self.assertEqual(response.status_code, 200)
        call = app.state.google_oauth.authorize_redirect.await_args
        self.assertEqual(call.args[1], "https://console.arctan.ai/auth/google/callback")
        self.assertEqual(call.kwargs["hd"], "arctan.ai")

    def test_verified_arctan_identity_creates_minimal_session(self):
        app = self.app()
        client = self.client(app)

        response = self.sign_in(app, client)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")
        self.assertEqual(
            client.get("/api/auth/me").json(),
            {
                "id": "google-user-123",
                "email": "operator@arctan.ai",
                "name": "Arctan Operator",
            },
        )
        session_cookie = client.cookies.get("open_asr_session")
        signed_payload = TimestampSigner(self.settings.session_secret).unsign(
            session_cookie.encode("utf-8")
        )
        stored_session = json.loads(b64decode(signed_payload))
        self.assertEqual(set(stored_session), {"user"})
        self.assertEqual(
            set(stored_session["user"]), {"id", "email", "name"}
        )
        cookie_header = response.headers["set-cookie"].lower()
        for attribute in ["httponly", "secure", "samesite=lax", "max-age=28800"]:
            self.assertIn(attribute, cookie_header)

    def test_wrong_missing_or_unverified_workspace_identity_is_forbidden(self):
        cases = [
            self.identity(hd="other.example"),
            self.identity(hd=None),
            self.identity(email_verified=False),
        ]
        for identity in cases:
            with self.subTest(identity=identity):
                app = self.app()
                client = self.client(app)
                response = self.sign_in(app, client, identity)
                self.assertEqual(response.status_code, 403)
                self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_callback_failure_does_not_create_a_session(self):
        app = self.app()
        client = self.client(app)
        app.state.google_oauth.authorize_access_token = mock.AsyncMock(
            side_effect=RuntimeError("invalid OAuth state")
        )

        response = client.get("/auth/google/callback")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Google sign-in failed"})
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_logout_clears_the_session(self):
        app = self.app()
        client = self.client(app)
        self.sign_in(app, client)

        response = client.post("/api/auth/logout")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_tampered_and_expired_sessions_are_rejected(self):
        app = self.app()
        client = self.client(app)
        client.cookies.set("open_asr_session", "tampered", domain="console.arctan.ai")
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

        session_data = b64encode(
            json.dumps({"user": self.identity()}).encode("utf-8")
        )
        past = time.time() - SESSION_MAX_AGE_SECONDS - 1
        with mock.patch("itsdangerous.timed.time.time", return_value=past):
            expired_cookie = TimestampSigner(self.settings.session_secret).sign(
                session_data
            )
        client.cookies.set(
            "open_asr_session",
            expired_cookie.decode("utf-8"),
            domain="console.arctan.ai",
        )
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_incomplete_configuration_fails_closed_but_liveness_is_public(self):
        app = self.app(AuthSettings())
        client = self.client(app)

        self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("GOOGLE_OAUTH_CLIENT_ID", response.json()["missing"])
        self.assertEqual(client.get("/auth/google/login").status_code, 503)

    def test_configuration_requires_https_origin_and_strong_session_secret(self):
        settings = AuthSettings(
            client_id="client-id",
            client_secret="client-secret",
            session_secret="too-short",
            public_url="http://console.arctan.ai/path",
        )

        self.assertFalse(settings.configured)
        self.assertIn(
            "OPEN_ASR_SESSION_SECRET (must be at least 32 characters)",
            settings.missing,
        )
        self.assertIn(
            "OPEN_ASR_PUBLIC_URL (must be an HTTPS origin)", settings.missing
        )

    def test_anonymous_console_routes_are_protected(self):
        client = self.client(self.app())
        requests = [
            ("GET", "/api/health"),
            ("GET", "/api/options"),
            ("GET", "/api/datasets?source_id=local"),
            ("POST", "/api/datasets/inspect"),
            ("GET", "/api/runs"),
            ("POST", "/api/runs"),
            ("GET", "/api/runs/id"),
            ("POST", "/api/runs/id/cancel"),
            ("POST", "/api/runs/id/retry"),
            ("GET", "/api/runs/id/events"),
            ("GET", "/api/runs/id/artifacts/run.log"),
        ]
        for method, path in requests:
            with self.subTest(method=method, path=path):
                self.assertEqual(client.request(method, path).status_code, 401)

        browser_response = client.get("/", follow_redirects=False)
        self.assertEqual(browser_response.status_code, 303)
        self.assertEqual(browser_response.headers["location"], "/auth/google/login")


if __name__ == "__main__":
    unittest.main()
