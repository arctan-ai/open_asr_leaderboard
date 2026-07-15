import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock


try:
    from api.dataset_catalog import DatasetValidationError
    from fastapi.testclient import TestClient
    from api.ui_server import RunCreate, RunManager, RunStore, create_app, options_payload
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
        self.manager = RunManager(self.store, runner_path=self.runner)

    def tearDown(self):
        self.tempdir.cleanup()
        self.dataset_validation_patcher.stop()

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
        )
        response = TestClient(app).get("/api/datasets?source_id=unknown")

        self.assertEqual(response.status_code, 404)

    def test_api_reports_dataset_incompatibility_before_creating_run(self):
        app = create_app(
            results_root=self.root / "incompatible-runs",
            runner_path=self.runner,
            web_dist=self.root / "missing-dist",
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
        )
        response = TestClient(app).get(f"/api/runs/{run['id']}/events")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: log", response.text)
        self.assertIn("event: progress", response.text)
        self.assertIn('"deepgram/nova-3": 1', response.text)
        self.assertIn("loading dataset", response.text)
        self.assertIn('"status": "completed"', response.text)


if __name__ == "__main__":
    unittest.main()
