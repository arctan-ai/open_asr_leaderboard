import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock


try:
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
        self.manager = RunManager(self.store, runner_path=self.runner)

    def tearDown(self):
        self.tempdir.cleanup()

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

    def test_validation_rejects_incompatible_url_mode(self):
        with self.assertRaisesRegex(ValueError, "URL mode"):
            self.request(use_url=True, audio_preprocessor="arctan")
        with self.assertRaisesRegex(ValueError, "URL mode"):
            self.request(
                model_name="soniox/stt-rt-v5",
                use_url=True,
            )

    def test_validation_rejects_unregistered_model(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model_name"):
            self.request(model_name="deepgram/not-a-real-model")

    def test_command_is_an_argument_list_and_uses_unique_output(self):
        request = self.request(dataset_path="dataset; touch /tmp/not-run")
        output = self.root / "isolated"
        command = self.manager.build_command(request, output)

        self.assertIn("dataset; touch /tmp/not-run", command)
        self.assertEqual(command[command.index("--output_dir") + 1], str(output))
        self.assertNotIn("sh", command[:2])

    def test_completed_run_persists_summary_and_artifacts(self):
        with mock.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "configured"}):
            run = self.manager.start(self.request())
        completed = wait_for(self.store, run["id"], {"completed"})

        self.assertEqual(completed["summary"]["wer_percent"], 12.5)
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
        self.assertIn("loading dataset", response.text)
        self.assertIn('"status": "completed"', response.text)


if __name__ == "__main__":
    unittest.main()
