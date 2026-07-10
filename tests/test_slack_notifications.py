import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


kaldialign_stub = types.ModuleType("kaldialign")
kaldialign_stub.batch_error_rate = lambda *args, **kwargs: {
    "ins": 0,
    "del": 0,
    "sub": 0,
    "err_rate": 0.0,
}

with mock.patch.dict(sys.modules, {"kaldialign": kaldialign_stub}):
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "eval_utils_under_test", repo_root / "normalizer" / "eval_utils.py"
    )
    eval_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_utils)


class FakeSlackResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


def sample_metrics():
    composite_wer = {"acme/model": 24.68}
    composite_audio_length = {"acme/model": 120.0}
    composite_inference_time = {"acme/model": 4.0}
    count_entries = {"acme/model": 2}
    results = {
        "acme/model | hf-audio-open-asr-leaderboard_ami_test": {
            "wer": 10.0,
            "audio_length": 60.0,
            "inference_time": 2.0,
            "rtfx": 30.0,
        },
        "acme/model | hf-audio-open-asr-leaderboard_librispeech_test.clean": {
            "wer": 14.68,
            "audio_length": 60.0,
            "inference_time": 2.0,
            "rtfx": 30.0,
        },
    }
    return (
        composite_wer,
        composite_audio_length,
        composite_inference_time,
        count_entries,
        results,
    )


class SlackNotificationTest(unittest.TestCase):
    def test_run_metadata_includes_vad_position(self):
        lines = eval_utils._build_run_metadata_lines(
            "acme/model",
            "hf-audio/open-asr-leaderboard",
            "ami",
            "test",
            10,
            4,
            "arctan",
            vad_position="post",
        )

        self.assertIn("*VAD position:* `post`", lines)

    def test_missing_slack_env_skips_http_call(self):
        metrics = sample_metrics()
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(eval_utils.urllib.request, "urlopen") as urlopen:
                eval_utils._post_slack_eval_summary(
                    "/tmp/results",
                    ["one.jsonl", "two.jsonl"],
                    "acme/model",
                    *metrics,
                    multilingual=False,
                    language="en",
                )

        urlopen.assert_not_called()

    def test_valid_env_posts_chat_message_payload(self):
        metrics = sample_metrics()

        with mock.patch.dict(
            os.environ,
            {
                "SLACK_BOT_TOKEN": "xoxb-test-token",
                "SLACK_CHANNEL_ID": "C123",
                "RESULTS_BUCKET": "hf-audio/results",
            },
            clear=True,
        ):
            with mock.patch.object(
                eval_utils.urllib.request,
                "urlopen",
                return_value=FakeSlackResponse({"ok": True}),
            ) as urlopen:
                eval_utils._post_slack_eval_summary(
                    "/tmp/results",
                    ["one.jsonl", "two.jsonl"],
                    "acme/model",
                    *metrics,
                    multilingual=False,
                    language="en",
                )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        rendered_blocks = "\n".join(
            block["text"]["text"] for block in payload["blocks"]
        )

        self.assertEqual(payload["channel"], "C123")
        self.assertIn("ASR evaluation completed", payload["text"])
        self.assertIn("acme/model", payload["text"])
        self.assertIn("WER 12.34%", payload["text"])
        self.assertIn("RTFx 30.00", payload["text"])
        self.assertIn("Result files:* 2", rendered_blocks)
        self.assertIn("Datasets:* 2", rendered_blocks)
        self.assertIn("RESULTS_BUCKET", rendered_blocks)
        self.assertIn("ami_test", rendered_blocks)
        self.assertIn("librispeech_test.clean", rendered_blocks)
        self.assertEqual(request.get_header("Authorization"), "Bearer xoxb-test-token")

    def test_slack_api_error_warns_without_raising(self):
        metrics = sample_metrics()

        with mock.patch.dict(
            os.environ,
            {"SLACK_BOT_TOKEN": "xoxb-test-token", "SLACK_CHANNEL_ID": "C123"},
            clear=True,
        ):
            with mock.patch.object(
                eval_utils.urllib.request,
                "urlopen",
                return_value=FakeSlackResponse(
                    {"ok": False, "error": "channel_not_found"}
                ),
            ):
                with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    eval_utils._post_slack_eval_summary(
                        "/tmp/results",
                        ["one.jsonl", "two.jsonl"],
                        "acme/model",
                        *metrics,
                        multilingual=False,
                        language="en",
                    )

        self.assertIn(
            "WARNING: Slack notification failed: channel_not_found", stdout.getvalue()
        )

    def test_network_error_warns_without_raising(self):
        metrics = sample_metrics()

        with mock.patch.dict(
            os.environ,
            {"SLACK_BOT_TOKEN": "xoxb-test-token", "SLACK_CHANNEL_ID": "C123"},
            clear=True,
        ):
            with mock.patch.object(
                eval_utils.urllib.request,
                "urlopen",
                side_effect=RuntimeError("network unavailable"),
            ):
                with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    eval_utils._post_slack_eval_summary(
                        "/tmp/results",
                        ["one.jsonl", "two.jsonl"],
                        "acme/model",
                        *metrics,
                        multilingual=False,
                        language="en",
                    )

        self.assertIn(
            "WARNING: Slack notification failed: network unavailable", stdout.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
