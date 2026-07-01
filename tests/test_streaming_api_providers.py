import asyncio
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "api"


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


def load_providers():
    assemblyai_stub = types.ModuleType("assemblyai")
    assemblyai_stub.settings = types.SimpleNamespace(api_key=None)
    assemblyai_stub.Transcriber = lambda: None
    assemblyai_stub.TranscriptionConfig = lambda *args, **kwargs: None
    assemblyai_stub.TranscriptStatus = types.SimpleNamespace(error="error")
    sys.modules["assemblyai"] = assemblyai_stub

    if str(API_DIR) not in sys.path:
        sys.path.insert(0, str(API_DIR))

    for name in list(sys.modules):
        if name == "providers" or name.startswith("providers."):
            del sys.modules[name]

    return importlib.import_module("providers")


def load_run_eval(module_name):
    providers = load_providers()

    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: []
    datasets_stub.Audio = lambda *args, **kwargs: ("Audio", args, kwargs)

    evaluate_stub = types.ModuleType("evaluate")
    evaluate_stub.load = lambda *args, **kwargs: None

    soundfile_stub = types.ModuleType("soundfile")
    soundfile_stub.write = lambda *args, **kwargs: None

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None

    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda value, *args, **kwargs: value

    normalizer_stub = types.ModuleType("normalizer")
    normalizer_stub.__path__ = [str(REPO_ROOT / "normalizer")]
    data_utils_stub = types.ModuleType("normalizer.data_utils")
    data_utils_stub.prepare_data = lambda ds, args=None: ds
    data_utils_stub.add_audio_preprocessor_args = lambda parser: None
    data_utils_stub.is_target_text_in_range = lambda text: True
    data_utils_stub.ml_normalizer = lambda text, lang=None: text
    data_utils_stub.post_slack_run_started = lambda *args, **kwargs: None
    data_utils_stub.post_slack_run_failed = lambda *args, **kwargs: None
    data_utils_stub.post_slack_single_run_summary = lambda *args, **kwargs: None
    eval_utils_stub = types.ModuleType("normalizer.eval_utils")
    eval_utils_stub.normalize_compound_pairs = lambda refs, preds: (refs, preds)

    module_path = API_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"{module_name}_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)

    with mock.patch.dict(
        sys.modules,
        {
            "datasets": datasets_stub,
            "evaluate": evaluate_stub,
            "soundfile": soundfile_stub,
            "dotenv": dotenv_stub,
            "tqdm": tqdm_stub,
            "normalizer": normalizer_stub,
            "normalizer.data_utils": data_utils_stub,
            "normalizer.eval_utils": eval_utils_stub,
            "providers": providers,
        },
    ):
        spec.loader.exec_module(module)
    return module


class StreamingProviderTest(unittest.TestCase):
    def test_streaming_routes_to_provider_streaming_method(self):
        run_eval = load_run_eval("run_eval")

        class FakeProvider:
            def __init__(self):
                self.streaming_called = False

            def transcribe(self, *args, **kwargs):
                raise AssertionError("static transcribe should not be called")

            def transcribe_streaming(self, *args, **kwargs):
                self.streaming_called = True
                return "streamed"

        provider = FakeProvider()
        with mock.patch.object(
            run_eval, "get_provider", return_value=(provider, "nova-3")
        ):
            transcript = run_eval.transcribe_with_retry(
                "deepgram/nova-3",
                "/tmp/audio.wav",
                {"audio": {"array": [], "sampling_rate": 16000}},
                streaming=True,
            )

        self.assertEqual(transcript, "streamed")
        self.assertTrue(provider.streaming_called)

    def test_unsupported_provider_streaming_fails_clearly(self):
        providers = load_providers()

        class DummyProvider(providers.APIProvider):
            def transcribe(self, *args, **kwargs):
                return ""

        with self.assertRaisesRegex(providers.PermanentError, "not supported"):
            DummyProvider().transcribe_streaming(
                "model",
                "/tmp/audio.wav",
                {},
                use_url=False,
            )

    def test_streaming_use_url_is_rejected_by_runners(self):
        run_eval = load_run_eval("run_eval")
        run_eval_ml = load_run_eval("run_eval_ml")

        with self.assertRaisesRegex(ValueError, "--streaming requires local audio"):
            run_eval.transcribe_dataset(
                "dataset",
                "config",
                "test",
                "deepgram/nova-3",
                use_url=True,
                streaming=True,
                args=types.SimpleNamespace(audio_preprocessor="none"),
            )

        with self.assertRaisesRegex(ValueError, "--streaming requires local audio"):
            run_eval_ml.transcribe_dataset(
                "dataset",
                "config",
                "test",
                "deepgram/nova-3",
                "en",
                use_url=True,
                streaming=True,
            )

    def test_deepgram_collects_final_messages(self):
        load_providers()
        from providers import deepgram_provider

        messages = [
            json.dumps(
                {
                    "type": "Results",
                    "is_final": False,
                    "channel": {"alternatives": [{"transcript": "partial"}]},
                }
            ),
            json.dumps(
                {
                    "type": "Results",
                    "is_final": True,
                    "channel": {"alternatives": [{"transcript": "hello"}]},
                }
            ),
            json.dumps(
                {
                    "type": "Results",
                    "is_final": True,
                    "channel": {"alternatives": [{"transcript": "world"}]},
                }
            ),
            json.dumps({"type": "Metadata"}),
        ]
        fake_ws = FakeWebSocket(messages)

        with mock.patch.object(deepgram_provider, "pcm16_chunks", return_value=[b"a"]):
            with mock.patch.object(
                deepgram_provider, "connect_websocket", return_value=fake_ws
            ):
                transcript = asyncio.run(
                    deepgram_provider._transcribe_streaming(
                        "/tmp/audio.wav",
                        "key",
                        "nova-3",
                        "en",
                    )
                )

        self.assertEqual(transcript, "hello world")

    def test_soniox_collects_final_tokens_and_strips_end(self):
        load_providers()
        from providers import soniox_provider

        messages = [
            json.dumps(
                {
                    "tokens": [
                        {"text": "hel", "is_final": False},
                        {"text": "hello", "is_final": True},
                        {"text": " ", "is_final": True},
                        {"text": "world", "is_final": True},
                        {"text": "<end>", "is_final": True},
                    ],
                    "finished": True,
                }
            )
        ]
        fake_ws = FakeWebSocket(messages)

        with mock.patch.object(soniox_provider, "pcm16_chunks", return_value=[b"a"]):
            with mock.patch.object(
                soniox_provider, "connect_websocket", return_value=fake_ws
            ):
                transcript = asyncio.run(
                    soniox_provider._transcribe_streaming(
                        "/tmp/audio.wav",
                        "key",
                        "stt-rt-v5",
                        "en",
                    )
                )

        self.assertEqual(transcript, "hello world")

    def test_assembly_collects_final_turns(self):
        load_providers()
        from providers import assemblyai_provider

        messages = [
            json.dumps({"type": "Begin"}),
            json.dumps(
                {
                    "type": "Turn",
                    "end_of_turn": False,
                    "transcript": "partial",
                }
            ),
            json.dumps(
                {
                    "type": "Turn",
                    "end_of_turn": True,
                    "transcript": "hello",
                }
            ),
            json.dumps(
                {
                    "type": "Turn",
                    "end_of_turn": True,
                    "transcript": "world",
                }
            ),
            json.dumps({"type": "Termination"}),
        ]
        fake_ws = FakeWebSocket(messages)

        with mock.patch.object(
            assemblyai_provider, "pcm16_chunks", return_value=[b"a"]
        ):
            with mock.patch.object(
                assemblyai_provider, "connect_websocket", return_value=fake_ws
            ):
                transcript = asyncio.run(
                    assemblyai_provider._transcribe_streaming(
                        "/tmp/audio.wav",
                        "key",
                        "universal-3-5-pro",
                    )
                )

        self.assertEqual(transcript, "hello world")
        self.assertIn(json.dumps({"type": "Terminate"}), fake_ws.sent)
        self.assertTrue(fake_ws.closed)


    def test_cartesia_collects_turn_end_transcripts(self):
        load_providers()
        from providers import cartesia_provider

        messages = [
            json.dumps({"type": "connected", "request_id": "abc"}),
            json.dumps({"type": "turn.start"}),
            json.dumps({"type": "turn.update", "transcript": "hello"}),
            json.dumps({"type": "turn.end", "transcript": "hello world"}),
            json.dumps({"type": "turn.start"}),
            json.dumps({"type": "turn.update", "transcript": "foo"}),
            json.dumps({"type": "turn.end", "transcript": "foo bar"}),
        ]
        fake_ws = FakeWebSocket(messages)

        with mock.patch.object(cartesia_provider, "pcm16_chunks", return_value=[b"a"]):
            with mock.patch.object(
                cartesia_provider, "connect_websocket", return_value=fake_ws
            ):
                transcript = asyncio.run(
                    cartesia_provider._transcribe_streaming(
                        "/tmp/audio.wav",
                        "key",
                        "ink-2",
                    )
                )

        self.assertEqual(transcript, "hello world foo bar")
        self.assertIn(json.dumps({"type": "close"}), fake_ws.sent)


if __name__ == "__main__":
    unittest.main()
