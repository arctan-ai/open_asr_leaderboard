import asyncio
import base64
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
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


class FakeSignal:
    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)

    def emit(self, event):
        for handler in self.handlers:
            handler(event)


class FakeFuture:
    def get(self):
        return None


def fake_azure_speech_sdk(
    recognized_texts=(),
    cancellation=None,
    stop_session=True,
    cancellation_reason=None,
    cancellation_error_code=None,
    details_reason=None,
    details_error_code=None,
    details_error_details=None,
):
    state = types.SimpleNamespace(formats=[], streams=[], configs=[], recognizers=[])

    class AudioStreamFormat:
        def __init__(self, **kwargs):
            state.formats.append(kwargs)

    class PushAudioInputStream:
        def __init__(self, stream_format):
            self.writes = []
            self.closed = False
            state.streams.append(self)

        def write(self, chunk):
            self.writes.append(chunk)

        def close(self):
            self.closed = True

    class AudioConfig:
        def __init__(self, stream):
            self.stream = stream

    class SpeechConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.speech_recognition_language = None
            state.configs.append(self)

    class SpeechRecognizer:
        def __init__(self, **kwargs):
            self.recognized = FakeSignal()
            self.recognizing = FakeSignal()
            self.canceled = FakeSignal()
            self.session_stopped = FakeSignal()
            self.stopped = False
            state.recognizers.append(self)

        def start_continuous_recognition_async(self):
            for text in recognized_texts:
                self.recognized.emit(
                    types.SimpleNamespace(result=types.SimpleNamespace(text=text))
                )
            if cancellation is not None:
                self.canceled.emit(
                    types.SimpleNamespace(
                        reason=cancellation_reason,
                        error_code=cancellation_error_code,
                        error_details=cancellation,
                        result=types.SimpleNamespace(text=""),
                    )
                )
            elif stop_session:
                self.session_stopped.emit(types.SimpleNamespace())
            return FakeFuture()

        def stop_continuous_recognition_async(self):
            self.stopped = True
            return FakeFuture()

    class CancellationDetails:
        def __new__(cls, result):
            return types.SimpleNamespace(
                reason=details_reason,
                error_code=details_error_code,
                error_details=details_error_details,
            )

    speech_module = types.ModuleType("azure.cognitiveservices.speech")
    speech_module.SpeechConfig = SpeechConfig
    speech_module.SpeechRecognizer = SpeechRecognizer
    speech_module.CancellationDetails = CancellationDetails
    speech_module.audio = types.SimpleNamespace(
        AudioStreamFormat=AudioStreamFormat,
        PushAudioInputStream=PushAudioInputStream,
        AudioConfig=AudioConfig,
    )
    return speech_module, state


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
                self.language = None

            def transcribe(self, *args, **kwargs):
                raise AssertionError("static transcribe should not be called")

            def transcribe_streaming(self, *args, **kwargs):
                self.streaming_called = True
                self.language = kwargs.get("language")
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
                language="unknown",
            )

        self.assertEqual(transcript, "streamed")
        self.assertTrue(provider.streaming_called)
        self.assertEqual(provider.language, "unknown")

    def test_soniox_realtime_model_forces_streaming(self):
        run_eval = load_run_eval("run_eval")

        class FakeProvider:
            def __init__(self):
                self.streaming_called = False

            def force_streaming_for_model(self, model_variant):
                return model_variant == "stt-rt-v5"

            def transcribe(self, *args, **kwargs):
                raise AssertionError("static transcribe should not be called")

            def transcribe_streaming(self, *args, **kwargs):
                self.streaming_called = True
                return "streamed"

        provider = FakeProvider()
        with mock.patch.object(
            run_eval, "get_provider", return_value=(provider, "stt-rt-v5")
        ):
            transcript = run_eval.transcribe_with_retry(
                "soniox/stt-rt-v5",
                "/tmp/audio.wav",
                {"audio": {"array": [], "sampling_rate": 16000}},
                streaming=False,
            )

        self.assertEqual(transcript, "streamed")
        self.assertTrue(provider.streaming_called)

    def test_soniox_async_model_keeps_static_mode(self):
        run_eval = load_run_eval("run_eval")

        class FakeProvider:
            def __init__(self):
                self.static_called = False

            def force_streaming_for_model(self, model_variant):
                return model_variant == "stt-rt-v5"

            def transcribe(self, *args, **kwargs):
                self.static_called = True
                return "static"

            def transcribe_streaming(self, *args, **kwargs):
                raise AssertionError("streaming transcribe should not be called")

        provider = FakeProvider()
        with mock.patch.object(
            run_eval, "get_provider", return_value=(provider, "stt-async-v5")
        ):
            transcript = run_eval.transcribe_with_retry(
                "soniox/stt-async-v5",
                "/tmp/audio.wav",
                {"audio": {"array": [], "sampling_rate": 16000}},
                streaming=False,
            )

        self.assertEqual(transcript, "static")
        self.assertTrue(provider.static_called)

    def test_forced_streaming_rejects_url_audio(self):
        run_eval = load_run_eval("run_eval")

        class FakeProvider:
            def force_streaming_for_model(self, model_variant):
                return model_variant == "stt-rt-v5"

            def transcribe(self, *args, **kwargs):
                raise AssertionError("static transcribe should not be called")

            def transcribe_streaming(self, *args, **kwargs):
                raise AssertionError("streaming transcribe should not be called")

        with mock.patch.object(
            run_eval, "get_provider", return_value=(FakeProvider(), "stt-rt-v5")
        ):
            with self.assertRaisesRegex(ValueError, "--streaming requires local audio"):
                run_eval.transcribe_with_retry(
                    "soniox/stt-rt-v5",
                    None,
                    {"row": {"audio": [{"src": "https://example.com/audio.wav"}]}},
                    use_url=True,
                    streaming=False,
                )

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

    def test_vad_use_url_is_rejected(self):
        run_eval = load_run_eval("run_eval")

        with self.assertRaisesRegex(ValueError, "--vad_position requires local audio"):
            run_eval.transcribe_dataset(
                "dataset",
                "config",
                "test",
                "deepgram/nova-3",
                use_url=True,
                args=types.SimpleNamespace(
                    audio_preprocessor="none",
                    vad_position="pre",
                ),
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

    def test_soniox_streaming_model_typo_fails_with_hint(self):
        load_providers()
        from providers import PermanentError, soniox_provider

        provider = soniox_provider.SonioxProvider()
        with mock.patch.dict(os.environ, {"SONIOX_API_KEY": "key"}):
            with self.assertRaisesRegex(
                PermanentError,
                "Did you mean 'stt-rt-v5'",
            ):
                provider.transcribe_streaming(
                    "stt-rtt-v5",
                    "/tmp/audio.wav",
                    {},
                )

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

    def test_sarvam_registration_and_static_request(self):
        providers = load_providers()
        from providers import sarvam_provider

        provider, variant = providers.get_provider("sarvam/saaras:v3")
        self.assertEqual(provider.__class__.__name__, "SarvamProvider")
        self.assertEqual(variant, "saaras:v3")

        response = mock.Mock(status_code=200, text="ok")
        response.json.return_value = {"transcript": "hello"}
        response.raise_for_status.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(b"RIFF-test-audio")
            audio_file.flush()
            with mock.patch.dict(os.environ, {"SARVAM_API_KEY": "key"}):
                with mock.patch.object(
                    sarvam_provider,
                    "_audio_info",
                    return_value=types.SimpleNamespace(duration=10, samplerate=16000),
                ):
                    with mock.patch.object(
                        sarvam_provider.requests,
                        "post",
                        return_value=response,
                    ) as post:
                        transcript = provider.transcribe(
                            variant,
                            audio_file.name,
                            {},
                            language="unknown",
                        )

        self.assertEqual(transcript, "hello")
        request = post.call_args.kwargs
        self.assertEqual(request["headers"]["Api-Subscription-Key"], "key")
        self.assertEqual(
            request["data"],
            {
                "model": "saaras:v3",
                "mode": "transcribe",
                "language_code": "unknown",
            },
        )
        self.assertEqual(request["files"]["file"][2], "audio/wav")

    def test_sarvam_normalizes_english_and_rejects_long_static_audio(self):
        providers = load_providers()
        from providers import PermanentError, sarvam_provider

        self.assertEqual(sarvam_provider._language_code("en"), "en-IN")
        self.assertEqual(sarvam_provider._language_code("ta-IN"), "ta-IN")
        self.assertEqual(sarvam_provider._language_code("unknown"), "unknown")

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            with mock.patch.dict(os.environ, {"SARVAM_API_KEY": "key"}):
                with mock.patch.object(
                    sarvam_provider,
                    "_audio_info",
                    return_value=types.SimpleNamespace(duration=30.1, samplerate=16000),
                ):
                    with self.assertRaisesRegex(PermanentError, "up to 30 seconds"):
                        sarvam_provider.SarvamProvider().transcribe(
                            "saaras:v3", audio_file.name, {}
                        )

    def test_sarvam_http_client_errors_are_permanent_except_rate_limits(self):
        load_providers()
        from providers import PermanentError, sarvam_provider

        client_error = mock.Mock(status_code=422, text="invalid audio")
        with self.assertRaisesRegex(PermanentError, "422: invalid audio"):
            sarvam_provider._raise_for_response(client_error)

        rate_limited = mock.Mock(status_code=429, text="slow down")
        rate_limited.raise_for_status.side_effect = RuntimeError("retry")
        with self.assertRaisesRegex(RuntimeError, "retry"):
            sarvam_provider._raise_for_response(rate_limited)

    def test_sarvam_streaming_sends_audio_flush_and_auto_detection(self):
        load_providers()
        from providers import sarvam_provider

        fake_ws = FakeWebSocket(
            [json.dumps({"type": "data", "data": {"transcript": "hello world"}})]
        )
        audio = b"wav-bytes"
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(audio)
            audio_file.flush()
            with mock.patch.object(
                sarvam_provider, "connect_websocket", return_value=fake_ws
            ) as connect:
                transcript = asyncio.run(
                    sarvam_provider._transcribe_streaming(
                        audio_file.name,
                        "key",
                        "saaras:v3",
                        "unknown",
                    )
                )

        self.assertEqual(transcript, "hello world")
        url = connect.call_args.args[0]
        self.assertIn("language-code=unknown", url)
        self.assertEqual(
            connect.call_args.kwargs["headers"], {"Api-Subscription-Key": "key"}
        )
        audio_message = json.loads(fake_ws.sent[0])
        self.assertEqual(
            base64.b64decode(audio_message["audio"]["data"]), audio
        )
        self.assertEqual(audio_message["audio"]["encoding"], "audio/wav")
        self.assertEqual(fake_ws.sent[-1], json.dumps({"type": "flush"}))

    def test_sarvam_streaming_error_is_permanent(self):
        load_providers()
        from providers import PermanentError, sarvam_provider

        fake_ws = FakeWebSocket(
            [json.dumps({"type": "error", "data": {"error": "bad audio"}})]
        )
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(b"audio")
            audio_file.flush()
            with mock.patch.object(
                sarvam_provider, "connect_websocket", return_value=fake_ws
            ):
                with self.assertRaisesRegex(PermanentError, "bad audio"):
                    asyncio.run(
                        sarvam_provider._transcribe_streaming(
                            audio_file.name,
                            "key",
                            "saaras:v3",
                            "en",
                        )
                    )

    def test_microsoft_streams_pcm16_and_collects_final_results(self):
        providers = load_providers()
        from providers import microsoft_azure_provider

        speechsdk, state = fake_azure_speech_sdk(["hello", "world"])
        azure_module = types.ModuleType("azure")
        cognitive_module = types.ModuleType("azure.cognitiveservices")
        azure_module.cognitiveservices = cognitive_module
        cognitive_module.speech = speechsdk

        with mock.patch.dict(
            sys.modules,
            {
                "azure": azure_module,
                "azure.cognitiveservices": cognitive_module,
                "azure.cognitiveservices.speech": speechsdk,
            },
        ), mock.patch.dict(
            os.environ,
            {
                "AZURE_SPEECH_ENDPOINT": "https://example.cognitiveservices.azure.com/",
                "AZURE_SPEECH_KEY": "key",
            },
            clear=True,
        ), mock.patch.object(
            microsoft_azure_provider.os.path, "isfile", return_value=True
        ), mock.patch.object(
            microsoft_azure_provider,
            "pcm16_chunks",
            return_value=[b"first", b"second"],
        ):
            result = microsoft_azure_provider.MicrosoftAzureProvider().transcribe_streaming(
                "MAI-Transcribe-1.5",
                "/tmp/audio.wav",
                {},
                language="en",
            )

        self.assertEqual(result, "hello world")
        self.assertEqual(
            state.formats,
            [{"samples_per_second": 16000, "bits_per_sample": 16, "channels": 1}],
        )
        self.assertEqual(state.streams[0].writes, [b"first", b"second"])
        self.assertTrue(state.streams[0].closed)
        self.assertEqual(state.configs[0].speech_recognition_language, "en-US")
        self.assertTrue(state.recognizers[0].stopped)

    def test_microsoft_provider_registration_and_input_guards(self):
        providers = load_providers()
        provider, variant = providers.get_provider(
            "microsoft/MAI-Transcribe-1.5"
        )
        self.assertEqual(variant, "MAI-Transcribe-1.5")
        self.assertEqual(provider.__class__.__name__, "MicrosoftAzureProvider")

        with self.assertRaisesRegex(providers.PermanentError, "only supports"):
            provider.transcribe_streaming("other-model", "/tmp/audio.wav", {})
        with self.assertRaisesRegex(providers.PermanentError, "local audio"):
            provider.transcribe_streaming(
                "MAI-Transcribe-1.5", None, {}, use_url=True
            )
        with self.assertRaisesRegex(providers.PermanentError, "audio file path"):
            provider.transcribe_streaming("MAI-Transcribe-1.5", None, {})

    def test_microsoft_reports_cancellation_and_timeout(self):
        providers = load_providers()
        from providers import microsoft_azure_provider

        for cancellation, expected, reason, error_code in [
            (
                "",
                "reason=Error; error_code=AuthenticationFailure; "
                "error_details=authentication failed",
                "ResultReason.Canceled",
                None,
            ),
            (None, "timed out", None, None),
        ]:
            speechsdk, _ = fake_azure_speech_sdk(
                cancellation=cancellation,
                stop_session=False,
                cancellation_reason=reason,
                cancellation_error_code=error_code,
                details_reason="Error" if reason else None,
                details_error_code="AuthenticationFailure" if reason else None,
                details_error_details="authentication failed" if reason else None,
            )
            azure_module = types.ModuleType("azure")
            cognitive_module = types.ModuleType("azure.cognitiveservices")
            azure_module.cognitiveservices = cognitive_module
            cognitive_module.speech = speechsdk
            with mock.patch.dict(
                sys.modules,
                {
                    "azure": azure_module,
                    "azure.cognitiveservices": cognitive_module,
                    "azure.cognitiveservices.speech": speechsdk,
                },
            ), mock.patch.dict(
                os.environ,
                {
                    "AZURE_SPEECH_ENDPOINT": "https://example/",
                    "AZURE_API_KEY": "fallback-key",
                },
                clear=True,
            ), mock.patch.object(
                microsoft_azure_provider.os.path, "isfile", return_value=True
            ), mock.patch.object(
                microsoft_azure_provider, "pcm16_chunks", return_value=[]
            ), mock.patch.object(
                microsoft_azure_provider.MicrosoftAzureProvider,
                "STREAMING_TIMEOUT_SECONDS",
                0.001,
            ):
                with self.assertRaisesRegex(providers.PermanentError, expected):
                    microsoft_azure_provider.MicrosoftAzureProvider().transcribe_streaming(
                        "MAI-Transcribe-1.5", "/tmp/audio.wav", {}
                    )

    def test_microsoft_treats_end_of_stream_cancellation_as_completion(self):
        providers = load_providers()
        from providers import microsoft_azure_provider

        speechsdk, _ = fake_azure_speech_sdk(
            recognized_texts=["hello"],
            cancellation="",
            stop_session=False,
            cancellation_reason="ResultReason.Canceled",
            details_reason="CancellationReason.EndOfStream",
        )
        azure_module = types.ModuleType("azure")
        cognitive_module = types.ModuleType("azure.cognitiveservices")
        azure_module.cognitiveservices = cognitive_module
        cognitive_module.speech = speechsdk
        with mock.patch.dict(
            sys.modules,
            {
                "azure": azure_module,
                "azure.cognitiveservices": cognitive_module,
                "azure.cognitiveservices.speech": speechsdk,
            },
        ), mock.patch.dict(
            os.environ,
            {
                "AZURE_SPEECH_ENDPOINT": "https://example/",
                "AZURE_API_KEY": "fallback-key",
            },
            clear=True,
        ), mock.patch.object(
            microsoft_azure_provider.os.path, "isfile", return_value=True
        ), mock.patch.object(
            microsoft_azure_provider, "pcm16_chunks", return_value=[]
        ):
            result = microsoft_azure_provider.MicrosoftAzureProvider().transcribe_streaming(
                "MAI-Transcribe-1.5", "/tmp/audio.wav", {}
            )

        self.assertEqual(result, "hello")

    def test_microsoft_requires_streaming_credentials(self):
        providers = load_providers()
        from providers import microsoft_azure_provider

        provider, _ = providers.get_provider("microsoft/MAI-Transcribe-1.5")
        with mock.patch.object(
            microsoft_azure_provider.os.path, "isfile", return_value=True
        ), mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "AZURE_SPEECH_ENDPOINT"):
                provider.transcribe_streaming(
                    "MAI-Transcribe-1.5", "/tmp/audio.wav", {}
                )


if __name__ == "__main__":
    unittest.main()
