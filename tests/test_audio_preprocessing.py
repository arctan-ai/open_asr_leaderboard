import builtins
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


def load_data_utils():
    repo_root = Path(__file__).resolve().parents[1]

    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    datasets_stub.Audio = lambda *args, **kwargs: ("Audio", args, kwargs)
    datasets_stub.IterableDataset = type("IterableDataset", (), {})

    num2words_stub = types.ModuleType("num2words")
    num2words_stub.num2words = lambda value, lang=None: str(value)

    normalizer_stub = types.ModuleType("normalizer")
    normalizer_stub.__path__ = [str(repo_root / "normalizer")]
    normalizer_stub.EnglishTextNormalizer = lambda: (lambda value: value)

    class BasicMultilingualTextNormalizer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, value):
            return value

    normalizer_stub.BasicMultilingualTextNormalizer = BasicMultilingualTextNormalizer

    eval_utils_stub = types.ModuleType("normalizer.eval_utils")
    eval_utils_stub.read_manifest = lambda *args, **kwargs: None
    eval_utils_stub.write_manifest = lambda *args, **kwargs: None
    eval_utils_stub.normalize_compound_pairs = lambda *args, **kwargs: None
    eval_utils_stub.post_slack_single_run_summary = lambda *args, **kwargs: None
    eval_utils_stub.post_slack_run_started = lambda *args, **kwargs: None
    eval_utils_stub.post_slack_run_failed = lambda *args, **kwargs: None

    with mock.patch.dict(
        sys.modules,
        {
            "datasets": datasets_stub,
            "num2words": num2words_stub,
            "normalizer": normalizer_stub,
            "normalizer.eval_utils": eval_utils_stub,
        },
    ):
        return importlib.import_module("normalizer.data_utils")


data_utils = load_data_utils()


class FakeProcessorConfig:
    def __init__(self, sample_rate, num_channels, num_frames):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.num_frames = num_frames


class FakeProcessor:
    instances = []

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.processed_shapes = []
        FakeProcessor.instances.append(self)

    def process(self, chunk):
        self.processed_shapes.append(chunk.shape)
        return chunk + 0.25

    def close(self):
        self.closed = True


class AudioPreprocessingTest(unittest.TestCase):
    def setUp(self):
        FakeProcessor.instances = []

    def test_none_preprocessor_is_noop(self):
        self.assertIsNone(data_utils._build_audio_preprocess_fn("none", 10))

    def test_arctan_processing_preserves_length_and_sample_rate(self):
        audio = {
            "array": np.zeros(5, dtype=np.float32),
            "sampling_rate": 1000,
        }

        processed = data_utils._process_audio_with_arctan(
            audio,
            FakeProcessor,
            FakeProcessorConfig,
            chunk_ms=2,
        )

        np.testing.assert_allclose(
            processed["array"],
            np.full(5, 0.25, dtype=np.float32),
        )
        self.assertEqual(processed["sampling_rate"], 1000)
        self.assertEqual(processed["array"].dtype, np.float32)
        self.assertEqual(FakeProcessor.instances[0].config.num_frames, 2)
        self.assertEqual(
            FakeProcessor.instances[0].processed_shapes,
            [(1, 2), (1, 2), (1, 2)],
        )
        self.assertTrue(FakeProcessor.instances[0].closed)

    def test_multichannel_audio_is_mixed_to_mono(self):
        audio = {
            "array": np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
            "sampling_rate": 1000,
        }

        processed = data_utils._process_audio_with_arctan(
            audio,
            FakeProcessor,
            FakeProcessorConfig,
            chunk_ms=2,
        )

        np.testing.assert_allclose(
            processed["array"],
            np.array([0.75, 0.75], dtype=np.float32),
        )

    def test_missing_arctan_dependency_fails_only_when_loaded(self):
        real_import = builtins.__import__

        def import_without_arctan(name, *args, **kwargs):
            if name == "arctan":
                raise ImportError("missing arctan")
            return real_import(name, *args, **kwargs)

        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = mock.Mock()

        with mock.patch.dict(sys.modules, {"dotenv": dotenv_stub}):
            with mock.patch("builtins.__import__", side_effect=import_without_arctan):
                with self.assertRaisesRegex(RuntimeError, "arctan-vi"):
                    data_utils._build_audio_preprocess_fn("arctan", 10)

    def test_missing_arctan_key_is_rejected(self):
        arctan_stub = types.ModuleType("arctan")
        arctan_stub.Processor = FakeProcessor
        arctan_stub.ProcessorConfig = FakeProcessorConfig
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = mock.Mock()

        with mock.patch.dict(
            sys.modules, {"arctan": arctan_stub, "dotenv": dotenv_stub}
        ):
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "ARCTAN_SDK_KEY"):
                    data_utils._build_audio_preprocess_fn("arctan", 10)
        dotenv_stub.load_dotenv.assert_called_once_with()

    def test_arctan_loader_reads_dotenv_before_returning_processor(self):
        arctan_stub = types.ModuleType("arctan")
        arctan_stub.Processor = FakeProcessor
        arctan_stub.ProcessorConfig = FakeProcessorConfig
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = mock.Mock()

        with mock.patch.dict(
            sys.modules, {"arctan": arctan_stub, "dotenv": dotenv_stub}
        ):
            with mock.patch.dict(
                "os.environ", {"ARCTAN_SDK_KEY": "test-key"}, clear=True
            ):
                processor_cls, config_cls = data_utils._load_arctan_processor()

        self.assertIs(processor_cls, FakeProcessor)
        self.assertIs(config_cls, FakeProcessorConfig)
        dotenv_stub.load_dotenv.assert_called_once_with()

    def test_invalid_chunk_ms_is_rejected(self):
        audio = {
            "array": np.zeros(5, dtype=np.float32),
            "sampling_rate": 1000,
        }

        with self.assertRaisesRegex(ValueError, "arctan_chunk_ms"):
            data_utils._process_audio_with_arctan(
                audio,
                FakeProcessor,
                FakeProcessorConfig,
                chunk_ms=0,
            )


if __name__ == "__main__":
    unittest.main()
