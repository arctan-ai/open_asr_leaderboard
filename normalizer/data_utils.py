import re
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import num2words
from datasets import load_dataset, Audio, IterableDataset
from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer

from .eval_utils import (
    read_manifest,
    write_manifest,
    normalize_compound_pairs,
    post_slack_single_run_summary,
    post_slack_run_started,
    post_slack_run_failed,
)


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """BasicMultilingualTextNormalizer with optional number normalization.

    Call with just text for standard normalization (backward-compatible).
    Pass lang= to also convert digits to words via num2words.
    """

    def _normalize_numbers(self, text, lang):
        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s, lang=None):
        s = super().__call__(s)
        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


normalizer = EnglishTextNormalizer()

ml_normalizer = MultilingualNormalizer(remove_diacritics=False)


def normalize(batch):
    batch["original_text"] = get_text(batch)
    batch["norm_text"] = normalizer(batch["original_text"])
    return batch


SUPPORTED_AUDIO_PREPROCESSORS = ("none", "arctan", "ai_coustics_vfl_2_1")
AI_COUSTICS_FILTER = "aic-quail-vfl"
AI_COUSTICS_ENV_PROJECT_DIR = "AI_COUSTICS_NOISE_CANCELLER_DIR"
AI_COUSTICS_LIVEKIT_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


def add_audio_preprocessor_args(parser):
    parser.add_argument(
        "--audio_preprocessor",
        choices=SUPPORTED_AUDIO_PREPROCESSORS,
        default="none",
        help="Optional eval-time audio preprocessor to run before ASR.",
    )
    parser.add_argument(
        "--arctan_chunk_ms",
        type=int,
        default=10,
        help="Chunk size in milliseconds for --audio_preprocessor arctan.",
    )
    parser.add_argument(
        "--ai_coustics_project_dir",
        type=str,
        default=None,
        help=(
            "Path to livekit-examples/noise-canceller, or to an "
            "lk-noise-canceller-examples checkout containing that submodule. "
            f"Defaults to ${AI_COUSTICS_ENV_PROJECT_DIR}."
        ),
    )
    parser.add_argument(
        "--ai_coustics_sample_rate",
        type=int,
        default=16000,
        help="Output sample rate for --audio_preprocessor ai_coustics_vfl_2_1.",
    )
    parser.add_argument(
        "--ai_coustics_pad_seconds",
        type=float,
        default=2.0,
        help="Leading silence pad to add before ai-coustics processing and trim after.",
    )
    parser.add_argument(
        "--ai_coustics_enhancement_level",
        type=float,
        default=1.0,
        help="Enhancement level for ai-coustics VFL 2.1 preprocessing.",
    )
    parser.add_argument(
        "--audio_preprocessor_batch_size",
        type=int,
        default=1,
        help="Batch size for eval-time audio preprocessing.",
    )


def _get_arg(args, name, default):
    return getattr(args, name, default) if args is not None else default


def _load_dotenv(feature_name):
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            f"--audio_preprocessor {feature_name} loads credentials from .env using "
            "python-dotenv. Install 'python-dotenv' in the eval environment."
        ) from exc

    load_dotenv()


def _load_arctan_processor():
    _load_dotenv("arctan")

    try:
        from arctan import Processor, ProcessorConfig
    except ImportError as exc:
        raise RuntimeError(
            "--audio_preprocessor arctan requires the optional 'arctan-vi' package "
            "to be installed in the eval environment."
        ) from exc

    if not os.environ.get("ARCTAN_SDK_KEY"):
        raise RuntimeError(
            "--audio_preprocessor arctan requires ARCTAN_SDK_KEY to be set in "
            "the eval environment."
        )

    return Processor, ProcessorConfig


def _process_audio_with_arctan(audio, processor_cls, config_cls, chunk_ms=10):
    import numpy as np

    if chunk_ms <= 0:
        raise ValueError("--arctan_chunk_ms must be greater than 0")

    sample_rate = int(audio["sampling_rate"])
    samples = np.asarray(audio["array"], dtype=np.float32)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    original_num_samples = len(samples)
    chunk_size = max(1, int(round(sample_rate * chunk_ms / 1000)))

    config = config_cls(
        sample_rate=sample_rate,
        num_channels=1,
        num_frames=chunk_size,
    )
    processor = processor_cls(config=config)
    output_chunks = []

    try:
        for start in range(0, original_num_samples, chunk_size):
            chunk = samples[start : start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            processed = processor.process(chunk.reshape(1, -1))
            output_chunks.append(np.asarray(processed[0], dtype=np.float32))
    finally:
        processor.close()

    if output_chunks:
        processed_audio = np.concatenate(output_chunks)[:original_num_samples]
    else:
        processed_audio = samples

    return {
        "array": processed_audio.astype(np.float32, copy=False),
        "sampling_rate": sample_rate,
    }


def _resolve_ai_coustics_project_dir(project_dir):
    raw_path = project_dir or os.environ.get(AI_COUSTICS_ENV_PROJECT_DIR)
    if not raw_path:
        raise RuntimeError(
            "--audio_preprocessor ai_coustics_vfl_2_1 requires "
            f"--ai_coustics_project_dir or ${AI_COUSTICS_ENV_PROJECT_DIR} pointing "
            "to livekit-examples/noise-canceller."
        )

    candidate = Path(raw_path).expanduser()
    if (candidate / "noise-canceller.py").is_file():
        return candidate
    if (candidate / "noise-canceller" / "noise-canceller.py").is_file():
        return candidate / "noise-canceller"

    raise RuntimeError(
        "Could not find noise-canceller.py. Set "
        f"${AI_COUSTICS_ENV_PROJECT_DIR} or --ai_coustics_project_dir to either "
        "livekit-examples/noise-canceller or an lk-noise-canceller-examples "
        "checkout with its noise-canceller submodule initialized."
    )


def _load_ai_coustics_config(
    project_dir,
    sample_rate,
    pad_seconds,
    enhancement_level,
):
    _load_dotenv("ai_coustics_vfl_2_1")

    missing_env = [name for name in AI_COUSTICS_LIVEKIT_ENV if not os.environ.get(name)]
    if missing_env:
        raise RuntimeError(
            "--audio_preprocessor ai_coustics_vfl_2_1 requires "
            + ", ".join(missing_env)
            + " to be set in the eval environment."
        )

    if sample_rate <= 0:
        raise ValueError("--ai_coustics_sample_rate must be greater than 0")
    if pad_seconds < 0:
        raise ValueError("--ai_coustics_pad_seconds must be non-negative")
    if not 0.0 <= enhancement_level <= 1.0:
        raise ValueError("--ai_coustics_enhancement_level must be between 0.0 and 1.0")
    if shutil.which("uv") is None:
        raise RuntimeError(
            "--audio_preprocessor ai_coustics_vfl_2_1 requires the 'uv' executable."
        )

    project_path = _resolve_ai_coustics_project_dir(project_dir)
    return {
        "project_dir": project_path,
        "script": project_path / "noise-canceller.py",
        "sample_rate": sample_rate,
        "pad_seconds": pad_seconds,
        "enhancement_level": enhancement_level,
    }


def _process_audio_with_ai_coustics(audio, config):
    import numpy as np
    import soundfile as sf

    input_sample_rate = int(audio["sampling_rate"])
    samples = np.asarray(audio["array"], dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    output_sample_rate = int(config["sample_rate"])
    pad_samples = int(round(input_sample_rate * float(config["pad_seconds"])))
    output_trim_samples = int(round(output_sample_rate * float(config["pad_seconds"])))

    with tempfile.TemporaryDirectory(prefix="asr-ai-coustics-") as tmpdir:
        tmp_path = Path(tmpdir)
        padded_path = tmp_path / "input_padded.wav"
        output_path = tmp_path / "output.wav"

        if pad_samples:
            padded = np.concatenate(
                [np.zeros(pad_samples, dtype=samples.dtype), samples]
            )
        else:
            padded = samples
        sf.write(padded_path, padded, input_sample_rate)

        cmd = [
            "uv",
            "run",
            "--project",
            str(config["project_dir"]),
            str(config["script"]),
            str(padded_path),
            "--filter",
            AI_COUSTICS_FILTER,
            "-o",
            str(output_path),
            "--sample-rate",
            str(output_sample_rate),
            "--ai-coustics-enhancement-level",
            str(config["enhancement_level"]),
            "--direct",
            "--silent",
        ]
        env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "ai-coustics VFL 2.1 preprocessing failed with exit code "
                f"{result.returncode}: {stderr}"
            )

        processed, actual_sample_rate = sf.read(output_path, dtype="float32")

    if processed.ndim > 1:
        processed = processed.mean(axis=1)
    if output_trim_samples:
        processed = processed[output_trim_samples:]

    return {
        "array": np.asarray(processed, dtype=np.float32),
        "sampling_rate": int(actual_sample_rate),
    }


def _build_audio_preprocess_fn(
    audio_preprocessor,
    arctan_chunk_ms,
    ai_coustics_project_dir=None,
    ai_coustics_sample_rate=16000,
    ai_coustics_pad_seconds=2.0,
    ai_coustics_enhancement_level=1.0,
):
    if audio_preprocessor == "none":
        return None

    if audio_preprocessor == "arctan":
        processor_cls, config_cls = _load_arctan_processor()

        def preprocess_audio(batch):
            batch["audio"] = [
                _process_audio_with_arctan(
                    audio, processor_cls, config_cls, arctan_chunk_ms
                )
                for audio in batch["audio"]
            ]
            return batch

        return preprocess_audio

    if audio_preprocessor == "ai_coustics_vfl_2_1":
        config = _load_ai_coustics_config(
            ai_coustics_project_dir,
            ai_coustics_sample_rate,
            ai_coustics_pad_seconds,
            ai_coustics_enhancement_level,
        )

        def preprocess_audio(batch):
            batch["audio"] = [
                _process_audio_with_ai_coustics(audio, config)
                for audio in batch["audio"]
            ]
            return batch

        return preprocess_audio

    else:
        raise ValueError(
            f"Unsupported audio preprocessor: {audio_preprocessor}. "
            f"Expected one of {SUPPORTED_AUDIO_PREPROCESSORS}."
        )


def load_data(args):
    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )

    return dataset


def prepare_data(dataset, sampling_rate=16000, args=None):
    audio_preprocessor = _get_arg(args, "audio_preprocessor", "none")
    arctan_chunk_ms = _get_arg(args, "arctan_chunk_ms", 10)
    ai_coustics_project_dir = _get_arg(args, "ai_coustics_project_dir", None)
    ai_coustics_sample_rate = _get_arg(args, "ai_coustics_sample_rate", 16000)
    ai_coustics_pad_seconds = _get_arg(args, "ai_coustics_pad_seconds", 2.0)
    ai_coustics_enhancement_level = _get_arg(
        args, "ai_coustics_enhancement_level", 1.0
    )
    audio_preprocessor_batch_size = _get_arg(args, "audio_preprocessor_batch_size", 1)

    if audio_preprocessor_batch_size <= 0:
        raise ValueError("--audio_preprocessor_batch_size must be greater than 0")

    # Re-sample and normalize transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    # NOTE (ebezzam) don't load from cache to account for potential changes in normalization logic
    # IterableDataset (streaming) has no cache, so the kwarg is only needed for Dataset
    map_kwargs = (
        {} if isinstance(dataset, IterableDataset) else {"load_from_cache_file": False}
    )
    audio_preprocess_fn = _build_audio_preprocess_fn(
        audio_preprocessor,
        arctan_chunk_ms,
        ai_coustics_project_dir,
        ai_coustics_sample_rate,
        ai_coustics_pad_seconds,
        ai_coustics_enhancement_level,
    )
    if audio_preprocess_fn is not None:
        dataset = dataset.map(
            audio_preprocess_fn,
            batched=True,
            batch_size=audio_preprocessor_batch_size,
            **map_kwargs,
        )
    dataset = dataset.map(normalize, **map_kwargs)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    return dataset


AUDIO_FILEPATH_METADATA_KEYS = [
    "id",  # Main: https://huggingface.co/datasets/hf-audio/open-asr-leaderboard
    "file_name",  # Multilingual: https://huggingface.co/datasets/nithinraok/asr-leaderboard-datasets
    "file_name",  # Private
]


def _basename_or_none(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return os.path.basename(value)


def extract_audio_filepath_from_sample(sample):
    if sample is None:
        return None

    for key in AUDIO_FILEPATH_METADATA_KEYS:
        try:
            if key in sample:
                basename = _basename_or_none(sample[key])
                if basename is not None:
                    return basename
        except TypeError:
            # AudioDecoder / other non-mapping sample types are not subscriptable.
            return None
    return None


def extract_audio_filepaths_from_batch(batch, batch_size=None):
    if batch_size is None:
        if "audio" in batch:
            batch_size = len(batch["audio"])
        elif len(batch) > 0:
            first_value = next(iter(batch.values()))
            if isinstance(first_value, list):
                batch_size = len(first_value)

    if batch_size is None:
        return []

    for key in AUDIO_FILEPATH_METADATA_KEYS:
        values = batch.get(key)
        if isinstance(values, list) and len(values) == batch_size:
            return [_basename_or_none(v) for v in values]
    return [None] * batch_size
