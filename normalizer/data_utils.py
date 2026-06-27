import re
import os

import num2words
from datasets import load_dataset, Audio, IterableDataset
from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer

from .eval_utils import read_manifest, write_manifest, normalize_compound_pairs


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


SUPPORTED_AUDIO_PREPROCESSORS = ("none", "arctan")


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


def _get_arg(args, name, default):
    return getattr(args, name, default) if args is not None else default


def _load_arctan_processor():
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "--audio_preprocessor arctan loads ARCTAN_SDK_KEY from .env using "
            "python-dotenv. Install 'python-dotenv' in the eval environment."
        ) from exc

    load_dotenv()

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


def _build_audio_preprocess_fn(audio_preprocessor, arctan_chunk_ms):
    if audio_preprocessor == "none":
        return None

    if audio_preprocessor != "arctan":
        raise ValueError(
            f"Unsupported audio preprocessor: {audio_preprocessor}. "
            f"Expected one of {SUPPORTED_AUDIO_PREPROCESSORS}."
        )

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

    # Re-sample and normalize transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    # NOTE (ebezzam) don't load from cache to account for potential changes in normalization logic
    # IterableDataset (streaming) has no cache, so the kwarg is only needed for Dataset
    map_kwargs = (
        {} if isinstance(dataset, IterableDataset) else {"load_from_cache_file": False}
    )
    audio_preprocess_fn = _build_audio_preprocess_fn(
        audio_preprocessor, arctan_chunk_ms
    )
    if audio_preprocess_fn is not None:
        dataset = dataset.map(audio_preprocess_fn, batched=True, **map_kwargs)
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
