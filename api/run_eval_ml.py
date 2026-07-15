import argparse
import json
from typing import Optional
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
from datasets import Audio
import evaluate
import soundfile as sf
import tempfile
import time
import os
import requests
import itertools
from tqdm import tqdm
from dotenv import load_dotenv
from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs
import concurrent.futures
import threading
from datetime import datetime, timezone
from providers import get_provider, PermanentError, is_rate_limit_error

load_dotenv()


def effective_streaming_for_model(model_name: str, streaming: bool) -> bool:
    provider, variant = get_provider(model_name)
    return streaming or provider.force_streaming_for_model(variant)


def fetch_audio_urls(dataset_path, config_name, split, batch_size=100, max_retries=20):
    API_URL = "https://datasets-server.huggingface.co/rows"

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={config_name}&split={split}"
    size_response = requests.get(size_url).json()
    total_rows = size_response["size"]["config"]["num_rows"]
    for offset in tqdm(range(0, total_rows, batch_size), desc="Fetching audio URLs"):
        params = {
            "dataset": dataset_path,
            "config": config_name,
            "split": split,
            "offset": offset,
            "length": min(batch_size, total_rows - offset),
        }

        retries = 0
        while retries <= max_retries:
            try:
                headers = {}
                if os.environ.get("HF_TOKEN") is not None:
                    headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
                else:
                    print("HF_TOKEN not set, might experience rate-limiting.")
                response = requests.get(API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                yield from data["rows"]
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                retries += 1
                print(
                    f"Error fetching data: {e}, retrying ({retries}/{max_retries})..."
                )
                time.sleep(10)
                if retries >= max_retries:
                    raise Exception("Max retries exceeded while fetching data.")


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
    streaming=False,
    language="en",
    prompt=None,
    stop_event=None,
):
    provider, variant = get_provider(model_name)
    effective_streaming = streaming or provider.force_streaming_for_model(variant)
    if effective_streaming and use_url:
        raise ValueError("--streaming requires local audio; do not use --use_url")

    kwargs = dict(use_url=use_url, language=language)
    if prompt is not None:
        kwargs["prompt"] = prompt
    retries = 0
    while retries <= max_retries:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Transcription cancelled after rate limit failure")
        try:
            transcribe_fn = (
                provider.transcribe_streaming
                if effective_streaming
                else provider.transcribe
            )
            return transcribe_fn(variant, audio_file_path, sample, **kwargs)
        except PermanentError as e:
            if is_rate_limit_error(e) and stop_event is not None:
                stop_event.set()
            raise
        except Exception as e:
            if is_rate_limit_error(e):
                if stop_event is not None:
                    stop_event.set()
                raise
            retries += 1
            if retries > max_retries:
                raise e

            if not use_url:
                sf.write(
                    audio_file_path,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
            delay = 1
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)


def transcribe_dataset(
    dataset_path,
    config_name,
    split,
    model_name,
    language,
    use_url=False,
    streaming=False,
    max_samples=None,
    max_workers=4,
    prompt=None,
    output_dir="./results",
):
    started_at = datetime.now(timezone.utc).isoformat()
    effective_streaming = effective_streaming_for_model(model_name, streaming)
    if effective_streaming and use_url:
        raise ValueError("--streaming requires local audio; do not use --use_url")

    if use_url:
        audio_rows = fetch_audio_urls(dataset_path, config_name, split)
        if max_samples:
            audio_rows = itertools.islice(audio_rows, max_samples)
        ds = audio_rows
    else:
        ds = datasets.load_dataset(
            dataset_path, config_name, split=split, streaming=False
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }
    stop_event = threading.Event()

    print(
        f"Transcribing with model: {model_name}, language: {language}, "
        f"config: {config_name}, mode: {'streaming' if effective_streaming else 'static'}"
    )

    def process_sample(sample):
        if stop_event.is_set():
            return None
        if use_url:
            reference = sample["row"]["text"].strip()
            audio_duration = sample["row"]["audio_length_s"]
            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name,
                    None,
                    sample,
                    use_url=True,
                    streaming=effective_streaming,
                    language=language,
                    prompt=prompt,
                    stop_event=stop_event,
                )
            except Exception as e:
                if is_rate_limit_error(e):
                    stop_event.set()
                    raise
                print(f"Failed to transcribe after retries: {e}")
                return None

        else:
            reference = sample.get("text", "").strip()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(
                    tmpfile.name,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
                tmp_path = tmpfile.name
                audio_duration = (
                    len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
                )

            start = time.time()
            try:
                transcription = transcribe_with_retry(
                    model_name,
                    tmp_path,
                    sample,
                    use_url=False,
                    streaming=effective_streaming,
                    language=language,
                    prompt=prompt,
                    stop_event=stop_event,
                )
            except Exception as e:
                if is_rate_limit_error(e):
                    stop_event.set()
                    raise
                print(f"Failed to transcribe after retries: {e}")
                os.unlink(tmp_path)
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                else:
                    print(f"File {tmp_path} does not exist")

        transcription_time = time.time() - start
        return reference, transcription, audio_duration, transcription_time

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(process_sample, sample): sample for sample in ds
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sample),
            total=len(future_to_sample),
            desc="Transcribing",
        ):
            try:
                result = future.result()
            except Exception:
                stop_event.set()
                for pending in future_to_sample:
                    pending.cancel()
                raise
            if result:
                reference, transcription, audio_duration, transcription_time = result
                results["predictions"].append(transcription)
                results["references"].append(reference)
                results["audio_length_s"].append(audio_duration)
                results["transcription_time_s"].append(transcription_time)

    # Filter empty references (consistent with English pipeline's prepare_data)
    filtered = [
        (ref, pred, dur, time_s)
        for ref, pred, dur, time_s in zip(
            results["references"],
            results["predictions"],
            results["audio_length_s"],
            results["transcription_time_s"],
        )
        if data_utils.is_target_text_in_range(ref)
    ]
    if filtered:
        (
            results["references"],
            results["predictions"],
            results["audio_length_s"],
            results["transcription_time_s"],
        ) = zip(*filtered)
        results = {k: list(v) for k, v in results.items()}

    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        model_name.replace("/", "-"),
        dataset_path,
        config_name,
        split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
        basedir=output_dir,
    )

    print("Results saved at path:", manifest_path)

    norm_refs = [
        data_utils.ml_normalizer(r, lang=language) for r in results["references"]
    ]
    norm_preds = [
        data_utils.ml_normalizer(t, lang=language) for t in results["predictions"]
    ]
    wer_metric = evaluate.load("wer")
    wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
    wer = wer_metric.compute(references=wer_refs, predictions=wer_preds)
    wer_percent = round(100 * wer, 2)
    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2
    )

    print("WER:", wer_percent, "%")
    print("RTFx:", rtfx)
    summary = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset_path": dataset_path,
        "dataset": config_name,
        "split": split,
        "language": language,
        "audio_preprocessor": "none",
        "vad_position": "none",
        "streaming": effective_streaming,
        "use_url": use_url,
        "num_samples": len(results["references"]),
        "wer_percent": wer_percent,
        "rtfx": rtfx,
        "manifest_path": str(Path(manifest_path).resolve()),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multilingual API Transcription Script with Concurrency"
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--config_name", required=True, help="Dataset config name, e.g. 'fleurs_de'"
    )
    parser.add_argument("--language", required=True, help="Language code, e.g. 'de'")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Prefix model name with provider prefix (e.g., 'assembly/', 'smallestai/', 'soniox/', or 'deepgram/')",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory for the manifest and structured summary.json output.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=300, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--use_url",
        action="store_true",
        help="Use URL-based audio fetching instead of datasets",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use the provider streaming ASR endpoint; requires local audio",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt to pass to the provider (e.g., 'Output must be in lexical format.')",
    )

    args = parser.parse_args()
    effective_streaming = effective_streaming_for_model(args.model_name, args.streaming)
    if effective_streaming and args.use_url:
        parser.error("--streaming requires local audio; do not use --use_url")

    data_utils.post_slack_run_started(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        dataset_name=args.config_name,
        split=args.split,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        audio_preprocessor="none",
        streaming=effective_streaming,
    )
    try:
        transcribe_dataset(
            dataset_path=args.dataset_path,
            config_name=args.config_name,
            split=args.split,
            model_name=args.model_name,
            language=args.language,
            use_url=args.use_url,
            streaming=effective_streaming,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            prompt=args.prompt,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "summary.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "model_name": args.model_name,
                    "dataset_path": args.dataset_path,
                    "dataset": args.config_name,
                    "split": args.split,
                    "language": args.language,
                    "audio_preprocessor": "none",
                    "vad_position": "none",
                    "streaming": effective_streaming,
                    "use_url": args.use_url,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        data_utils.post_slack_run_failed(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            dataset_name=args.config_name,
            split=args.split,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            audio_preprocessor="none",
            error=exc,
            streaming=effective_streaming,
        )
        raise
