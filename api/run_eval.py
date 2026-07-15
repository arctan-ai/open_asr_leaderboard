import argparse
import json
from typing import Optional
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
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
import concurrent.futures
from datetime import datetime, timezone
from providers import get_provider, PermanentError

load_dotenv()


def effective_streaming_for_model(model_name: str, streaming: bool) -> bool:
    provider, variant = get_provider(model_name)
    return streaming or provider.force_streaming_for_model(variant)


def fetch_audio_urls(dataset_path, dataset, split, batch_size=100, max_retries=20):
    API_URL = "https://datasets-server.huggingface.co/rows"

    headers = {}
    if os.environ.get("HF_TOKEN") is not None:
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    else:
        print("HF_TOKEN not set, might experience rate-limiting.")

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={dataset}&split={split}"
    size_response = requests.get(size_url, headers=headers).json()
    total_rows = size_response["size"]["config"]["num_rows"]
    audio_urls = []
    for offset in tqdm(range(0, total_rows, batch_size), desc="Fetching audio URLs"):
        params = {
            "dataset": dataset_path,
            "config": dataset,
            "split": split,
            "offset": offset,
            "length": min(batch_size, total_rows - offset),
        }

        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(API_URL, params=params, headers=headers)
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
        try:
            transcribe_fn = (
                provider.transcribe_streaming
                if effective_streaming
                else provider.transcribe
            )
            return transcribe_fn(variant, audio_file_path, sample, **kwargs)
        except PermanentError:
            raise
        except Exception as e:
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
    dataset,
    split,
    model_name,
    language="en",
    use_url=False,
    streaming=False,
    max_samples=None,
    max_workers=4,
    prompt=None,
    args=None,
    output_dir="./results",
):
    started_at = datetime.now(timezone.utc).isoformat()
    effective_streaming = effective_streaming_for_model(model_name, streaming)
    if effective_streaming and use_url:
        raise ValueError("--streaming requires local audio; do not use --use_url")

    if use_url:
        if getattr(args, "audio_preprocessor", "none") != "none":
            raise ValueError(
                "--audio_preprocessor requires local audio; do not use --use_url"
            )
        if getattr(args, "vad_position", "none") != "none":
            raise ValueError("--vad_position requires local audio; do not use --use_url")
        audio_rows = fetch_audio_urls(dataset_path, dataset, split)
        if max_samples:
            audio_rows = itertools.islice(audio_rows, max_samples)
        ds = audio_rows
    else:
        ds = datasets.load_dataset(dataset_path, dataset, split=split, streaming=False)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        ds = data_utils.prepare_data(ds, args=args)

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    mode = "streaming" if effective_streaming else "static"
    print(f"Transcribing with model: {model_name}, language: {language} ({mode})")

    def process_sample(sample):
        if use_url:
            reference = sample["row"]["text"].strip() or " "
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
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                transcription = ""

        else:
            reference = sample.get("original_text", "").strip() or " "
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
                )
            except Exception as e:
                print(f"Failed to transcribe after retries: {e}")
                transcription = ""
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

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
            reference, transcription, audio_duration, transcription_time = (
                future.result()
            )
            results["predictions"].append(transcription)
            results["references"].append(reference)
            results["audio_length_s"].append(audio_duration)
            results["transcription_time_s"].append(transcription_time)

    manifest_path = data_utils.write_manifest(
        results["references"],
        results["predictions"],
        model_name.replace("/", "-"),
        dataset_path,
        dataset,
        split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
        basedir=output_dir,
    )

    print("Results saved at path:", manifest_path)

    norm_refs = [data_utils.normalizer(r) or " " for r in results["references"]]
    norm_preds = [data_utils.normalizer(p) or " " for p in results["predictions"]]
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(references=norm_refs, predictions=norm_preds)
    wer_percent = round(100 * wer, 2)
    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2
    )

    print("WER:", wer_percent, "%")
    print("RTFx:", rtfx)
    data_utils.post_slack_single_run_summary(
        manifest_path=manifest_path,
        model_name=model_name,
        dataset_path=dataset_path,
        dataset_name=dataset,
        split=split,
        wer_percent=wer_percent,
        rtfx=rtfx,
        num_samples=len(results["references"]),
        audio_preprocessor=getattr(args, "audio_preprocessor", "none"),
        vad_position=getattr(args, "vad_position", "none"),
        streaming=effective_streaming,
    )
    summary = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset_path": dataset_path,
        "dataset": dataset,
        "split": split,
        "language": language,
        "audio_preprocessor": getattr(args, "audio_preprocessor", "none"),
        "vad_position": getattr(args, "vad_position", "none"),
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
        description="Unified Transcription Script with Concurrency"
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--language",
        default="en",
        help="Language code passed to the provider; use 'unknown' for Sarvam auto-detection",
    )
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
    data_utils.add_audio_preprocessor_args(parser)

    args = parser.parse_args()
    effective_streaming = effective_streaming_for_model(args.model_name, args.streaming)
    if effective_streaming and args.use_url:
        parser.error("--streaming requires local audio; do not use --use_url")
    if args.use_url and args.audio_preprocessor != "none":
        parser.error("--audio_preprocessor requires local audio; do not use --use_url")
    if args.use_url and args.vad_position != "none":
        parser.error("--vad_position requires local audio; do not use --use_url")

    data_utils.post_slack_run_started(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        audio_preprocessor=args.audio_preprocessor,
        vad_position=args.vad_position,
        streaming=effective_streaming,
    )
    try:
        transcribe_dataset(
            dataset_path=args.dataset_path,
            dataset=args.dataset,
            split=args.split,
            model_name=args.model_name,
            language=args.language,
            use_url=args.use_url,
            streaming=effective_streaming,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            prompt=args.prompt,
            args=args,
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
                    "dataset": args.dataset,
                    "split": args.split,
                    "language": args.language,
                    "audio_preprocessor": args.audio_preprocessor,
                    "vad_position": args.vad_position,
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
            dataset_name=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            audio_preprocessor=args.audio_preprocessor,
            vad_position=args.vad_position,
            error=exc,
            streaming=effective_streaming,
        )
        raise
