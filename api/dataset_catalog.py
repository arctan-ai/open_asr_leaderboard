from __future__ import annotations

import csv
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DATASET_ROOT = REPO_ROOT.parent / "dataset"
DEFAULT_HF_DATASET_REPO = "bettercallaaryan/nc_agent_clips_openasr"
TRANSCRIPT_COLUMNS = (
    "text",
    "sentence",
    "normalized_text",
    "transcript",
    "transcription",
)
INVALID_DATASET_MESSAGE = (
    "This dataset does not follow the required format expected by the evaluator "
    "and cannot be selected"
)


def local_dataset_root() -> Path:
    return Path(
        os.environ.get("OPEN_ASR_LOCAL_DATASET_ROOT", DEFAULT_LOCAL_DATASET_ROOT)
    ).expanduser()


def huggingface_dataset_repo() -> str:
    return os.environ.get("OPEN_ASR_HF_DATASET_REPO", DEFAULT_HF_DATASET_REPO)


def source_options() -> list[dict]:
    return [
        {
            "id": "huggingface",
            "label": "Hugging Face",
            "kind": "huggingface",
            "description": huggingface_dataset_repo(),
        },
        {
            "id": "local",
            "label": "Local datasets",
            "kind": "local",
            "description": str(local_dataset_root()),
        },
    ]


def validation_error(reason: str) -> str:
    return f"{INVALID_DATASET_MESSAGE}: {reason}."


def validate_schema(features: list[str], splits: list[str]) -> str | None:
    feature_names = set(features)
    if "audio" not in feature_names:
        return validation_error("missing required 'audio' column")
    if not feature_names.intersection(TRANSCRIPT_COLUMNS):
        expected = ", ".join(f"'{name}'" for name in TRANSCRIPT_COLUMNS)
        return validation_error(
            f"missing a transcript column; expected one of {expected}"
        )
    if not splits:
        return validation_error("no usable splits were found")
    return None


def find_local_manifest(dataset_dir: Path) -> tuple[Path | None, str | None]:
    preferred = dataset_dir / "metadata.csv"
    if preferred.is_file():
        return preferred, None

    manifests = sorted(dataset_dir.glob("*.csv"), key=lambda path: path.name.casefold())
    if not manifests:
        return None, validation_error("no CSV manifest was found")
    if len(manifests) > 1:
        return None, validation_error(
            "multiple CSV manifests were found and none is named 'metadata.csv'"
        )
    return manifests[0], None


def read_csv_features(manifest: Path) -> list[str]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV manifest is empty") from exc
    features = [column.strip() for column in header if column.strip()]
    if not features:
        raise ValueError("CSV manifest has no columns")
    return features


def local_dataset_catalog() -> dict:
    root = local_dataset_root()
    if not root.is_dir():
        raise ValueError(f"Local dataset root does not exist: {root}")

    entries = []
    children = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name.casefold(),
    )
    for child in children:
        manifest, error = find_local_manifest(child)
        features: list[str] = []
        if manifest is not None:
            try:
                features = read_csv_features(manifest)
                error = validate_schema(features, ["test"])
            except Exception as exc:
                error = validation_error(str(exc))
        entries.append(
            {
                "id": child.name,
                "label": child.name,
                "dataset_source": "local",
                "dataset_path": child.name,
                "dataset": "default",
                "splits": ["test"] if manifest is not None else [],
                "features": features,
                "valid": error is None,
                "error": error,
            }
        )

    return {"source_id": "local", "datasets": entries}


def huggingface_dataset_catalog() -> dict:
    import datasets

    repo = huggingface_dataset_repo()
    token = os.environ.get("HF_TOKEN") or None
    configs = datasets.get_dataset_config_names(repo, token=token)
    entries = []
    for config in configs:
        try:
            splits = datasets.get_dataset_split_names(repo, config, token=token)
            builder = datasets.load_dataset_builder(repo, config, token=token)
            features = list(builder.info.features or {})
            error = validate_schema(features, splits)
        except Exception as exc:
            splits = []
            features = []
            error = validation_error(f"inspection failed: {exc}")
        entries.append(
            {
                "id": config,
                "label": config,
                "dataset_source": "huggingface",
                "dataset_path": repo,
                "dataset": config,
                "splits": splits,
                "features": features,
                "valid": error is None,
                "error": error,
            }
        )
    return {"source_id": "huggingface", "datasets": entries}


def dataset_catalog(source_id: str) -> dict:
    if source_id == "local":
        return local_dataset_catalog()
    if source_id == "huggingface":
        return huggingface_dataset_catalog()
    raise KeyError(source_id)


def resolve_local_dataset_dir(dataset_path: str) -> Path:
    relative = Path(dataset_path)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name.startswith("."):
        raise ValueError("Local dataset path must name an immediate child directory")

    root = local_dataset_root().resolve()
    candidate = (root / relative).resolve()
    if candidate.parent != root:
        raise ValueError("Local dataset path escapes the configured dataset root")
    if not candidate.is_dir():
        raise ValueError(f"Local dataset does not exist: {dataset_path}")
    return candidate


def load_evaluation_dataset(
    dataset_source: str,
    dataset_path: str,
    dataset: str,
    split: str,
):
    import datasets

    if dataset_source == "huggingface":
        return datasets.load_dataset(
            dataset_path,
            dataset,
            split=split,
            streaming=False,
        )
    if dataset_source != "local":
        raise ValueError(f"Unsupported dataset source: {dataset_source}")
    if dataset != "default" or split != "test":
        raise ValueError("Local datasets expose only config 'default' and split 'test'")

    dataset_dir = resolve_local_dataset_dir(dataset_path)
    manifest, error = find_local_manifest(dataset_dir)
    if manifest is None:
        raise ValueError(error or "Local dataset manifest was not found")
    features = read_csv_features(manifest)
    schema_error = validate_schema(features, ["test"])
    if schema_error:
        raise ValueError(schema_error)

    loaded = datasets.load_dataset(
        "csv",
        data_files={"test": str(manifest)},
        split="test",
        streaming=False,
    )

    def resolve_audio_path(sample: dict) -> dict:
        raw_path = sample.get("audio")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Local dataset audio values must be non-empty relative paths")
        relative_audio = Path(raw_path)
        if relative_audio.is_absolute():
            raise ValueError("Local dataset audio paths must be relative")
        resolved = (dataset_dir / relative_audio).resolve()
        try:
            resolved.relative_to(dataset_dir)
        except ValueError as exc:
            raise ValueError(
                "Local dataset audio path escapes the selected dataset directory"
            ) from exc
        return {"audio": str(resolved)}

    return loaded.map(resolve_audio_path, load_from_cache_file=False)
