import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from api import dataset_catalog as catalog


class FakeLoadedDataset:
    def __init__(self, audio_value: str):
        self.audio_value = audio_value
        self.mapped = None

    def map(self, function, **kwargs):
        self.mapped = function({"audio": self.audio_value})
        return self


class DatasetCatalogTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_manifest(self, name: str, header: str, filename: str = "metadata.csv"):
        dataset_dir = self.root / name
        dataset_dir.mkdir()
        (dataset_dir / filename).write_text(header + "\n", encoding="utf-8")
        return dataset_dir

    def local_env(self):
        return mock.patch.dict(
            os.environ,
            {"OPEN_ASR_LOCAL_DATASET_ROOT": str(self.root)},
            clear=False,
        )

    def test_local_catalog_is_sorted_without_reading_manifests(self):
        self.write_manifest("Valid", "audio,text")
        self.write_manifest("Missing audio", "id,text", filename="stt.csv")
        self.write_manifest("Missing transcript", "audio,id")
        (self.root / "No manifest").mkdir()
        self.write_manifest(".hidden", "audio,text")

        with self.local_env(), mock.patch(
            "api.dataset_catalog.read_csv_features"
        ) as read_features:
            payload = catalog.local_dataset_catalog()

        entries = payload["datasets"]
        self.assertEqual(
            [entry["label"] for entry in entries],
            ["Missing audio", "Missing transcript", "No manifest", "Valid"],
        )
        self.assertTrue(all(entry["valid"] is None for entry in entries))
        self.assertTrue(all(entry["validation_status"] == "unchecked" for entry in entries))
        self.assertTrue(all(entry["splits"] == ["test"] for entry in entries))
        read_features.assert_not_called()

    def test_local_selection_is_validated_only_when_requested(self):
        dataset_dir = self.write_manifest("Preferred", "audio,text")
        (dataset_dir / "extra.csv").write_text("id,text\n", encoding="utf-8")

        with self.local_env():
            result = catalog.validate_dataset_selection(
                "local", "Preferred", "default", "test"
            )

        self.assertEqual(result["features"], ["audio", "text"])

        self.write_manifest("Broken", "id,text")
        with self.local_env(), self.assertRaisesRegex(
            catalog.DatasetValidationError, "missing required 'audio' column"
        ):
            catalog.validate_dataset_selection("local", "Broken", "default", "test")

    def test_local_directory_resolution_rejects_traversal(self):
        self.write_manifest("Valid", "audio,text")
        with self.local_env():
            self.assertEqual(catalog.resolve_local_dataset_dir("Valid"), self.root / "Valid")
            for invalid in ("../outside", "/tmp/outside", ".hidden", "nested/path"):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        catalog.resolve_local_dataset_dir(invalid)

    def test_default_huggingface_repositories_include_acefone(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                catalog.huggingface_dataset_repos(),
                [
                    "bettercallaaryan/nc_agent_clips_openasr",
                    "bettercallaaryan/acefone_stt_eval_openasr",
                ],
            )

    def test_huggingface_catalog_lists_configs_without_builder_inspection(self):
        def configs(repo, **_kwargs):
            return ["default", "other"] if repo == "owner/one" else ["default"]

        with mock.patch.dict(
            os.environ,
            {
                "OPEN_ASR_HF_DATASET_REPOS": "owner/one,owner/two",
                "HF_TOKEN": "secret-token",
            },
            clear=False,
        ), mock.patch(
            "datasets.get_dataset_config_names", side_effect=configs
        ) as get_configs, mock.patch(
            "datasets.get_dataset_split_names", return_value=["test", "validation"]
        ), mock.patch("datasets.load_dataset_builder") as load_builder:
            payload = catalog.huggingface_dataset_catalog()

        self.assertEqual(
            [entry["id"] for entry in payload["datasets"]],
            [
                "owner/one::default",
                "owner/one::other",
                "owner/two::default",
            ],
        )
        self.assertTrue(all(entry["valid"] is None for entry in payload["datasets"]))
        self.assertNotIn("secret-token", str(payload))
        self.assertEqual(get_configs.call_count, 2)
        load_builder.assert_not_called()

    def test_huggingface_selection_runs_schema_validation(self):
        builder = types.SimpleNamespace(
            info=types.SimpleNamespace(features={"audio": object(), "text": object()})
        )
        with mock.patch.dict(
            os.environ, {"HF_TOKEN": "secret-token"}, clear=False
        ), mock.patch(
            "datasets.get_dataset_split_names", return_value=["test", "validation"]
        ) as get_splits, mock.patch(
            "datasets.load_dataset_builder", return_value=builder
        ) as load_builder:
            result = catalog.validate_dataset_selection(
                "huggingface", "owner/repo", "default", "test"
            )

        self.assertEqual(result["features"], ["audio", "text"])
        get_splits.assert_called_once_with("owner/repo", "default", token="secret-token")
        load_builder.assert_called_once_with("owner/repo", "default", token="secret-token")

    def test_huggingface_selection_reports_incompatible_schema(self):
        builder = types.SimpleNamespace(
            info=types.SimpleNamespace(features={"text": object()})
        )
        with mock.patch(
            "datasets.get_dataset_split_names", return_value=["test"]
        ), mock.patch("datasets.load_dataset_builder", return_value=builder), self.assertRaisesRegex(
            catalog.DatasetValidationError, "missing required 'audio' column"
        ):
            catalog.validate_dataset_selection(
                "huggingface", "owner/repo", "default", "test"
            )

    def test_local_loader_resolves_relative_audio(self):
        dataset_dir = self.write_manifest("Valid", "audio,text")
        fake = FakeLoadedDataset("audio/sample.wav")

        with self.local_env(), mock.patch(
            "datasets.load_dataset", return_value=fake
        ) as load_dataset:
            loaded = catalog.load_evaluation_dataset(
                "local", "Valid", "default", "test"
            )

        self.assertIs(loaded, fake)
        self.assertEqual(
            fake.mapped,
            {
                "audio": {
                    "path": str(dataset_dir / "audio" / "sample.wav"),
                    "bytes": None,
                }
            },
        )
        load_dataset.assert_called_once_with(
            "csv",
            data_files={"test": str(dataset_dir / "metadata.csv")},
            split="test",
            streaming=False,
        )

    def test_local_loader_rejects_unsafe_audio_paths(self):
        self.write_manifest("Valid", "audio,text")
        with self.local_env(), mock.patch(
            "datasets.load_dataset",
            return_value=FakeLoadedDataset("../outside.wav"),
        ):
            with self.assertRaisesRegex(ValueError, "escapes"):
                catalog.load_evaluation_dataset(
                    "local", "Valid", "default", "test"
                )

        with self.local_env(), mock.patch(
            "datasets.load_dataset",
            return_value=FakeLoadedDataset("/tmp/outside.wav"),
        ):
            with self.assertRaisesRegex(ValueError, "must be relative"):
                catalog.load_evaluation_dataset(
                    "local", "Valid", "default", "test"
                )


if __name__ == "__main__":
    unittest.main()
