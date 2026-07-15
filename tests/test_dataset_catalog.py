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

    def test_local_catalog_is_sorted_and_reports_schema_errors(self):
        self.write_manifest("Valid", "audio,text")
        self.write_manifest("Missing audio", "id,text", filename="stt.csv")
        self.write_manifest("Missing transcript", "audio,id")
        (self.root / "No manifest").mkdir()
        ambiguous = self.root / "Ambiguous"
        ambiguous.mkdir()
        (ambiguous / "one.csv").write_text("audio,text\n", encoding="utf-8")
        (ambiguous / "two.csv").write_text("audio,text\n", encoding="utf-8")
        self.write_manifest(".hidden", "audio,text")

        with self.local_env():
            payload = catalog.local_dataset_catalog()

        entries = payload["datasets"]
        self.assertEqual(
            [entry["label"] for entry in entries],
            ["Ambiguous", "Missing audio", "Missing transcript", "No manifest", "Valid"],
        )
        by_name = {entry["label"]: entry for entry in entries}
        self.assertTrue(by_name["Valid"]["valid"])
        self.assertEqual(by_name["Valid"]["splits"], ["test"])
        self.assertIn("missing required 'audio' column", by_name["Missing audio"]["error"])
        self.assertIn("missing a transcript column", by_name["Missing transcript"]["error"])
        self.assertIn("no CSV manifest", by_name["No manifest"]["error"])
        self.assertIn("multiple CSV manifests", by_name["Ambiguous"]["error"])

    def test_metadata_csv_is_preferred_over_other_csv_files(self):
        dataset_dir = self.write_manifest("Preferred", "audio,text")
        (dataset_dir / "extra.csv").write_text("id,text\n", encoding="utf-8")

        with self.local_env():
            entry = catalog.local_dataset_catalog()["datasets"][0]

        self.assertTrue(entry["valid"])
        self.assertEqual(entry["features"], ["audio", "text"])

    def test_local_directory_resolution_rejects_traversal(self):
        self.write_manifest("Valid", "audio,text")
        with self.local_env():
            self.assertEqual(catalog.resolve_local_dataset_dir("Valid"), self.root / "Valid")
            for invalid in ("../outside", "/tmp/outside", ".hidden", "nested/path"):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        catalog.resolve_local_dataset_dir(invalid)

    def test_huggingface_configs_are_validated_independently(self):
        features = {
            "valid": {"audio": object(), "text": object()},
            "missing-audio": {"text": object()},
            "no-splits": {"audio": object(), "transcript": object()},
        }

        def splits(_repo, config, **_kwargs):
            if config == "broken":
                raise RuntimeError("private config unavailable")
            return [] if config == "no-splits" else ["test", "validation"]

        def builder(_repo, config, **_kwargs):
            return types.SimpleNamespace(
                info=types.SimpleNamespace(features=features[config])
            )

        with mock.patch.dict(
            os.environ,
            {
                "OPEN_ASR_HF_DATASET_REPO": "owner/repo",
                "HF_TOKEN": "secret-token",
            },
            clear=False,
        ), mock.patch(
            "datasets.get_dataset_config_names",
            return_value=["valid", "missing-audio", "no-splits", "broken"],
        ) as get_configs, mock.patch(
            "datasets.get_dataset_split_names", side_effect=splits
        ), mock.patch(
            "datasets.load_dataset_builder", side_effect=builder
        ):
            payload = catalog.huggingface_dataset_catalog()

        by_name = {entry["id"]: entry for entry in payload["datasets"]}
        self.assertTrue(by_name["valid"]["valid"])
        self.assertFalse(by_name["missing-audio"]["valid"])
        self.assertIn("no usable splits", by_name["no-splits"]["error"])
        self.assertIn("inspection failed", by_name["broken"]["error"])
        self.assertNotIn("secret-token", str(payload))
        get_configs.assert_called_once_with("owner/repo", token="secret-token")

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
            {"audio": str(dataset_dir / "audio" / "sample.wav")},
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
