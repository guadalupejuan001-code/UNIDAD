"""Catalog integrity and availability without loading model tensors."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ia_propia.conversation.catalog import load_catalog
from ia_propia.conversation.configuration import count_config_parameters


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _fixture(root, name="small-own"):
    directory = root / name
    directory.mkdir()
    config = {"vocab_size": 262, "d_model": 8, "n_layers": 1, "n_heads": 1,
              "context_length": 16, "dropout": 0.0, "ffn_multiplier": 4, "seed": 123}
    # Deliberately not a neural tensor file: catalog inspection cannot load it.
    contents = {"config.json": _canonical(config), "tokenizer.json": b'{"fixture":true}',
                "weights.safetensors": b"metadata inspection only"}
    files = {}
    for filename, data in contents.items():
        (directory / filename).write_bytes(data)
        files[filename] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    identity = {"schemaVersion": 1, "modelType": "unidad-conversation-transformer",
                "files": {filename: descriptor["sha256"] for filename, descriptor in files.items()}}
    model_id = "unidad-conversation-" + hashlib.sha256(_canonical(identity)).hexdigest()
    count = count_config_parameters(config)
    manifest = {"schemaVersion": 1, "modelType": "unidad-conversation-transformer", "modelId": model_id,
                "files": files, "config": config, "training": {"parameterCount": count, "trainedFrom": "random_weights"},
                "evaluation": {"productionAccepted": False}}
    (directory / "manifest.json").write_bytes(_canonical(manifest))
    return {"name": name, "path": name, "modelId": model_id,
            "weightsSha256": files["weights.safetensors"]["sha256"],
            "tokenizerSha256": files["tokenizer.json"]["sha256"], "config": config,
            "parameterCount": count, "artifactKind": "complete", "rootOrigin": "random_weights",
            "productionAccepted": False}


def _write_catalog(root, entries):
    path = root / "catalog.json"
    path.write_bytes(_canonical({"schemaVersion": 1, "entries": entries}))
    return path


class ConversationCatalogTests(unittest.TestCase):
    def test_delivered_catalog_lists_all_distinct_variants_and_primary_order(self):
        root = Path(__file__).resolve().parents[1]
        entries = load_catalog("configs/conversation-models.json", root)
        self.assertEqual(len(entries), 35)
        self.assertEqual([entry["name"] for entry in entries[:4]], ["reference-wiki1", "memory-candidate", "C7", "V7"])
        self.assertEqual(len({entry["modelId"] for entry in entries}), 35)
        self.assertEqual(len({entry["weightsSha256"] for entry in entries}), 35)
        self.assertTrue(all(len(entry["name"]) <= 64 for entry in entries))
        self.assertTrue(all(entry["rootOrigin"] == "random_weights" for entry in entries))
        self.assertFalse(any(entry["productionAccepted"] for entry in entries))
        reconstructed = [entry for entry in entries if entry["artifactKind"] == "complete_reconstructed"]
        self.assertEqual(len(reconstructed), 1)

    def test_catalog_reads_metadata_without_loading_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = _fixture(root)
            result = load_catalog(_write_catalog(root, [entry]), root)
            self.assertTrue(result[0]["available"])
            self.assertTrue(result[0]["loadable"])
            self.assertEqual(result[0]["status"], "ready")
            self.assertFalse(result[0]["weightsVerifiedDuringCatalogLoad"])
            self.assertEqual(result[0]["artifactDirectory"], str(root / entry["path"]))
            self.assertNotIn("available", entry)

    def test_missing_artifact_and_weight_remain_listed_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = _fixture(root)
            (root / entry["path"] / "weights.safetensors").unlink()
            result = load_catalog(_write_catalog(root, [entry]), root)[0]
            self.assertFalse(result["loadable"])
            self.assertEqual(result["missingFiles"], ["weights.safetensors"])
            entry["path"] = "not-installed"
            result = load_catalog(_write_catalog(root, [entry]), root)[0]
            self.assertEqual(result["name"], "small-own")
            self.assertFalse(result["available"])
            self.assertEqual(set(result["missingFiles"]), {"manifest.json", "config.json", "tokenizer.json", "weights.safetensors"})

    def test_delta_source_does_not_become_ready_without_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = _fixture(root)
            entry["artifactKind"] = "delta_source"
            result = load_catalog(_write_catalog(root, [entry]), root)[0]
            self.assertFalse(result["loadable"])
            self.assertEqual(result["status"], "pending_reconstruction")

    def test_corrupt_present_metadata_and_wrong_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = _fixture(root)
            path = _write_catalog(root, [entry])
            (root / entry["path"] / "tokenizer.json").write_bytes(b'{"fixture":null}')
            with self.assertRaisesRegex(ValueError, "byte count|hash"):
                load_catalog(path, root)
            entry = _fixture(root, "second-own")
            altered = deepcopy(entry)
            altered["modelId"] = "unidad-conversation-" + "0" * 64
            with self.assertRaisesRegex(ValueError, "identity"):
                load_catalog(_write_catalog(root, [altered]), root)

    def test_duplicate_members_and_untrusted_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = _fixture(root)
            duplicate = deepcopy(entry)
            duplicate["name"] = "alias"
            with self.assertRaisesRegex(ValueError, "distinct"):
                load_catalog(_write_catalog(root, [entry, duplicate]), root)
            for path in ("../outside", "/outside", "models/../outside", "https://example.test/model"):
                changed = {**entry, "path": path}
                with self.assertRaises(ValueError):
                    load_catalog(_write_catalog(root, [changed]), root)
            (root / "linked").symlink_to(root / entry["path"], target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                load_catalog(_write_catalog(root, [{**entry, "path": "linked"}]), root)

    def test_partition_provenance_and_parameter_claims_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = _fixture(root)
            for altered in ({**entry, "rootOrigin": "external_checkpoint"},
                            {**entry, "parameterCount": entry["parameterCount"] + 1},
                            {**entry, "productionAccepted": True}):
                with self.assertRaises(ValueError):
                    load_catalog(_write_catalog(root, [altered]), root)

    def test_duplicate_json_keys_and_symlink_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = _fixture(root)
            path = root / "catalog.json"
            path.write_text('{"schemaVersion":1,"schemaVersion":1,"entries":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_catalog(path, root)
            weight = root / entry["path"] / "weights.safetensors"
            target = root / "weight-copy"
            target.write_bytes(weight.read_bytes())
            weight.unlink()
            weight.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                load_catalog(_write_catalog(root, [entry]), root)


if __name__ == "__main__":
    unittest.main()
