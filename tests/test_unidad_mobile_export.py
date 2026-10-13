"""ONNX packages must preserve native weights, dynamic contexts and replies."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

if any(importlib.util.find_spec(name) is None for name in
       ("torch", "tokenizers", "safetensors", "onnx", "onnxruntime")):
    raise unittest.SkipTest("Optional conversation/mobile dependencies are unavailable")

import torch

from ia_propia.conversation.artifacts import save_artifact
from ia_propia.conversation.model import ConversationModel
from ia_propia.conversation.runtime import ConversationEngine
from ia_propia.conversation.tokenizer import fit_tokenizer
from scripts import export_unidad_mobile as mobile


class MobileExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tokenizer = fit_tokenizer([{"messages": [
            {"role": "user", "content": "Hola. ¿Cómo estás?"},
            {"role": "assistant", "content": "Estoy bien."}]}], vocab_size=262)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def artifact(self, root, *, copy=False, context=512):
        config = {"vocab_size": self.tokenizer.get_vocab_size(), "d_model": 16,
                  "n_layers": 1, "n_heads": 2, "context_length": context,
                  "ffn_multiplier": 2, "dropout": 0.0, "seed": 17}
        if copy:
            config["copy_head"] = {"version": 1, "rank": 32, "seed": 23}
        source = root / "source"
        save_artifact(source, ConversationModel(config), self.tokenizer,
                      training={"trainedFrom": "random_weights"})
        return source

    def export(self, source, output, evidence):
        args = ["export_unidad_mobile.py", "--model", str(source),
                "--output", str(output), "--evidence", str(evidence)]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            mobile.main()

    def test_minimum_vocabulary_packages_preserve_both_native_heads(self):
        for copy in (False, True):
            with self.subTest(copy=copy), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.artifact(root, copy=copy)
                before = {p.name: p.read_bytes() for p in source.iterdir()}
                output, evidence = root / "model.unidad", root / "parity.json"
                self.export(source, output, evidence)
                receipt = json.loads(evidence.read_text())
                engine = ConversationEngine(source)
                self.assertEqual(receipt["modelId"], engine.model_id)
                self.assertEqual(receipt["logitCases"], 7)
                self.assertEqual(receipt["greedyTurnsIdentical"], 4)
                self.assertFalse(receipt["qualityImprovementClaimed"])
                self.assertFalse(receipt["androidDeviceTested"])
                with zipfile.ZipFile(output) as archive:
                    self.assertEqual(set(archive.namelist()),
                                     {"manifest.json", "model.onnx", "config.json", "tokenizer.json"})
                    self.assertIsNone(archive.testzip())
                    manifest = json.loads(archive.read("manifest.json"))
                    self.assertEqual(manifest["modelId"], engine.model_id)
                    self.assertFalse(manifest["productionAccepted"])
                    for name in ("config.json", "tokenizer.json"):
                        self.assertEqual(archive.read(name), before[name])
                    archive.extract("model.onnx", root / "restored")
                options = mobile.ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                session = mobile.ort.InferenceSession(str(root / "restored/model.onnx"), options,
                                                     providers=["CPUExecutionProvider"])
                # Reopen the packaged graph and compare a context absent from
                # the exporter's own seven parity cases, including EOS resets.
                ids = [4, 11, 11, 12, 2, 3, 13, 2, 4, 14, 15, 2, 5]
                values = torch.tensor([ids])
                selected = torch.zeros_like(values, dtype=torch.bool)
                selected[:, -1] = True
                with torch.inference_mode():
                    expected = engine._model.selected_logits(values, selected)[0].numpy()
                mobile.np.testing.assert_allclose(mobile.infer(session, ids), expected,
                                                 rtol=3e-4, atol=3e-4)
                self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_existing_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, evidence = root / "model.unidad", root / "parity.json"
            output.write_bytes(b"existing artifact")
            with self.assertRaises(SystemExit) as caught:
                self.export(root / "missing-source", output, evidence)
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.read_bytes(), b"existing artifact")
            self.assertFalse(evidence.exists())

    def test_unsupported_context_does_not_publish_a_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.artifact(root, context=128)
            output, evidence = root / "model.unidad", root / "parity.json"
            with self.assertRaises(SystemExit) as caught:
                self.export(source, output, evidence)
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(evidence.exists())

    def test_source_replacement_blocks_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.artifact(root)
            output, evidence = root / "model.unidad", root / "parity.json"
            native_export = torch.onnx.export

            def replace_source(*args, **kwargs):
                native_export(*args, **kwargs)
                config = source / "config.json"
                config.write_bytes(config.read_bytes() + b"\n")

            with patch.object(torch.onnx, "export", side_effect=replace_source), \
                    self.assertRaisesRegex(ValueError, "Source artifact changed"):
                self.export(source, output, evidence)
            self.assertFalse(output.exists())
            self.assertFalse(evidence.exists())


if __name__ == "__main__":
    unittest.main()
