import json
from pathlib import Path
import tempfile
import unittest

from ia_propia.conversation.data import CorpusError
from ia_propia.conversation.language_data import load_text_corpus, validate_text_record
from ia_propia.conversation import training


def text_record(identifier="document-1", split="train", text="Un documento ficticio para comprobar el contrato."):
    return {"id": identifier, "groupId": identifier, "split": split,
            "source": {"dataset": "authored-test", "kind": "authored-synthetic"}, "text": text}


class LanguageCorpusTests(unittest.TestCase):
    def load(self, records):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"text.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
            return load_text_corpus(path)

    def test_family_chunks_preserve_exact_utf8_and_provenance(self):
        rows = [text_record(text="Primera línea.\n  Ñandú y café.\n"),
                text_record("document-2", text="Segundo fragmento de la misma familia.")]
        rows[1]["groupId"] = rows[0]["groupId"]
        self.assertEqual(self.load(rows), rows)

    def test_reject_ids_family_leakage_and_exact_clone_in_any_fold(self):
        base = text_record()
        variants = [dict(text_record("other", text="Otro texto."), id=base["id"]),
                    dict(text_record("other", "validation", "Otro texto."), groupId=base["groupId"]),
                    text_record("other", "validation", base["text"]),
                    text_record("other", "train", base["text"])]
        for other in variants:
            with self.subTest(other=other), self.assertRaises(CorpusError): self.load([base, other])

    def test_limits_controls_and_separate_text_schema(self):
        invalid = [dict(text_record(), text=""), dict(text_record(), text=True),
                   dict(text_record(), text="á"*32769), dict(text_record(), text="\ud800"),
                   dict(text_record(), text="Antes <user> después"), dict(text_record(), split="development"),
                   dict(text_record(), source={"dataset": "x"}), dict(text_record(), id=True),
                   dict(text_record(), messages=[{"role":"user","content":"falso diálogo"}])]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(CorpusError): validate_text_record(row)
        validate_text_record(text_record(text="á"*32768))

    def test_strict_json_duplicate_keys_and_nonfinite_metadata(self):
        for raw in ['{"id":"one","id":"two"}',
                    json.dumps(text_record()).replace('"authored-synthetic"', 'NaN')]:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"bad.jsonl"; path.write_text(raw+"\n")
                with self.assertRaises(CorpusError): load_text_corpus(path)

    def test_optin_keeps_legacy_configuration_and_call_arity(self):
        self.assertNotIn("objective", training._configs(None, None)[1])
        self.assertEqual(training._loss_metadata({}), {})
        from unittest.mock import patch
        with patch.object(training, "_encoded_data", return_value=({"sentinel":1},None,None,None)) as encode:
            self.assertEqual(training._encoded([], "tokenizer", 8), {"sentinel":1})
            encode.assert_called_once_with([], "tokenizer", 8, "contiguous_nonoverlapping")
        for objective in [None, True, {}, [], "full", "causal-lm"]:
            with self.subTest(objective=objective), self.assertRaises(ValueError):
                training._configs(None, {"objective":objective})
        for objective in ["assistant_last_reply", "causal_lm"]:
            config = training._configs(None, {"objective":objective})[1]
            self.assertEqual(config["objective"], objective)
        self.assertEqual(training._loss_metadata({"objective":"causal_lm"}),
                         {"trainingObjective":"causal_lm", "assistantTargets":"all_text_tokens", "corpusFormat":"text"})


if __name__ == "__main__": unittest.main()
