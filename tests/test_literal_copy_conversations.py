"""Offline literal-copy oracles use independent fixtures, never model replies."""

from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Literal-copy authoring requires optional tokenizers")

from examples import build_literal_copy_conversations as AUTHOR
from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, fit_tokenizer


class LiteralCopyConversationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        # The independent byte-alphabet fixture is not an existing training or
        # evaluation corpus. A larger context accommodates its unmerged bytes.
        cls.tokenizer = fit_tokenizer([{"messages": [
            {"role": "user", "content": "Prueba explícita de espacios y Unicode."},
            {"role": "assistant", "content": "Sólo un fixture técnico declarado."}]}], vocab_size=262)
        cls.tokenizer_path = cls.root / "fixture-tokenizer.json"
        cls.tokenizer_path.write_text(cls.tokenizer.to_str(), encoding="utf-8")
        cls.output, cls.manifest = cls.root / "copy.jsonl", cls.root / "copy.manifest.json"
        cls.options = {"seed": 449, "worlds": 40, "context_length": 4096}
        cls.report = AUTHOR.build(cls.tokenizer_path, cls.output, cls.manifest, **cls.options)
        cls.rows = load_corpus(cls.output)

    def test_manual_oracles_keep_spaces_diacritics_and_corrected_value_after_distractor(self):
        spans = {"a": "  José  ", "b": "café  azul", "c": "Jose\u0301"}
        expected = {"direct-update-recall": [spans["a"], spans["b"], spans["a"]],
                    "labels-update-recall": [spans["a"], spans["b"], spans["b"], spans["c"]]}
        records = []
        for template, answers in expected.items():
            for split in AUTHOR.SPLITS:
                messages = AUTHOR.render(spans, split, template)
                self.assertEqual([message["content"] for message in messages[1::2]], answers)
                self.assertIn("«  José  »", messages[0]["content"])
                self.assertIn("«café  azul»", messages[2]["content"])
                if template == "labels-update-recall":
                    self.assertIn("«Jose\u0301»", messages[4]["content"])
                    self.assertNotIn(spans["c"], messages[6]["content"])
                record = {"id": template + "-" + split, "groupId": "manual-" + template + "-" + split,
                          "split": split, "source": {"dataset": "independent-literal-fixture", "kind": "authored-synthetic"},
                          "messages": messages}
                validate_record(record)
                records.append(record)
        path = self.root / "manual-oracle.jsonl"
        path.write_bytes(b"".join(AUTHOR.canonical(record) + b"\n" for record in records))
        self.assertEqual(load_corpus(path), records)
        for record in load_corpus(path):
            tokens, _ = dialogue_tokens(record["messages"], self.tokenizer)
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            for answer in record["messages"][1::2]:
                self.assertIn(answer["content"], decoded)
        # Isolate supervised assistant tokens: finding a value somewhere in the
        # full dialogue could otherwise pass by finding it in the user prompt.
        for span in (spans["a"], spans["c"]):
            ids, supervised = dialogue_tokens([
                {"role": "user", "content": "Salida literal del fixture técnico."},
                {"role": "assistant", "content": span}], self.tokenizer)
            targets = [token for token, teach in zip(ids, supervised) if teach]
            self.assertEqual(self.tokenizer.decode(targets, skip_special_tokens=True).encode("utf-8"), span.encode("utf-8"))
        self.assertNotEqual(spans["a"].strip().encode(), spans["c"].encode())

    def test_every_target_comes_from_exact_prior_user_span_with_family_disjoint_folds(self):
        groups, prompts, target_folds, group_prefixes = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
        for record in self.rows:
            source = record["source"]
            self.assertEqual(source["dataset"], "literal-copy-v1")
            self.assertFalse(source["modelResponsesUsed"])
            self.assertFalse(source["privateConversationUsed"])
            origin = record["messages"][2 * (source["targetFromUserTurn"] - 1)]["content"]
            original = origin.split("«", 1)[1].split("»", 1)[0]
            actual = record["messages"][-1]["content"]
            self.assertEqual(actual.encode("utf-8"), original.encode("utf-8"))
            self.assertEqual(source["targetSha256"], hashlib.sha256(actual.encode("utf-8")).hexdigest())
            target_ids = self.tokenizer.encode(actual, add_special_tokens=False).ids
            self.assertTrue(1 <= len(target_ids) <= 20)
            self.assertEqual(self.tokenizer.decode(target_ids, skip_special_tokens=False), actual)
            self.assertLessEqual(len(dialogue_tokens(record["messages"], self.tokenizer)[0]), 4097)
            groups[record["groupId"]].add(record["split"])
            prompts[AUTHOR.canonical(record["messages"][:-1])].add(record["split"])
            target_folds[actual].add(record["split"])
            group_prefixes[record["groupId"]].add((source["templateId"], source["prefixPairs"]))
            self.assertNotIn(record["groupId"], "".join(message["content"] for message in record["messages"]))
        self.assertTrue(all(len(folds) == 1 for folds in groups.values()))
        self.assertTrue(all(len(folds) == 1 for folds in prompts.values()))
        self.assertTrue(all(len(folds) == 1 for folds in target_folds.values()))
        required = {("direct-update-recall", i) for i in (1, 2, 3)} | {("labels-update-recall", i) for i in (1, 2, 3, 4)}
        self.assertTrue(all(prefixes == required for prefixes in group_prefixes.values()))
        self.assertEqual(Counter(next(iter(folds)) for folds in groups.values()), {"train": 30, "validation": 5, "test": 5})
        self.assertEqual(self.report["uniqueLiteralTargets"], 120)
        self.assertEqual(self.report["records"], 280)

    def test_publication_is_reproducible_immutable_and_leaves_fixed_tokenizer_unchanged(self):
        before = self.tokenizer_path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, manifest = root / "copy.jsonl", root / "copy.manifest.json"
            report = AUTHOR.build(self.tokenizer_path, output, manifest, **self.options)
            self.assertEqual(report, self.report)
            self.assertEqual(output.read_bytes(), self.output.read_bytes())
            self.assertEqual(manifest.read_bytes(), self.manifest.read_bytes())
            original = output.read_bytes(), manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, "immutable"):
                AUTHOR.build(self.tokenizer_path, output, manifest, **self.options)
            self.assertEqual((output.read_bytes(), manifest.read_bytes()), original)
            sentinel = root / "occupied.json"
            sentinel.write_bytes(b"Existing evidence")
            missing = root / "not-created.jsonl"
            with self.assertRaisesRegex(ValueError, "immutable"):
                AUTHOR.build(self.tokenizer_path, missing, sentinel, **self.options)
            self.assertFalse(missing.exists())
            self.assertEqual(sentinel.read_bytes(), b"Existing evidence")
            with self.assertRaisesRegex(ValueError, "distinct"):
                AUTHOR.build(self.tokenizer_path, self.tokenizer_path, root / "unused.json", **self.options)
        self.assertEqual(self.tokenizer_path.read_bytes(), before)
        self.assertEqual(self.report["tokenizer"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertFalse(self.report["tokenizer"]["refitted"])


if __name__ == "__main__":
    unittest.main()
