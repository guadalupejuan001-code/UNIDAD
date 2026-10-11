import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation.data import CorpusError, load_corpus


_HELPER = Path(__file__).resolve().parents[1] / "examples" / "prepare_expanded_conversations.py"
_SPEC = importlib.util.spec_from_file_location("expanded_conversation_curriculum", _HELPER)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
prepare = _MODULE.prepare


def record(identifier, split="train", group=None, text=None, kind="public-human-reviewed"):
    return {"id": identifier, "groupId": group or f"family-{identifier}", "split": split,
            "source": {"dataset": "fictional curriculum fixture", "kind": kind},
            "messages": [{"role": "user", "content": f"Pregunta {text or identifier}."},
                         {"role": "assistant", "content": f"Respuesta {text or identifier}."}]}


class ConversationCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.basic = [record("basic-train", "train"), record("basic-val", "validation"),
                      record("basic-test", "test")]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, records):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                        encoding="utf-8")
        return path

    def inputs(self, public, basic=None):
        return self.write("basic.jsonl", basic or self.basic), self.write("oasst.jsonl", public)

    def test_preserves_every_basic_record_and_original_family_folds(self):
        base = self.basic + [record("basic-train-copy", "train", text="basic-train")]
        new = record("new-continuation", "validation", group="family-basic-val")
        public = [copy.deepcopy(self.basic[0]), new, record("new-test", "test")]
        basic_path, public_path = self.inputs(public, base)
        before = (basic_path.read_bytes(), public_path.read_bytes())
        output = self.root / "expanded.jsonl"
        report = prepare(basic_path, public_path, output)
        rows = load_corpus(output)
        self.assertEqual(rows[:len(base)], base)
        self.assertEqual(rows[len(base):], public[1:])
        self.assertEqual((basic_path.read_bytes(), public_path.read_bytes()), before)
        self.assertEqual(report["preservedBasicRecords"], 4)
        self.assertEqual(report["appendedRecords"], 2)
        self.assertEqual(report["groups"], 5)
        self.assertEqual(report["splits"], {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(report["groupSplits"], {"train": 2, "validation": 1, "test": 2})
        self.assertEqual(report["duplicateIdsSkipped"], 1)
        self.assertEqual(report["rawStats"]["combined"]["duplicateIds"], 1)
        self.assertEqual(report["rawStats"]["basic"]["exactDuplicateMessages"], 1)

    def test_new_train_copies_of_basic_holdouts_are_omitted(self):
        public = []
        for heldout in self.basic[1:]:
            new = record(f"copied-{heldout['id']}")
            new["messages"] = copy.deepcopy(heldout["messages"])
            public.append(new)
        public.append(record("novel-train"))
        basic_path, public_path = self.inputs(public)
        output = self.root / "expanded.jsonl"
        report = prepare(basic_path, public_path, output)
        rows = load_corpus(output)
        self.assertEqual([row["id"] for row in rows],
                         [row["id"] for row in self.basic] + ["novel-train"])
        holdout_messages = [row["messages"] for row in self.basic[1:]]
        self.assertFalse(any(row["messages"] in holdout_messages for row in rows if row["split"] == "train"))
        self.assertEqual(report["exactDuplicatesRemoved"], 2)
        self.assertEqual(report["exactDuplicatesAgainstBasicRemoved"], 2)

    def test_exact_new_duplicates_keep_first_without_score_selection(self):
        first = record("new-first")
        first["source"]["qualityScore"] = 0.01
        second = record("new-second", "test")
        second["messages"] = copy.deepcopy(first["messages"])
        second["source"]["qualityScore"] = 0.99
        distinct = record("new-distinct", "validation")
        basic_path, public_path = self.inputs([first, second, distinct])
        output = self.root / "expanded.jsonl"
        report = prepare(basic_path, public_path, output)
        self.assertEqual(load_corpus(output)[len(self.basic):], [first, distinct])
        self.assertEqual(report["exactDuplicatesRemoved"], 1)
        self.assertEqual(report["exactDuplicatesAgainstBasicRemoved"], 0)

    def test_cross_corpus_family_conflict_rejected_before_message_dedup(self):
        conflicting = copy.deepcopy(self.basic[2])
        conflicting["id"] = "new-copy"
        conflicting["split"] = "train"
        basic_path, public_path = self.inputs([conflicting])
        output = self.root / "expanded.jsonl"
        with self.assertRaises(CorpusError):
            prepare(basic_path, public_path, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".expanded-conversations-*")), [])

    def test_shared_id_conflict_rejected_even_when_messages_match(self):
        for field, value in (("groupId", "another-family"),
                             ("source", {"dataset": "different revision", "kind": "public-human-reviewed"})):
            with self.subTest(field=field):
                conflicting = copy.deepcopy(self.basic[0])
                conflicting[field] = value
                basic_path, public_path = self.inputs([conflicting])
                with self.assertRaises(CorpusError):
                    prepare(basic_path, public_path, self.root / "expanded.jsonl")
                self.assertFalse((self.root / "expanded.jsonl").exists())

    def test_invalid_source_duplicate_ids_and_families_are_rejected(self):
        for public in ([record("same"), record("same")],
                       [record("a", "train", group="same-family"),
                        record("b", "test", group="same-family")]):
            with self.subTest(public=public):
                basic_path, public_path = self.inputs(public)
                with self.assertRaises(CorpusError):
                    prepare(basic_path, public_path, self.root / "expanded.jsonl")
                self.assertFalse((self.root / "expanded.jsonl").exists())

    def test_existing_output_and_input_aliases_are_never_modified(self):
        basic_path, public_path = self.inputs([record("new")])
        existing = self.root / "existing.jsonl"
        existing.write_bytes(b"existing evidence\n")
        originals = {path: path.read_bytes() for path in (existing, basic_path, public_path)}
        for output in (existing, basic_path, public_path):
            with self.subTest(output=output), self.assertRaises(ValueError):
                prepare(basic_path, public_path, output)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_output_hash_provenance_and_repeat_are_reproducible(self):
        public = [record("new", kind="authored-synthetic")]
        basic_path, public_path = self.inputs(public)
        outputs = [self.root / "one.jsonl", self.root / "two.jsonl"]
        reports = [prepare(basic_path, public_path, path) for path in outputs]
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0]["sha256"], hashlib.sha256(outputs[0].read_bytes()).hexdigest())
        self.assertEqual(reports[0]["inputSha256"],
                         {"basic": hashlib.sha256(basic_path.read_bytes()).hexdigest(),
                          "oasst": hashlib.sha256(public_path.read_bytes()).hexdigest()})
        self.assertEqual(reports[0]["sources"], {"authored-synthetic": 1, "public-human-reviewed": 3})
        self.assertEqual(outputs[0].stat().st_mode & 0o777, 0o600)
        original = outputs[0].read_bytes()
        with self.assertRaises(ValueError):
            prepare(basic_path, public_path, outputs[0])
        self.assertEqual(outputs[0].read_bytes(), original)

    def test_concurrent_output_creation_is_not_overwritten(self):
        basic_path, public_path = self.inputs([record("new")])
        output = self.root / "expanded.jsonl"
        original_link = _MODULE.os.link

        def another_writer_wins(source, destination):
            Path(destination).write_bytes(b"evidence from another writer\n")
            return original_link(source, destination)

        with patch.object(_MODULE.os, "link", side_effect=another_writer_wins):
            with self.assertRaises(ValueError):
                prepare(basic_path, public_path, output)
        self.assertEqual(output.read_bytes(), b"evidence from another writer\n")
        self.assertEqual(list(self.root.glob(".expanded-conversations-*")), [])

    def test_missing_fold_does_not_publish_partial_output(self):
        basic_path, public_path = self.inputs([record("new")], [record("train-only")])
        output = self.root / "expanded.jsonl"
        with self.assertRaises(CorpusError):
            prepare(basic_path, public_path, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
