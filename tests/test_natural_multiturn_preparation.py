"""Authored prefixes retain family folds; length/collision guards remove families."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ia_propia.conversation.data import assign_split, load_corpus


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "natural_multiturn_preparation", ROOT / "examples/prepare_natural_multiturn.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


def fixture_row(label, split):
    index = 0
    while assign_split(group := f"multiturn-test-{label}-{index}") != split:
        index += 1
    return {"id": f"authored-{label}", "groupId": group, "split": split,
        "source": {"dataset": "unit-test-fiction", "kind": "authored-synthetic",
                   "fictional": True, "privateConversationUsed": False, "modelResponsesUsed": False},
        "messages": [
            {"role": "user", "content": f"En la ficción hay un objeto {label}. ¿Cuál es?"},
            {"role": "assistant", "content": f"El objeto ficticio es {label}."},
            {"role": "user", "content": "Aclara que esto es un ejemplo inventado."},
            {"role": "assistant", "content": "Es un ejemplo inventado para una prueba."},
            {"role": "user", "content": f"Ahora cambio el objeto por otro {label}."},
            {"role": "assistant", "content": f"Ahora el objeto es otro {label}."},
            {"role": "user", "content": "Repite el objeto actual de esta ficción."},
            {"role": "assistant", "content": f"El objeto actual es otro {label}."},
        ]}


class PrefixGuardTests(unittest.TestCase):
    def test_all_four_prefixes_keep_users_golds_family_and_fold(self):
        row = fixture_row("prueba", "train")
        original = copy.deepcopy(row)
        prefixes = builder.expand_authored([row])
        self.assertEqual(row, original)
        self.assertEqual(len(prefixes), 4)
        for prefix, record in enumerate(prefixes, 1):
            self.assertEqual(record["messages"], row["messages"][:2 * prefix])
            self.assertEqual(record["groupId"], row["groupId"])
            self.assertEqual(record["split"], row["split"])
            self.assertEqual(record["source"]["prefixPairs"], prefix)
            self.assertEqual(record["source"]["parentRecordId"], row["id"])

    def test_crossfold_prefix_collision_excludes_both_complete_families(self):
        a, b = fixture_row("uno", "train"), fixture_row("dos", "validation")
        b["messages"][:2] = copy.deepcopy(a["messages"][:2])
        prefixes = builder.expand_authored([a, b])
        excluded = builder.crossfold_families(prefixes)
        self.assertEqual(excluded, {a["groupId"], b["groupId"]})
        self.assertFalse([row for row in prefixes if row["groupId"] not in excluded])

    def test_changed_fold_and_id_collision_are_rejected(self):
        a = fixture_row("uno", "train")
        wrong = {**a, "split": "validation"}
        with self.assertRaisesRegex(ValueError, "stable SHA fold"):
            builder.expand_authored([wrong])
        with self.assertRaisesRegex(ValueError, "IDs collide"):
            builder.check_union([a, copy.deepcopy(a)])

    def test_human_selection_keeps_whole_families_and_rejects_test(self):
        records = []
        for index, split in enumerate(("train", "train", "validation")):
            for prefix in range(1, 4):
                records.append({"id": f"human-{index}-{prefix}", "groupId": f"human-group-{index}",
                    "split": split, "source": {"kind": "public-human-reviewed"}, "messages": []})
        selected = builder.human_anchors(records, train_limit=4, validation_limit=4)
        self.assertEqual(len([row for row in selected if row["split"] == "train"]), 3)
        self.assertEqual(len([row for row in selected if row["split"] == "validation"]), 3)
        groups = {row["groupId"] for row in selected}
        self.assertEqual(selected, [row for row in records if row["groupId"] in groups])
        self.assertEqual(selected, builder.human_anchors(records, 4, 4))
        with self.assertRaisesRegex(ValueError, "not TEST"):
            builder.human_anchors([{**records[0], "split": "test"}])


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers is an optional conversation dependency")
class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer = ROOT / "models/conversation-language-v7/tokenizer.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        return path

    def test_native_train_validation_and_test_are_separate_and_sources_frozen(self):
        from ia_propia.conversation import training
        authored = [fixture_row("uno", "train"), fixture_row("dos", "validation"),
                    fixture_row("tres", "test")]
        human = fixture_row("humano", "train")
        human["messages"] = human["messages"][:2]
        human["source"] = {"dataset": "public-human-fixture", "kind": "public-human-reviewed",
                           "license": "Apache-2.0"}
        source = self.write("authored.jsonl", authored)
        human_source = self.write("human.jsonl", [human])
        report = builder.prepare([source], self.tokenizer, self.root / "prepared",
                                 human_corpus=human_source,
                                 expected_human_corpus_sha256=builder.sha(human_source))
        train_path = self.root / "prepared/train-validation.jsonl"
        trainval = load_corpus(train_path)
        test = load_corpus(self.root / "prepared/authored-test.jsonl")
        self.assertEqual(len(trainval), 9)
        self.assertEqual(len(test), 4)
        self.assertFalse(any(row["split"] == "test" for row in trainval))
        self.assertTrue(all(row["split"] == "test" for row in test))
        self.assertEqual(next(row for row in trainval if row["id"] == human["id"]), human)
        self.assertFalse(report["modelsLoaded"])
        self.assertFalse(report["testEvaluated"])
        self.assertFalse(report["tokenizerRefitted"])
        self.assertEqual(report["inputsSha256"][str(source)], builder.sha(source))
        for name, identity in report["files"].items():
            self.assertEqual(builder.sha(self.root / "prepared" / name), identity["sha256"])
        # Exercise the real native partition guard, not a mirrored assertion.
        records = training._load_training_corpus(train_path, {"partition_policy": "train_validation_only"})
        self.assertEqual(records, trainval)
        with self.assertRaises(FileExistsError):
            builder.prepare([source], self.tokenizer, self.root / "prepared")

    def test_overlong_late_reply_removes_whole_family_in_every_fold(self):
        rows = [fixture_row("corto-train", "train"), fixture_row("corto-val", "validation"),
                fixture_row("corto-test", "test")]
        long_rows = [fixture_row(f"largo-{split}", split) for split in ("train", "validation", "test")]
        for row in long_rows:
            row["messages"][-1]["content"] = "Texto ficticio extenso. " * 100
        source = self.write("families.jsonl", rows + long_rows)
        report = builder.prepare([source], self.tokenizer, self.root / "prepared")
        groups = set(report["excludedLengthFamilies"])
        self.assertEqual(groups, {row["groupId"] for row in long_rows})
        retained = load_corpus(self.root / "prepared/train-validation.jsonl") + load_corpus(
            self.root / "prepared/authored-test.jsonl")
        self.assertEqual(len(retained), 12)
        self.assertFalse(any(row["groupId"] in groups for row in retained))


if __name__ == "__main__":
    unittest.main()
