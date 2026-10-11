import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ia_propia.conversation.data import CorpusError, assign_split, import_oasst, load_corpus


def message(identifier, role, text="Una respuesta en español.", **extra):
    return {"message_id": identifier, "role": role, "text": text, "lang": "es", "review_result": True, "deleted": False, "synthetic": False, "replies": [], **extra}


def tree(identifier="root"):
    root = message(identifier, "prompter", "Quiero organizar mis ideas.")
    first = message(identifier + "-a", "assistant", rank=1)
    follow = message(identifier + "-u", "prompter", "¿Qué conecta esas dos ideas?")
    last = message(identifier + "-b", "assistant", "Comparten el objetivo de avanzar sin perder contexto.", rank=2)
    follow["replies"] = [last]
    first["replies"] = [follow]
    root["replies"] = [first, message(identifier + "-alternative", "assistant", "Podemos empezar por una lista breve.", rank=3)]
    return {"message_tree_id": identifier, "tree_state": "ready_for_export", "prompt": root}


def canonical(identifier="one", group="family", split="train"):
    return {"id": identifier, "groupId": group, "split": split, "source": {"dataset": "fictional test fixture", "kind": "authored-synthetic"}, "messages": [{"role": "user", "content": "¿Qué hacemos?"}, {"role": "assistant", "content": "Primero aclaremos el objetivo."}]}


class ConversationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, objects, compressed=False):
        path = self.root / name
        data = "".join(json.dumps(obj, ensure_ascii=False) + "\n" for obj in objects).encode("utf-8")
        path.write_bytes(gzip.compress(data) if compressed else data)
        return path

    def test_import_preserves_every_accepted_assistant_and_groups_branches(self):
        source = self.write("trees.jsonl.gz", [tree()], compressed=True)
        target = self.root / "corpus.jsonl"
        report = import_oasst(source, target, revision="fictional-revision")
        records = load_corpus(target)
        self.assertEqual(len(records), 3)
        self.assertEqual({row["groupId"] for row in records}, {"oasst:root"})
        self.assertEqual({row["split"] for row in records}, {assign_split("oasst:root")})
        self.assertEqual(sorted(len(row["messages"]) for row in records), [2, 2, 4])
        self.assertEqual(report["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(report["sourceSha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(records[1]["source"]["messageIds"], ["root", "root-a", "root-u", "root-b"])
        self.assertNotIn("rank", records[0]["messages"][1])
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_filters_rejected_ancestors_without_splicing_history(self):
        item = tree()
        first = item["prompt"]["replies"][0]
        first["review_result"] = False
        item["prompt"]["replies"].extend([
            message("deleted", "assistant", deleted=True),
            message("generated", "assistant", synthetic=True, model_name="external-model"),
            message("pii", "assistant", labels={"pii": {"value": 1.0, "count": 2}}),
            message("low", "assistant", labels=[{"name": "quality", "value": 0.1, "count": 2}]),
            message("english", "assistant", lang="en"),
        ])
        report = import_oasst(self.write("trees.jsonl", [item]), self.root / "out.jsonl")
        records = load_corpus(self.root / "out.jsonl")
        self.assertEqual([row["id"] for row in records], ["oasst:root-alternative"])
        for reason in ("not_positively_reviewed", "deleted_or_unspecified", "external_synthetic", "label_pii", "low_quality", "other_language"):
            self.assertEqual(report["filters"][reason], 1)
        self.assertEqual(report["filters"]["ancestor_filtered"], 2)

    def test_tree_order_does_not_change_split_and_limit_keeps_family(self):
        source = self.write("trees.jsonl", [tree("first"), tree("second")])
        report = import_oasst(source, self.root / "out.jsonl", max_records=1)
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["groups"], 1)
        self.assertTrue(report["maxRecordsIsSoftFamilyLimit"])
        reversed_source = self.write("reversed.jsonl", [tree("second"), tree("first")])
        import_oasst(reversed_source, self.root / "reversed-out.jsonl")
        for record in load_corpus(self.root / "reversed-out.jsonl"):
            self.assertEqual(record["split"], assign_split(record["groupId"]))

    def test_official_flat_format_reconstructs_history_even_out_of_order(self):
        root = message("root", "prompter", parent_id=None, message_tree_id="root", tree_state="ready_for_export")
        reply = message("reply", "assistant", parent_id="root", message_tree_id="root", tree_state="ready_for_export")
        for item in (root, reply):
            del item["replies"]
        report = import_oasst(self.write("flat.jsonl.gz", [reply, root], compressed=True), self.root / "out.jsonl")
        self.assertEqual(report["records"], 1)
        self.assertEqual(load_corpus(self.root / "out.jsonl")[0]["source"]["messageIds"], ["root", "reply"])

    def test_bad_flat_parent_and_rootless_cycle_rejected(self):
        for rows in (
            [message("a", "assistant", parent_id="missing", message_tree_id="tree")],
            [message("a", "assistant", parent_id="b", message_tree_id="tree"), message("b", "prompter", parent_id="a", message_tree_id="tree")],
        ):
            with self.subTest(rows=rows), self.assertRaises(CorpusError):
                import_oasst(self.write("flat.jsonl", rows), self.root / "out.jsonl")

    def test_family_leakage_duplicate_ids_and_invalid_roles_rejected(self):
        for records in (
            [canonical(), canonical("two", split="test")],
            [canonical(), canonical(group="another")],
            [{**canonical(), "messages": [{"role": "assistant", "content": "No user."}, {"role": "assistant", "content": "Again."}]}],
            [{**canonical(), "privatePath": "/fictional/private"}],
            [{**canonical(), "messages": [{"role": "user", "content": "Pregunta"}, {"role": "assistant", "content": " "}]}],
        ):
            with self.subTest(records=records), self.assertRaises(CorpusError):
                load_corpus(self.write("bad.jsonl", records))

    def test_strict_utf8_duplicate_json_nonfinite_and_bounds(self):
        invalid = [b'{"id":"a","id":"b"}\n', b'{"value":NaN}\n', b'{"value":1e999}\n', b'{"value":"\xff"}\n', b'{"value":"\\ud800"}\n', b'\n', b'[' * 1000 + b'0' + b']' * 1000 + b'\n']
        for raw in invalid:
            path = self.root / "bad.jsonl"
            path.write_bytes(raw)
            with self.subTest(raw=raw[:30]), self.assertRaises(CorpusError):
                load_corpus(path)
        row = canonical()
        row["messages"][1]["content"] = "x" * 65_537
        with self.assertRaises(CorpusError):
            load_corpus(self.write("long.jsonl", [row]))

    def test_optional_system_and_provenance_preserved(self):
        row = canonical()
        row["messages"].insert(0, {"role": "system", "content": "Habla claro y conecta hechos explícitos."})
        row["source"]["note"] = "Authored, no private conversation."
        self.assertEqual(load_corpus(self.write("ok.jsonl", [row])), [row])

    def test_failed_import_is_atomic_and_cleans_temporary_file(self):
        target = self.root / "out.jsonl"
        target.write_text("existing content", encoding="utf-8")
        item = tree()
        item["prompt"]["lang"] = "en"
        with self.assertRaises(CorpusError):
            import_oasst(self.write("empty.jsonl", [item]), target)
        self.assertEqual(target.read_text(), "existing content")
        self.assertEqual(list(self.root.glob(".conversation-*")), [])

    def test_demo_has_disjoint_families_authored_provenance_and_multiturn(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "conversation-demo.jsonl"
        records = load_corpus(path)
        self.assertGreaterEqual(len(records), 100)
        self.assertLessEqual(len(records), 150)
        self.assertEqual({row["split"] for row in records}, {"train", "validation", "test"})
        self.assertTrue(all(len(row["messages"]) >= 4 for row in records))
        self.assertTrue(all(row["source"]["kind"] == "authored-synthetic" for row in records))
        self.assertTrue(all(row["split"] == assign_split(row["groupId"]) for row in records))
        counts = {}
        for row in records:
            counts[row["groupId"]] = counts.get(row["groupId"], 0) + 1
        self.assertTrue(all(count == 2 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
