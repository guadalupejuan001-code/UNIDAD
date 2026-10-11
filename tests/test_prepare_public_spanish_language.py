"""Small stdlib ingestion guards; no downloads, tokenizer, or model imports."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


_PATH = Path(__file__).resolve().parents[1] / "examples" / "prepare_public_spanish_language.py"
_SPEC = importlib.util.spec_from_file_location("prepare_public_spanish_language", _PATH)
helper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helper)


def dialogue(record_id, group, split, user, reply, *, authored=False):
    return {"id": record_id, "groupId": group, "split": split,
            "source": {"dataset": "fictional-fixture" if authored else "OpenAssistant/oasst1",
                       "kind": "authored-synthetic" if authored else "public-human-reviewed"},
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]}


def text(record_id, group, split, content):
    return {"id": record_id, "groupId": group, "split": split,
            "source": {"dataset": "wikimedia/wikipedia", "kind": "public-encyclopedic"},
            "text": content}


class PublicSpanishIngestionTests(unittest.TestCase):
    def test_prompt_clones_remove_entire_families_without_reassigning_folds(self):
        records = [dialogue("a1", "tree-a", "train", "Una consulta duplicada.", "Respuesta primera."),
                   dialogue("a2", "tree-a", "train", "Una continuación exclusiva.", "Otra respuesta."),
                   dialogue("b1", "tree-b", "validation", "Una consulta  duplicada.", "Respuesta distinta."),
                   dialogue("b2", "tree-b", "validation", "Segundo tema exclusivo.", "Su respuesta."),
                   dialogue("c1", "tree-c", "test", "Consulta ajena.", "Texto ajeno.")]
        original_folds = {r["id"]: r["split"] for r in records}
        groups, counts = helper.ambiguous_sft_families(records)
        kept, removed = helper.exclude_whole_families(records, groups)
        self.assertEqual(groups, {"tree-a", "tree-b"})
        self.assertEqual(counts["fullPromptCrossSplit"], 1)
        self.assertEqual({r["id"] for r in removed}, {"a1", "a2", "b1", "b2"})
        self.assertEqual([r["id"] for r in kept], ["c1"])
        self.assertEqual({r["id"]: r["split"] for r in records}, original_folds)

    def test_authored_crossfold_collision_is_not_silently_removed(self):
        records = [dialogue("a", "authored", "train", "Consulta compartida.", "Respuesta.", authored=True),
                   dialogue("b", "public", "validation", "Consulta compartida.", "Otra respuesta.")]
        with self.assertRaisesRegex(ValueError, "authored collision"):
            helper.ambiguous_sft_families(records)

    def test_missing_and_explicit_default_system_have_same_effective_prompt(self):
        implicit = dialogue("implicit", "family-a", "train", "Consulta compartida.", "Primera respuesta.")
        explicit = dialogue("explicit", "family-b", "validation", "Consulta compartida.", "Otra respuesta.")
        explicit["messages"].insert(0, {"role": "system", "content": helper.DEFAULT_SYSTEM_PROMPT})
        groups, counts = helper.ambiguous_sft_families([implicit, explicit])
        self.assertEqual(groups, {"family-a", "family-b"})
        self.assertEqual(counts["fullPromptCrossSplit"], 1)
        self.assertEqual(counts["fullDialogueCrossSplit"], 0)

    def test_authored_record_in_colliding_public_family_is_not_silently_removed(self):
        records = [dialogue("a1", "mixed-family", "train", "Prompt clonado.", "Una respuesta."),
                   dialogue("b1", "public-family", "test", "Prompt clonado.", "Otra respuesta."),
                   dialogue("a2", "mixed-family", "train", "Prompt único.", "Texto propio.", authored=True)]
        with self.assertRaisesRegex(ValueError, "family contains authored"):
            helper.ambiguous_sft_families(records)

    def test_mutated_input_is_rejected_instead_of_relabeling_its_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.jsonl"
            source.write_bytes(b"frozen original bytes\n")
            authorized = {source: helper.file_sha(source)}
            helper.verify_frozen_inputs(authorized)
            source.write_bytes(b"changed during preparation\n")
            with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                helper.verify_frozen_inputs(authorized)
            self.assertNotEqual(authorized[source], helper.file_sha(source))

    def test_full_human_replaces_subset_and_keeps_authored_and_rich(self):
        prior = dialogue("old-synthetic", "synthetic", "train", "Tema viejo.", "Respuesta vieja.", authored=True)
        h1 = dialogue("human-1", "human", "validation", "Tema humano.", "Respuesta humana.")
        h2 = dialogue("human-2", "human", "validation", "Otro tema.", "Otra respuesta.")
        rich = dialogue("rich", "rich-family", "test", "Tema rico.", "Respuesta rica.", authored=True)
        result = helper.sft_union([prior, h1], [h1, h2], [rich])
        self.assertEqual([r["id"] for r in result], ["old-synthetic", "human-1", "human-2", "rich"])
        self.assertEqual([r["split"] for r in result], ["train", "validation", "validation", "test"])

    def test_subset_human_content_mismatch_is_rejected(self):
        original = dialogue("same-id", "same-family", "train", "Consulta.", "Respuesta original.")
        changed = dialogue("same-id", "same-family", "train", "Consulta.", "Respuesta cambiada.")
        with self.assertRaisesRegex(ValueError, "differs from full reviewed"):
            helper.sft_union([changed], [original], [])

    def test_raw_text_crossfold_clones_exclude_both_whole_families(self):
        messages = {"m1": {"text": "Texto idéntico.", "groupId": "family-a", "split": "train"},
                    "m2": {"text": "Texto idéntico.", "groupId": "family-b", "split": "test"},
                    "m3": {"text": "Otro texto.", "groupId": "family-a", "split": "train"}}
        groups, receipt = helper.crossfold_text_clone_families(messages)
        self.assertEqual(groups, {"family-a", "family-b"})
        self.assertEqual(receipt, {"classes": 1, "messageIds": 2, "families": 2})
        records = [text("a1", "family-a", "train", "Texto idéntico."),
                   text("a2", "family-a", "train", "Otro texto."),
                   text("b", "family-b", "test", "Texto idéntico.")]
        self.assertEqual(helper.exclude_whole_families(records, groups)[0], [])

    def test_holdout_budget_keeps_whole_chunks_and_all_train(self):
        records = [text("t", "train-family", "train", "Entrenamiento completo."),
                   text("v1", "small-family", "validation", "abcd"),
                   text("v2", "small-family", "validation", "efgh"),
                   text("v3", "large-family", "validation", "123456789"),
                   text("v4", "large-family", "validation", "ABCDEFGHI")]
        kept, receipt = helper.select_holdouts(records, character_cap=8,
                                               family_limits={"wikimedia/wikipedia": 64})
        self.assertEqual({r["id"] for r in kept}, {"t", "v1", "v2"})
        self.assertEqual(receipt["validation"]["wikimedia/wikipedia"]["selectedCharacters"], 8)
        self.assertEqual([r["split"] for r in kept], ["train", "validation", "validation"])
        reversed_kept, _ = helper.select_holdouts(list(reversed(records)), character_cap=8,
                                                family_limits={"wikimedia/wikipedia": 64})
        self.assertEqual({r["id"] for r in reversed_kept}, {r["id"] for r in kept})

    def test_utf8_chunking_preserves_every_character_and_byte_bound(self):
        original = ("Información: pingüino, 中文 y café.\n\n" * 4_000)
        chunks = helper.split_utf8_exact(original)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), original)
        self.assertTrue(all(0 < len(chunk.encode()) <= 65_536 for chunk in chunks))

    def test_existing_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "already-present"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"keep these bytes")
            with self.assertRaises(FileExistsError):
                helper.create_output_directory(output)
            self.assertEqual(sentinel.read_bytes(), b"keep these bytes")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_existing_jsonl_bytes_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus.jsonl"
            output.write_bytes(b"original bytes\n")
            with self.assertRaises(FileExistsError):
                helper.write_jsonl(output, [text("new", "new-family", "train", "Nuevo contenido.")])
            self.assertEqual(output.read_bytes(), b"original bytes\n")


if __name__ == "__main__":
    unittest.main()
