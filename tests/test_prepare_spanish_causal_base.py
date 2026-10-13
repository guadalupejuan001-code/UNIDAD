"""Source-preserving causal data guards; no network, model or tokenizer calls."""
import copy
import importlib.util
from pathlib import Path
import unittest

_PATH = Path(__file__).resolve().parents[1] / "examples" / "prepare_spanish_causal_base.py"
_SPEC = importlib.util.spec_from_file_location("prepare_spanish_causal_base", _PATH)
helper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helper)


def record(identifier, group, split, text):
    return {"id": identifier, "groupId": group, "split": split,
            "source": {"dataset": "fictional-source-fixture", "kind": "test-fixture"}, "text": text}


class SpanishCausalPreparationTests(unittest.TestCase):
    def test_paragraph_filter_keeps_original_words_and_drops_headings_and_lists(self):
        paragraph = ("En la ciudad se construyó una biblioteca para que los vecinos pudieran estudiar "
                     "y compartir los libros de distintas épocas, con espacios abiertos para todas las familias.")
        text = "Título\n\n  " + paragraph + "  \n\n1991 1992 1993 1994\nReferencias"
        kept, reasons = helper.prose_paragraphs(text)
        self.assertEqual(kept, [paragraph])
        self.assertGreater(reasons["short_heading_list_or_fragment"], 0)
        self.assertIn(paragraph, text)

    def test_crossfold_paragraph_clones_implicate_complete_article_families(self):
        rows = [record("a", "article-a", "train", "Texto completo A."),
                record("a-sibling", "article-a", "train", "Texto independiente del mismo documento."),
                record("b", "article-b", "validation", "Texto completo B."),
                record("c", "article-c", "test", "Contenido diferente.")]
        original = copy.deepcopy(rows)
        paragraphs = {"a": ["Un párrafo compartido."], "b": ["Un  párrafo\ncompartido."]}
        groups, receipt = helper.clone_families(rows, paragraphs)
        self.assertEqual(groups, {"article-a", "article-b"})
        self.assertEqual(receipt["crossfoldNormalizedTextClasses"], 1)
        self.assertEqual([r["id"] for r in rows if r["groupId"] not in groups], ["c"])
        self.assertEqual(rows, original)

    def test_budget_selection_preserves_whole_families_and_existing_folds(self):
        rows = [record("a1", "a", "train", "a" * 12), record("a2", "a", "train", "b" * 12),
                record("b", "b", "train", "c" * 30), record("v", "v", "validation", "d" * 20),
                record("t", "t", "test", "e" * 20)]
        original = copy.deepcopy(rows)
        budgets = {"train": 1, "validation": 1, "test": 1}
        selected, receipt = helper.select_families(rows, budgets, "fixed-fixture")
        reversed_selected, reversed_receipt = helper.select_families(list(reversed(rows)), budgets, "fixed-fixture")
        self.assertEqual({r["id"] for r in selected}, {r["id"] for r in reversed_selected})
        self.assertEqual(receipt, reversed_receipt)
        for group in {r["groupId"] for r in selected}:
            self.assertEqual({r["id"] for r in selected if r["groupId"] == group},
                             {r["id"] for r in rows if r["groupId"] == group})
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
