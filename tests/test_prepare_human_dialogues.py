"""Fixture-only public dialogue leakage guards; no sources or models loaded."""

import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Human dialogue preparation requires optional tokenizers")

from ia_propia.conversation.data import load_corpus
from ia_propia.conversation.tokenizer import fit_tokenizer

_PATH = Path(__file__).resolve().parents[1] / "examples" / "prepare_human_dialogues.py"
_SPEC = importlib.util.spec_from_file_location("prepare_human_dialogues", _PATH)
helper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helper)


def dialogue(identifier, group, split, question, answer):
    return {"id": identifier, "groupId": group, "split": split,
            "source": {"dataset": "fictional-public-fixture", "kind": "public-human-reviewed"},
            "messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


class HumanDialoguePreparationTests(unittest.TestCase):
    def test_explicit_foreign_self_claims_are_detected_in_spanish_clauses(self):
        for text in ("Soy ChatGPT.", "Me llamo Open Assistant.",
                     "Como OpenAssistant, puedo ayudarte.",
                     "Soy una IA creada por OpenAI.",
                     "Soy un asistente virtual desarrollado por OpenAI.",
                     "Soy ChatGPT, ¿en qué puedo ayudarte?",
                     "¡Hola! Soy ChatGPT.", "Hola, soy OpenAssistant.",
                     'Soy "ChatGPT".',
                     "Puedo ayudarte con tu pregunta. Fui creado por OpenAI."):
            with self.subTest(text=text):
                self.assertIsNotNone(helper.foreign_self_identity(text))

    def test_mentions_negations_questions_quotes_and_code_are_not_self_claims(self):
        for text in ("ChatGPT es un asistente de inteligencia artificial.",
                     "Open Assistant es un proyecto público.",
                     "OpenAI creó ChatGPT.",
                     "No soy ChatGPT; soy Unidad.",
                     "Nunca fui creado por OpenAI.",
                     "Soy un asistente similar a ChatGPT.",
                     "¿Soy ChatGPT?", "Soy ChatGPT?",
                     "Ella dijo: “Soy ChatGPT”.",
                     "La frase «Soy ChatGPT» es un ejemplo.",
                     '"Soy ChatGPT."',
                     "No soy ChatGPT. La cita es: 'Soy ChatGPT.'",
                     "Ejemplo: '\nSoy ChatGPT.\n'",
                     "> Soy ChatGPT.\n\nEsta es una cita, no mi identidad.",
                     "Ejemplo: `Soy ChatGPT.`",
                     "Ejemplo: `\nSoy ChatGPT.\n`",
                     "```text\nSoy ChatGPT.\n```",
                     "```text\nSoy ChatGPT.\n",
                     "    Soy ChatGPT.\n"):
            with self.subTest(text=text):
                self.assertIsNone(helper.foreign_self_identity(text))

    def test_foreign_self_claim_outside_selection_excludes_whole_selected_family_only(self):
        selected = [dialogue("a1", "family-a", "train", "Primera pregunta.", "Respuesta correcta."),
                    dialogue("a2", "family-a", "train", "Segundo tema.", "Respuesta diferente."),
                    dialogue("b", "family-b", "validation", "Otra pregunta.", "ChatGPT es un modelo de lenguaje."),
                    dialogue("c", "family-c", "test", "Soy ChatGPT.", "Tú escribiste ese nombre.")]
        source = selected + [dialogue("a3", "family-a", "train", "¿Quién eres?", "Soy OpenAssistant.")]
        original_selected, original_source = copy.deepcopy(selected), copy.deepcopy(source)
        kept, receipt = helper.filter_foreign_self_identity(selected, source_records=source)
        self.assertEqual(kept, selected[2:])
        self.assertEqual(selected, original_selected)
        self.assertEqual(source, original_source)
        self.assertEqual(helper.filter_foreign_self_identity(selected)[0], selected)

    def test_self_identity_guard_excludes_whole_families_in_every_fold(self):
        selected, source = [], []
        claims = {"train": "Mi nombre es Open Assistant.",
                  "validation": "Fui desarrollado por OpenAI.",
                  "test": "Soy ChatGPT."}
        for split, claim in claims.items():
            group = "identity-family-" + split
            siblings = [dialogue(split + "-a", group, split, "Una consulta.", "Respuesta útil."),
                        dialogue(split + "-b", group, split, "Otra consulta.", "Otra respuesta.")]
            selected.extend(siblings)
            source.extend(siblings + [dialogue(split + "-identity", group, split, "¿Quién eres?", claim)])
        control = dialogue("control", "explained-brand", "validation", "¿Qué es ChatGPT?", "ChatGPT es un modelo.")
        selected.append(control)
        source.append(control)
        original = copy.deepcopy(selected)
        kept, receipt = helper.filter_foreign_self_identity(selected, source)
        self.assertEqual(kept, [control])
        self.assertEqual(receipt["selectedFamiliesExcluded"], 3)
        self.assertEqual(receipt["selectedRecordsExcludedBySplit"], {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(receipt["historicalFoldsMoved"], 0)
        self.assertEqual(selected, original)

    def test_prompt_clones_exclude_both_families_including_unique_siblings(self):
        records = [dialogue("a1", "tree-a", "train", "Consulta  compartida.", "Primera respuesta."),
                   dialogue("a2", "tree-a", "train", "Consulta exclusiva A.", "Respuesta exclusiva A."),
                   dialogue("b1", "tree-b", "validation", " Consulta\ncompartida. ", "Segunda respuesta."),
                   dialogue("b2", "tree-b", "validation", "Consulta exclusiva B.", "Respuesta exclusiva B."),
                   dialogue("c", "tree-c", "test", "Tema independiente.", "Su respuesta.")]
        original = copy.deepcopy(records)
        groups, receipt = helper.crossfold_families(records)
        self.assertEqual(groups, {"tree-a", "tree-b"})
        self.assertEqual(receipt["prompt"], {"classes": 1, "families": 2})
        self.assertEqual(receipt["dialogue"], {"classes": 0, "families": 0})
        self.assertEqual([r["id"] for r in records if r["groupId"] not in groups], ["c"])
        self.assertEqual(records, original)

    def test_shared_answer_text_is_guarded_even_when_prompts_differ(self):
        records = [dialogue("a", "family-a", "train", "Pregunta primera.", "Una respuesta común."),
                   dialogue("b", "family-b", "test", "Pregunta segunda.", "Una\trespuesta común.")]
        groups, receipt = helper.crossfold_families(records)
        self.assertEqual(groups, {"family-a", "family-b"})
        self.assertEqual(receipt["messageText"], {"classes": 1, "families": 2})
        self.assertEqual(receipt["prompt"]["classes"], 0)
        self.assertEqual(receipt["dialogue"]["classes"], 0)

    def test_dialogue_clone_decisions_are_order_independent_and_keep_samefold_branches(self):
        records = [dialogue("a", "family-a", "train", "Pregunta repetida.", "Respuesta repetida."),
                   dialogue("b", "family-b", "test", "Pregunta  repetida.", "Respuesta\nrepetida."),
                   dialogue("c", "family-c", "validation", "Otra consulta.", "Primera alternativa."),
                   dialogue("d", "family-c", "validation", "Otra consulta.", "Segunda alternativa.")]
        groups, receipt = helper.crossfold_families(records)
        self.assertEqual(groups, {"family-a", "family-b"})
        self.assertEqual(receipt["dialogue"], {"classes": 1, "families": 2})
        self.assertEqual(helper.crossfold_families(list(reversed(records))), (groups, receipt))
        kept = [r for r in records if r["groupId"] not in groups]
        self.assertEqual([r["id"] for r in kept], ["c", "d"])
        self.assertEqual(helper.crossfold_families(kept)[0], set())

    def test_normalized_guard_and_canonical_output_preserve_source_content_and_folds(self):
        records = [dialogue("a", "family-a", "train", "  Café\tcon leche.\n", "Sí,  gracias.\n"),
                   dialogue("b", "family-b", "validation", "Otra  pregunta.", "Otra\trespuesta.")]
        original = copy.deepcopy(records)
        normalized = helper.normalized_messages(records[0])
        self.assertEqual(normalized[0]["content"], "Café con leche.")
        self.assertEqual(helper.crossfold_families(records)[0], set())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.jsonl"
            receipt = helper.write_records(path, records)
            self.assertEqual(load_corpus(path), original)
            self.assertEqual(receipt["splits"], {"train": 1, "validation": 1})
            self.assertEqual(receipt["sha256"], helper.sha(path))
        self.assertEqual(records, original)

    def test_source_failure_leaves_no_output_or_partial_preparation(self):
        fixture = dialogue("fixture", "fixture-family", "train", "Pregunta ficticia.", "Respuesta ficticia.")
        tokenizer = fit_tokenizer([fixture], vocab_size=300)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            args = argparse.Namespace(raw_directory=raw, output=root / "output",
                                      tokenizer=tokenizer_path,
                                      tokenizer_sha256=helper.sha(tokenizer_path), download=False,
                                      max_sequence_tokens=513, min_assistant_tokens=1,
                                      max_assistant_tokens=95, max_pairs=4, max_records=5000)
            source = helper.SOURCES[0]
            for corrupt in (False, True):
                if corrupt:
                    helper.raw_path(raw, source, source["file"]).write_bytes(b"not the pinned source")
                with self.subTest(corrupt=corrupt), self.assertRaisesRegex(
                        ValueError, "hash mismatch" if corrupt else "missing pinned source"):
                    helper.prepare(args)
                self.assertFalse(args.output.exists())
                self.assertEqual({path.name for path in root.iterdir()}, {"raw", "tokenizer.json"})


if __name__ == "__main__":
    unittest.main()
