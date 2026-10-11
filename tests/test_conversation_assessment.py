import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.conversation.assessment import assess, load_cases


HAS_EXTRAS = all(importlib.util.find_spec(p) is not None for p in ("torch", "tokenizers", "safetensors"))


def small_suite():
    candidates = ("El martes.", "El jueves.")
    cases = []
    for index, day in enumerate(("martes", "jueves")):
        cases.append({"id": f"case-{index}", "familyId": "corrections", "pairId": "date",
            "messages": [{"role": "user", "content": f"La entrega es el {day}. ¿Cuándo entrego?"}],
            "referenceGoodReply": candidates[index], "plausibleDecoyResponse": candidates[1 - index],
            "candidateOrder": ["referenceGoodReply", "plausibleDecoyResponse"] if index == 0 else
                              ["plausibleDecoyResponse", "referenceGoodReply"],
            "expectedBehaviors": ["Usar la fecha explícita."]})
    return {"schemaVersion": 1, "evaluationId": "unit-development", "purpose": "development",
            "counts": {"cases": 2, "pairs": 1, "families": 1}, "cases": cases}


@unittest.skipUnless(HAS_EXTRAS, "conversation optional dependencies unavailable")
class ConversationAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cases_path = self.root / "cases.json"
        self.suite = small_suite()
        self.write_suite()

    def tearDown(self):
        self.temporary.cleanup()

    def write_suite(self):
        self.cases_path.write_text(json.dumps(self.suite, ensure_ascii=False), encoding="utf-8")

    def build_model(self, context=256):
        from ia_propia.conversation.model import ConversationModel
        from ia_propia.conversation.tokenizer import fit_tokenizer
        records = [{"messages": [*c["messages"], {"role": "assistant", "content": c["referenceGoodReply"]}]}
                   for c in self.suite["cases"]]
        tokenizer = fit_tokenizer(records, vocab_size=300)
        model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 8,
            "n_layers": 1, "n_heads": 2, "context_length": context, "dropout": 0.2, "ffn_multiplier": 1})
        return model, tokenizer

    def test_suite_integrity_and_counterfactual_swaps_are_required(self):
        suite, digest = load_cases(self.cases_path)
        self.assertEqual(len(digest), 64)
        self.assertEqual(suite["counts"]["pairs"], 1)
        for mutation in (
            lambda s: s["cases"][1].update(referenceGoodReply="Otra respuesta."),
            lambda s: s["cases"][1].update(id="case-0"),
            lambda s: s["counts"].update(cases=True),
            lambda s: s.update(purpose="final-test"),
            lambda s: s["cases"][1].update(messages=s["cases"][0]["messages"]),
            lambda s: s["cases"][0].update(candidateOrder=["referenceGoodReply"] * 2),
            lambda s: s.update(source={"finalTest": True}),
            lambda s: s.update(pairs=[{"id": "other"}]),
            lambda s: s.update(families=[{"id": "other"}]),
        ):
            self.suite = small_suite()
            mutation(self.suite)
            self.write_suite()
            with self.assertRaises(ValueError):
                load_cases(self.cases_path)
        self.cases_path.write_text('{"schemaVersion":1,"schemaVersion":1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_cases(self.cases_path)

    def test_scoring_matches_autoregressive_chat_context_and_preserves_model(self):
        import torch
        from torch.nn import functional as F
        from ia_propia.conversation.scoring import score_reply
        from ia_propia.conversation.tokenizer import EOS_ID, encode_prompt
        for context in (256, 16):
            with self.subTest(context=context):
                model, tokenizer = self.build_model(context)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                messages = self.suite["cases"][0]["messages"]
                text = self.suite["cases"][0]["referenceGoodReply"]
                result = score_reply(model, tokenizer, messages, text)
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for name, value in before.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]))
                prompt, _ = encode_prompt(messages, tokenizer, context)
                targets = tokenizer.encode(text, add_special_tokens=False).ids + [EOS_ID]
                expected = []
                model.eval()
                with torch.inference_mode():
                    for target in targets:
                        logits = model(torch.tensor([prompt[-context:]], dtype=torch.long))[0, -1]
                        expected.append(float(F.log_softmax(logits, dim=-1)[target]))
                        prompt.append(target)
                self.assertAlmostEqual(result["totalLogProbability"], math.fsum(expected), places=5)
                self.assertEqual(result["targetTokens"], len(targets))
                self.assertEqual(result["targetsWithSlidingPrompt"] > 0, context == 16)

    def test_uniform_distribution_has_known_likelihood_including_eos(self):
        import torch
        from ia_propia.conversation.scoring import score_reply
        model, tokenizer = self.build_model()
        def uniform(inputs):
            return torch.zeros((*inputs.shape, tokenizer.get_vocab_size()))
        with patch.object(model, "forward", side_effect=uniform):
            result = score_reply(model, tokenizer, self.suite["cases"][0]["messages"], "El martes.")
        self.assertAlmostEqual(result["meanNll"], math.log(tokenizer.get_vocab_size()), places=6)
        self.assertTrue(result["includesEos"])

    def test_unconditional_preferences_cancel_and_do_not_pass_both_cases(self):
        class UnconditionalEngine:
            model_id = "unit-model"
            manifest = {"modelId": model_id}
            def __init__(self, path):
                pass
            def score_reply(self, messages, text):
                # An unconditional phrase preference must fail the swapped pair.
                total = -2.0 if text == "El martes." else -7.0
                return {"totalLogProbability": total, "meanLogProbability": total / 3}
            def reply(self, messages, **settings):
                return {"text": "Respuesta real de la instancia de prueba."}
        with patch("ia_propia.conversation.runtime.ConversationEngine", UnconditionalEngine):
            result = assess("unused", self.cases_path, torch_threads=1)
        self.assertEqual(result["pairs"][0]["contextSensitivity"], 0)
        self.assertFalse(result["pairs"][0]["bothReferencesPreferred"])
        self.assertEqual(result["familyMacroPairWinRate"], 0)
        self.assertFalse(result["humanReviewCompleted"])
        self.assertFalse(result["generalConversationValidated"])

    def test_assessment_uses_one_real_artifact_and_does_not_touch_training(self):
        from ia_propia.conversation.artifacts import load_artifact, save_artifact
        model, tokenizer = self.build_model()
        artifact = self.root / "model"
        save_artifact(artifact, model, tokenizer, training={"trainedFrom": "random_weights"})
        output = self.root / "report.json"
        with patch("ia_propia.conversation.artifacts.load_artifact", wraps=load_artifact) as loading, \
             patch("ia_propia.conversation.training.finalize", side_effect=AssertionError("No test evaluation")):
            report = assess(artifact, self.cases_path, output=output, max_new_tokens=3, torch_threads=1)
            self.assertEqual(loading.call_count, 1)
        self.assertEqual(json.loads(output.read_text()), report)
        self.assertEqual(report["cases"][0]["generation"]["modelId"], report["modelId"])
        self.assertFalse(report["weightsUpdated"])
        self.assertFalse(report["testEvaluated"])
        self.assertFalse((self.root / "evaluation.json").exists())
        original = output.read_bytes()
        with patch("ia_propia.conversation.runtime.ConversationEngine") as loading:
            with self.assertRaises(FileExistsError):
                assess(artifact, self.cases_path, output=output)
            loading.assert_not_called()
        self.assertEqual(output.read_bytes(), original)
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.root / "absent.json")
        with patch("ia_propia.conversation.runtime.ConversationEngine") as loading:
            with self.assertRaises(FileExistsError):
                assess(artifact, self.cases_path, output=dangling)
            loading.assert_not_called()
        run = self.root / "run"
        (run / "checkpoints").mkdir(parents=True)
        (run / "status.json").write_text("{}", encoding="utf-8")
        with patch("ia_propia.conversation.runtime.ConversationEngine") as loading:
            with self.assertRaisesRegex(ValueError, "outside training"):
                assess(artifact, self.cases_path, output=run / "evaluation.json")
            loading.assert_not_called()
        self.assertFalse((run / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
