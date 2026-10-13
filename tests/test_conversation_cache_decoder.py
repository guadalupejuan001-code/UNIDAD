"""Execution equivalence and selection policy; these are not language assessments."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "tokenizers", "safetensors")):
    raise unittest.SkipTest("Optional conversation dependencies unavailable")

import torch
from ia_propia.conversation.artifacts import save_artifact
from ia_propia.conversation.decoding import torch_token, validate_controls
from ia_propia.conversation.generation_cache import GenerationCache
from ia_propia.conversation.model import ConversationModel, chat
from ia_propia.conversation.nerva_decoder import NervaDecoder
from ia_propia.conversation.runtime import ConversationEngine, validate_reply_request
from ia_propia.conversation.tokenizer import ConversationError, fit_tokenizer


class GenerationFixture:
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)
        cls.messages = [{"role": "system", "content": "Español."}, {"role": "user", "content": "La caja es verde."}]
        cls.tokenizer = fit_tokenizer([{"messages": cls.messages + [{"role": "assistant", "content": "Es verde."}]}], vocab_size=320)

    def model(self, copy=False, **kwargs):
        return ConversationModel({"vocab_size": self.tokenizer.get_vocab_size(), "d_model": 32,
            "n_layers": 2, "n_heads": 4, "context_length": 64, "ffn_multiplier": 2, "seed": 73,
            **({"copy_head": {"version": 1, "rank": 32, "seed": 19}} if copy else {}), **kwargs})


class CachedGenerationTests(GenerationFixture, unittest.TestCase):
    def test_logits_match_full_context_including_copy_head_and_window_reset(self):
        for copy in (False, True):
            model = self.model(copy, context_length=32).eval()
            ids = [1, 4, 30, 31, 32, 2, 5]
            cache = GenerationCache(model)
            with torch.inference_mode():
                for step in range(35):
                    inputs = torch.tensor([ids[-32:]])
                    mask = torch.zeros_like(inputs, dtype=torch.bool)
                    mask[:, -1] = True
                    expected = model.selected_logits(inputs, mask)[0]
                    actual = cache.last_logits(inputs)
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    ids.append(40 + step)
                # Every full window after the first must rebuild its positions.
                self.assertEqual(cache.incremental_steps, 25)
                self.assertEqual(cache.prefills, 10)
                changed = inputs.clone()
                changed[0, 3] = 99
                torch.testing.assert_close(cache.last_logits(changed), model.selected_logits(changed, mask)[0], rtol=2e-5, atol=2e-6)

    def test_cached_generation_preserves_rng_weights_modes_and_reference_tokens(self):
        for copy in (False, True):
            model = self.model(copy, dropout=.2)
            state = torch.get_rng_state().clone()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            for temperature in (0, .7):
                common = dict(max_new_tokens=16, temperature=temperature, seed=13)
                reference = chat(model, self.tokenizer, self.messages, use_cache=False, **common)
                actual = chat(model, self.tokenizer, self.messages, **common)
                self.assertEqual(actual.pop("decoding")["kvCache"], True)
                reference.pop("decoding")
                self.assertEqual(actual, reference)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_cache_is_request_local_under_concurrency_and_exceptions(self):
        model = self.model()
        def run(text):
            return chat(model, self.tokenizer, [{"role": "user", "content": text}], max_new_tokens=12, temperature=0)
        prompts = ("verde", "azul", "verde")
        expected = list(map(run, prompts))
        with ThreadPoolExecutor(max_workers=3) as pool:
            self.assertEqual(list(pool.map(run, prompts)), expected)
        with patch.object(GenerationCache, "last_logits", side_effect=RuntimeError("deliberate test failure")):
            with self.assertRaises(RuntimeError):
                run("verde")
        self.assertTrue(model.training)
        self.assertEqual(run("verde"), expected[0])

    def test_controls_are_validated_and_host_fields_cannot_enter_requests(self):
        for controls in ({"top_k": True}, {"top_k": -1}, {"top_p": 0}, {"top_p": float("nan")},
                         {"repetition_penalty": .9}, {"repetition_penalty": 3}, {"top_k": 65537}):
            with self.assertRaises(ValueError):
                validate_controls(**controls)
        for key in ("decoder_backend", "use_cache", "node", "program"):
            with self.assertRaises(ValueError):
                validate_reply_request({"messages": self.messages, key: "nerva"})

    def test_torch_neutral_controls_preserve_original_sampling(self):
        logits = torch.tensor([.3, .7, -.8, 1.2])
        for temperature in (0, .7):
            gen = torch.Generator().manual_seed(42)
            ref_gen = torch.Generator().manual_seed(42)
            reference = logits.clone()
            reference[1] = -torch.inf
            expected = int(torch.argmax(reference)) if temperature == 0 else int(torch.multinomial(torch.softmax(reference / temperature, -1), 1, generator=ref_gen))
            self.assertEqual(torch_token(logits, [1], [], temperature=temperature, generator=gen), expected)


@unittest.skipUnless(shutil.which("node"), "NERVA requires Node.js 18+")
class NervaConversationTests(GenerationFixture, unittest.TestCase):
    def test_actual_worker_masks_ties_penalizes_repetition_and_filters_sampling(self):
        controls = validate_controls()
        with NervaDecoder(42) as decoder:
            self.assertEqual(decoder.select(torch.tensor([9., 3., 3., 2.]), [0], [], temperature=0, **controls), 1)
            self.assertEqual(decoder.select(torch.tensor([9., 3., 3., 2.]), [0], [1], temperature=0,
                                            **validate_controls(repetition_penalty=1.2)), 2)
            # A nucleus that fits in the highest token and top-k=1 are deterministic.
            for filtered in (validate_controls(top_k=1), validate_controls(top_p=.1)):
                for _ in range(5):
                    self.assertEqual(decoder.select(torch.tensor([9., 3., 2., 1.]), [0], [], temperature=.7, **filtered), 1)
            self.assertIsNone(decoder.process.poll())
        self.assertIsNotNone(decoder.process.poll())
        self.assertFalse(Path(decoder.directory.name).exists())

    def test_nucleus_keeps_crossing_token_and_never_samples_excluded_tail(self):
        # Exact probabilities .50, .30, .20; p=.60 must keep the first TWO.
        scores = torch.tensor([.5, .3, .2], dtype=torch.float64).log()
        with NervaDecoder(42) as decoder:
            for draw, expected in ((0, 0), (.6, 0), (.7, 1), (.99999, 1)):
                with patch.object(decoder.rng, "random", return_value=draw):
                    self.assertEqual(decoder.select(scores, [], [], temperature=1,
                        **validate_controls(top_p=.6)), expected)
            # A negative repeated logit becomes more negative, not more likely.
            self.assertEqual(decoder.select(torch.tensor([-1., -1.1]), [], [0], temperature=0,
                                            **validate_controls(repetition_penalty=1.2)), 1)

    def test_engine_and_brain_execute_nerva_and_keep_identity(self):
        from ia_propia.brain import Brain
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            manifest = save_artifact(artifact, self.model(copy=True), self.tokenizer,
                                    training={"fixtureOnly": True}, evaluation={"fixtureOnly": True})
            reference = ConversationEngine(artifact, use_cache=False).reply(self.messages, max_new_tokens=12, temperature=0)
            brain = Brain(Path(directory) / "state", conversation_model=artifact, conversation_decoder="nerva")
            result = brain.connect("test").request({"schemaVersion": 1, "requestId": "nerva-conversation",
                "operation": "conversation.reply", "input": {"messages": self.messages, "max_new_tokens": 12, "temperature": 0}})
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(result["result"]["generatedTokenIds"], reference["generatedTokenIds"])
            self.assertEqual(result["result"]["modelId"], manifest["modelId"])
            self.assertEqual(result["result"]["decoding"]["backend"], "nerva")
            self.assertEqual(brain.models.capabilities()["models"]["conversation"]["decoderBackend"], "nerva")

    def test_sampling_repeats_with_own_seed_and_does_not_touch_global_rng(self):
        model = self.model()
        before = random.getstate()
        settings = dict(max_new_tokens=10, temperature=.7, decoder_backend="nerva", top_k=20, top_p=.9)
        first = chat(model, self.tokenizer, self.messages, **settings)
        self.assertEqual(first, chat(model, self.tokenizer, self.messages, **settings))
        self.assertEqual(before, random.getstate())

    def test_failure_does_not_fallback_and_cleans_up_worker(self):
        with self.assertRaises(ConversationError):
            with NervaDecoder(42, node="/nonexistent/unidad-node"):
                self.fail("Missing Node must fail explicitly")
        decoder = NervaDecoder(42)
        with self.assertRaises(ConversationError):
            with decoder:
                decoder.select(torch.tensor([1., 2.]), [0, 1], [], temperature=0, **validate_controls())
        self.assertIsNotNone(decoder.process.poll())

    def test_timeout_closes_worker_and_temporary_files(self):
        decoder = NervaDecoder(42, timeout=.001)
        with self.assertRaisesRegex(ConversationError, "exceeded"):
            with decoder, patch.object(decoder, "_exchange", side_effect=lambda _: time.sleep(.05)):
                decoder.select(torch.tensor([1., 2.]), [], [], temperature=0, **validate_controls())
        self.assertIsNotNone(decoder.process.poll())
        self.assertFalse(Path(decoder.directory.name).exists())


if __name__ == "__main__":
    unittest.main()
