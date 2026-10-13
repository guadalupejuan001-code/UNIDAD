"""Behavioral acceptance of the optional specialized worker, without training."""
import importlib.util
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ('torch', 'tokenizers', 'safetensors')):
    raise unittest.SkipTest('Optional conversation dependencies unavailable')

import torch
from ia_propia.conversation.model import chat
from ia_propia.conversation.nerva_decoder import NervaDecoder
from ia_propia.conversation.decoding import validate_controls
from ia_propia.conversation.tokenizer import ConversationError
from ia_propia.conversation.runtime import ConversationEngine, validate_reply_request
from ia_propia.conversation.artifacts import save_artifact
from test_conversation_cache_decoder import GenerationFixture


@unittest.skipUnless(shutil.which('node'), 'Node required')
class FastNervaTests(GenerationFixture, unittest.TestCase):
    def test_real_workers_match_tokens_with_same_draws_and_inputs(self):
        rng = random.Random(341)
        with NervaDecoder(19) as original, NervaDecoder(19, optimized=True) as fast:
            for size in (3, 128, 2048, 8192):
                scores = torch.tensor([rng.uniform(-12, 12) for _ in range(size)])
                before = scores.clone()
                for temperature in (0, .000001, .7, 2):
                    for top_k, top_p in ((0, 1), (1, 1), (7, .8)):
                        settings = dict(temperature=temperature, **validate_controls(top_k, top_p, 1.3))
                        args = (scores, [0, 0], [size - 1, size - 1])
                        self.assertEqual(original.select(*args, **settings), fast.select(*args, **settings))
                self.assertTrue(torch.equal(before, scores))

    def test_full_generation_matches_original_nerva_and_preserves_state(self):
        for copy in (False, True):
            model = self.model(copy=copy)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            for temperature in (0, .7):
                settings = dict(max_new_tokens=20, temperature=temperature, seed=73, top_k=20, top_p=.8, repetition_penalty=1.2)
                expected = chat(model, self.tokenizer, self.messages, decoder_backend='nerva', **settings)
                actual = chat(model, self.tokenizer, self.messages, decoder_backend='nerva-fast', **settings)
                self.assertEqual(actual['decoding'].pop('backend'), 'nerva-fast')
                expected['decoding'].pop('backend')
                self.assertEqual(actual, expected)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_engine_and_brain_share_the_opt_in_path_and_identity(self):
        from ia_propia.brain import Brain
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'artifact'
            manifest = save_artifact(artifact, self.model(copy=True), self.tokenizer,
                                    training={'fixtureOnly': True}, evaluation={'fixtureOnly': True})
            engine = ConversationEngine(artifact, decoder_backend='nerva-fast')
            expected = engine.reply(self.messages, max_new_tokens=12, temperature=0)
            brain = Brain(Path(directory) / 'state', conversation_model=artifact, conversation_decoder='nerva-fast')
            result = brain.connect('test').request({'schemaVersion': 1, 'requestId': 'fast',
                'operation': 'conversation.reply', 'input': {'messages': self.messages, 'max_new_tokens': 12, 'temperature': 0}})
            self.assertEqual(result['status'], 'ok', result)
            self.assertEqual(result['result']['generatedTokenIds'], expected['generatedTokenIds'])
            self.assertEqual(result['result']['modelId'], manifest['modelId'])
            self.assertEqual(engine.execution['hostRuntimeRequired'], 'Node.js 18+')
        with self.assertRaises(ValueError):
            validate_reply_request({'messages': self.messages, 'decoder_backend': 'nerva-fast'})

    def test_errors_and_timeout_never_fallback_and_cleanup_resources(self):
        decoder = NervaDecoder(1, optimized=True)
        with self.assertRaises(ConversationError):
            with decoder:
                decoder.select(torch.tensor([1., 2.]), [0, 1], [], temperature=0, **validate_controls())
        self.assertIsNotNone(decoder.process.poll())
        self.assertFalse(Path(decoder.directory.name).exists())
        with self.assertRaises(ConversationError):
            with NervaDecoder(1, optimized=True, node='/nonexistent/node'):
                self.fail('Missing Node must fail')
        import time
        decoder = NervaDecoder(1, optimized=True, timeout=.001)
        with self.assertRaisesRegex(ConversationError, 'exceeded'):
            with decoder, patch.object(decoder, '_exchange', side_effect=lambda _: time.sleep(.05)):
                decoder.select(torch.tensor([1., 2.]), [], [], temperature=0, **validate_controls())
        self.assertIsNotNone(decoder.process.poll())
        self.assertFalse(Path(decoder.directory.name).exists())

    def test_host_switch_rejects_coercion(self):
        for value in (1, 'true', None):
            with self.assertRaises(ConversationError):
                NervaDecoder(1, optimized=value)
