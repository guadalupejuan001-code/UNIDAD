"""Sequence search must follow model scores and preserve native generation."""
import importlib.util
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(n) is None for n in ('torch','tokenizers','safetensors')):
    raise unittest.SkipTest('Optional conversation dependencies are unavailable')

import torch
from ia_propia.conversation.model import ConversationModel, chat
from ia_propia.conversation.tokenizer import fit_tokenizer, ConversationError
from ia_propia.conversation.runtime import validate_reply_request
from ia_propia.conversation.contract import ASSISTANT_ID, EOS_ID


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.messages = [{'role':'user','content':'El cuaderno es verde. ¿De qué color es?'}]
        cls.tokenizer = fit_tokenizer([{'messages':cls.messages + [{'role':'assistant','content':'Es verde.'}]}],vocab_size=320)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, **extra):
        return ConversationModel({'vocab_size':self.tokenizer.get_vocab_size(),'d_model':16,
                                  'n_layers':1,'n_heads':2,'context_length':64,
                                  'ffn_multiplier':2,'dropout':0.0,'seed':17,**extra})

    def test_beam_recovers_more_likely_sequence_after_inferior_first_choice(self):
        model = self.model(context_length=8)
        observed = []
        def distribution(ids, mask):
            self.assertEqual(mask.sum().item(), ids.shape[0])
            self.assertTrue(bool(mask[:, -1].all()))
            self.assertLessEqual(ids.shape[1], 8)
            observed.append(ids.clone())
            logits = torch.full((ids.shape[0],self.tokenizer.get_vocab_size()),-100.0)
            for row,last in enumerate(ids[:,-1].tolist()):
                if last == ASSISTANT_ID:
                    logits[row,100] = torch.log(torch.tensor(0.6))
                    logits[row,101] = torch.log(torch.tensor(0.4))
                elif last == 100:
                    logits[row,EOS_ID] = torch.log(torch.tensor(0.5))
                    logits[row,102] = torch.log(torch.tensor(0.5))
                else:
                    logits[row,EOS_ID] = torch.log(torch.tensor(0.99))
                    logits[row,102] = torch.log(torch.tensor(0.01))
            return logits
        rng = torch.get_rng_state().clone()
        with patch.object(model,'selected_logits',side_effect=distribution):
            greedy = chat(model,self.tokenizer,self.messages,max_new_tokens=3,temperature=0)
            beam = chat(model,self.tokenizer,self.messages,max_new_tokens=3,temperature=0,beam_width=2)
        self.assertEqual(greedy['generatedTokenIds'],[100,EOS_ID])
        self.assertEqual(beam['generatedTokenIds'],[101,EOS_ID])
        self.assertEqual(beam['stopReason'],'eos')
        self.assertAlmostEqual(beam['sequenceLogProbability'],float(torch.log(torch.tensor(0.396))),places=5)
        self.assertEqual(beam['promptTokens'],greedy['promptTokens'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_default_search_preserves_full_projection_generation(self):
        for head in ({},{'copy_head':{'version':1,'rank':32,'seed':23}}):
            model=self.model(**head)
            for temperature in (0,0.7):
                with self.subTest(head=bool(head),temperature=temperature):
                    efficient=chat(model,self.tokenizer,self.messages,max_new_tokens=8,temperature=temperature,seed=11)
                    with patch.object(model,'selected_logits',side_effect=lambda ids,mask:model(ids)[mask]):
                        original=chat(model,self.tokenizer,self.messages,max_new_tokens=8,temperature=temperature,seed=11)
                    self.assertEqual(efficient,original)

    def test_search_is_bounded_and_deterministic_mode_is_explicit(self):
        model=self.model()
        for width in (0,9,True,2.0,'2'):
            with self.subTest(width=width), self.assertRaises((ValueError,ConversationError)):
                chat(model,self.tokenizer,self.messages,temperature=0,beam_width=width)
        with self.assertRaises(ConversationError):
            chat(model,self.tokenizer,self.messages,temperature=0.7,beam_width=2)
        request=validate_reply_request({'messages':self.messages,'beam_width':2,'temperature':0})
        self.assertEqual(request['beam_width'],2)
        with self.assertRaises((ValueError,ConversationError)):
            validate_reply_request({'messages':self.messages,'beam_width':9})
