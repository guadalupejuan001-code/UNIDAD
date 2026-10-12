"""Final-position projection must preserve full-context native generation."""
import importlib.util
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(n) is None for n in ('torch','tokenizers','safetensors')):
    raise unittest.SkipTest('Optional conversation dependencies are unavailable')

import torch
from ia_propia.conversation.model import ConversationModel, chat
from ia_propia.conversation.tokenizer import fit_tokenizer


class GenerationProjectionTests(unittest.TestCase):
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

    def test_final_projection_preserves_full_context_generation(self):
        for head in ({},{'copy_head':{'version':1,'rank':32,'seed':23}}):
            model=self.model(**head)
            for temperature in (0,0.7):
                with self.subTest(head=bool(head),temperature=temperature):
                    efficient=chat(model,self.tokenizer,self.messages,max_new_tokens=8,temperature=temperature,seed=11)
                    with patch.object(model,'selected_logits',side_effect=lambda ids,mask:model(ids)[mask]):
                        original=chat(model,self.tokenizer,self.messages,max_new_tokens=8,temperature=temperature,seed=11)
                    self.assertEqual(efficient,original)

