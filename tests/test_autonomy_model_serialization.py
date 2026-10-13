"""Declared adapter engine fixture: lossless observations, no model weights."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from types import SimpleNamespace
import unittest

from ia_propia.brain.autonomy_model import ACTION_INSTRUCTION, ModelActionProposer
from ia_propia.brain.protocol import BrainError
from ia_propia.conversation.contract import MAX_MESSAGE_BYTES, SPECIAL_TOKENS


HAS_TOKENIZERS = importlib.util.find_spec('tokenizers') is not None


class DeclaredEngineFixture:
    model_id = 'adapter-wire-fixture-no-own-weights-loaded'

    def __init__(self):
        from ia_propia.conversation.tokenizer import fit_tokenizer
        self._tokenizer = fit_tokenizer([{'messages': [
            {'role': 'user', 'content': 'Fixture explícito para comprobar el contrato JSON.'},
            {'role': 'assistant', 'content': 'No es una prueba de planificación aprendida.'}]}], vocab_size=384)
        self._model = SimpleNamespace(config={'context_length': 4096})
        self.calls = []
        self.raw_reply = '```json\n{"action":"stop","reason":"declared fixture"}\n```'

    def prompt_usage(self, messages):
        from ia_propia.conversation.tokenizer import encode_prompt
        capacity = self._model.config['context_length']
        ids, discarded = encode_prompt(messages, self._tokenizer, capacity)
        return {'tokens': len(ids) + discarded, 'capacity': capacity,
                'discarded': discarded}

    def reply(self, **request):
        from ia_propia.conversation.runtime import validate_reply_request
        validate_reply_request(request)
        self.calls.append(deepcopy(request))
        return {'text': self.raw_reply, 'modelId': self.model_id}


def serialized(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


@unittest.skipUnless(HAS_TOKENIZERS, 'conversation tokenizer is optional')
class ModelSerializationTests(unittest.TestCase):
    def setUp(self):
        self.engine = DeclaredEngineFixture()
        self.proposer = ModelActionProposer(SimpleNamespace(conversation_engine=self.engine))

    def context(self, text):
        result = {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                  'nested': {'literal': text}, 'sourceBytes': len(text.encode())}
        task = {'goal': 'Analiza el archivo original y conserva sus bytes.',
                'task_inputs': {'files': ['contract.py'], 'note': text},
                'journal': [{'kind': 'effect_started', 'tool': 'read_file', 'arguments': {'filename': 'contract.py'}},
                            {'kind': 'handler_returned', 'tool': 'read_file', 'result': result},
                            {'kind': 'tool_result', 'tool': 'read_file', 'result': result,
                             'verification': {'verified': True}}]}
        tools = [{'name': 'read_file',
                  'schema': {'type': 'object', 'properties': {'filename': {'enum': ['contract.py']}},
                             'const': {'observedMarker': text}}, 'effect': 'read'}]
        expected = {'task': {key: deepcopy(task[key]) for key in ('goal', 'task_inputs')},
                    'tools': deepcopy(tools), 'observations': [{'tool': 'read_file',
                    'arguments': {'filename': 'contract.py'}, 'result': deepcopy(result), 'verified': True}]}
        return {'task': task, 'tools': tools}, expected

    def test_all_reserved_tokens_and_literal_backslash_unicode_round_trip_without_redaction(self):
        text = ' '.join(SPECIAL_TOKENS) + r' literal \u003c \u003e \\ruta' + ' español café 🧠'
        context, expected = self.context(text)
        original = deepcopy(context)
        self.assertEqual(self.proposer(context), self.engine.raw_reply)
        wire = self.engine.calls[0]['messages'][-1]['content']
        self.assertEqual(json.loads(wire), expected)
        self.assertEqual(context, original)
        for token in SPECIAL_TOKENS:
            self.assertNotIn(token, wire)
        self.assertIn(r'\u003c', wire)
        self.assertIn(r'\\u003c', wire)  # An existing literal escape remains literal data.
        self.assertEqual(self.proposer.last_metadata['contextSha256'], hashlib.sha256(wire.encode()).hexdigest())
        self.assertEqual(self.proposer.last_metadata['request'], self.engine.calls[0])
        self.assertEqual(self.proposer.last_metadata['rawReply'], self.engine.raw_reply)

    def test_wire_content_cannot_embed_role_ids_and_delimiter_free_frames_are_byte_exact(self):
        from ia_propia.conversation.tokenizer import encode_prompt
        context, _ = self.context(''.join(SPECIAL_TOKENS))
        self.proposer(context)
        request = self.engine.calls[0]
        wire = request['messages'][-1]['content']
        self.assertEqual(set(self.engine._tokenizer.encode(wire, add_special_tokens=False).ids) & set(range(6)), set())
        encoded, discarded = encode_prompt(request['messages'], self.engine._tokenizer, 4096)
        self.assertEqual(discarded, 0)
        self.assertEqual([token for token in encoded if token < 6], [1, 3, 2, 4, 2, 5])
        self.assertEqual(request['messages'][0], {'role': 'system', 'content': ACTION_INSTRUCTION})
        for text in ('JSON anterior con café y literal \\u003c.', 'Comparación anterior 3 > 2 sin apertura.'):
            with self.subTest(text=text):
                context, expected = self.context(text)
                self.proposer(context)
                self.assertEqual(self.engine.calls[-1]['messages'][-1]['content'], serialized(expected))

    def test_size_and_token_limits_apply_to_expanded_wire_before_generation(self):
        context = {'task': {'goal': '<' * 3000}, 'tools': []}
        expected = {'task': {'goal': '<' * 3000}, 'tools': [], 'observations': []}
        self.assertLess(len(serialized(expected).encode()), MAX_MESSAGE_BYTES)
        wire = serialized(expected).replace('<', r'\u003c').replace('>', r'\u003e')
        self.assertGreater(len(wire.encode()), MAX_MESSAGE_BYTES)
        with self.assertRaises(BrainError) as rejected:
            self.proposer(context)
        self.assertEqual(rejected.exception.code, 'context_too_large')
        self.assertEqual(self.engine.calls, [])
        metadata = self.proposer.last_metadata
        self.assertEqual(metadata['contextRejected'], 'conversation_message_limit')
        self.assertEqual(metadata['contextBytes'], len(wire.encode()))
        self.assertEqual(metadata['contextSha256'], hashlib.sha256(wire.encode()).hexdigest())
        self.engine._model.config['context_length'] = 32
        with self.assertRaises(BrainError) as rejected:
            self.proposer({'task': {'goal': '<system> original data'}, 'tools': []})
        self.assertEqual(rejected.exception.code, 'context_too_large')
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.proposer.last_metadata['contextRejected'], 'model_token_limit')
        self.assertGreater(self.proposer.last_metadata['discardedPromptTokens'], 0)


if __name__ == '__main__':
    unittest.main()
