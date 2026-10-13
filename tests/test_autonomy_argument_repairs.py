"""Explicit test fixtures exercise READ correction; no learned planning claim."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ia_propia.brain.autonomy import AutonomyController, AutonomyLimits, AutonomyTool
from ia_propia.brain.autonomy_model import _observations
from ia_propia.brain.protocol import BrainError


def action(filename, tool='inspect_file'):
    return {'action': 'tool', 'tool': tool, 'arguments': {'filename': filename},
            'reason': 'Explicit controller test fixture'}


class FixtureProposer:
    origin = 'test_fixture'
    proposer_id = 'read-argument-repair-fixture'

    def __init__(self, replies):
        self.replies = iter(replies)
        self.contexts = []

    def __call__(self, context):
        self.contexts.append(context)
        reply = next(self.replies)
        return json.dumps(reply) if type(reply) is dict else reply


class ArgumentRepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'uno.txt'
        self.source.write_bytes(b'Actual source bytes remain unchanged.\n')
        self.original_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.effects = []

    def validator(self, arguments):
        return arguments == {'filename': 'uno.txt'}

    def read_file(self, arguments):
        self.effects.append(dict(arguments))
        content = self.source.read_bytes()
        return {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}

    def verified(self, arguments, result):
        return {'verified': arguments == {'filename': 'uno.txt'} and result['sha256'] == self.original_sha}

    def completed(self, state):
        return any(event['kind'] == 'tool_result' and event.get('verification', {}).get('verified') is True
                   and event['arguments'] == {'filename': 'uno.txt'}
                   and event['result']['sha256'] == hashlib.sha256(self.source.read_bytes()).hexdigest()
                   for event in state['journal'])

    def tool(self, effect='read', validator=None, policy_id='original-whitelist'):
        return AutonomyTool('inspect_file', self.read_file, validator or self.validator,
                            effect=effect, verifier=self.verified, policy_id=policy_id)

    def controller(self, replies, *, tools=None, limits=None, state='state'):
        proposer = FixtureProposer(replies)
        controller = AutonomyController(self.root / state, proposer, tools or [self.tool()],
                                        limits=limits or AutonomyLimits(max_argument_repairs=1),
                                        completion_verifier=self.completed)
        return controller, proposer

    def test_read_correction_uses_new_proposal_real_bytes_and_preserves_raw_rejection(self):
        bad, good = action('k.txt'), action('uno.txt')
        controller, proposer = self.controller([bad, good])
        task_id = controller.start('Read only the explicit source', task_inputs={'files': ['uno.txt']})['task_id']
        denied = controller.step(task_id)
        self.assertEqual((denied['status'], denied['failures'], denied['argument_repairs']), ('ready', 1, 1))
        self.assertIsNone(denied['current'])
        self.assertEqual(self.effects, [])
        self.assertFalse(any(event['kind'] == 'effect_started' for event in denied['journal']))
        kinds = [event['kind'] for event in denied['journal']]
        self.assertLess(kinds.index('failure'), kinds.index('argument_repair_allowed'))
        result = controller.step(task_id)
        self.assertEqual((result['status'], result['steps'], result['failures']), ('completed', 2, 1))
        self.assertEqual(self.effects, [{'filename': 'uno.txt'}])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.original_sha)
        raw_events = [event['raw'] for event in result['journal'] if event['kind'] == 'proposal']
        self.assertEqual(raw_events, [json.dumps(bad), json.dumps(good)])
        observed = _observations(proposer.contexts[1]['task']['journal'])
        rejected = next(item['rejected_action'] for item in observed if 'rejected_action' in item)
        self.assertEqual(rejected, {'tool': 'inspect_file', 'arguments': {'filename': 'k.txt'}})
        self.assertEqual(proposer.contexts[1]['task']['task_inputs'], {'files': ['uno.txt']})

    def test_default_zero_old_tasks_restart_limit_and_policy_binding_do_not_escalate(self):
        default, _ = self.controller([action('k.txt')], limits=AutonomyLimits(), state='default')
        task_id = default.start('Default policy remains terminal')['task_id']
        # Simulate an old persisted task without the new optional fields.
        def old_state(state):
            state['limits'].pop('max_argument_repairs')
            state.pop('argument_repairs')
        default._mutate(task_id, old_state)
        self.assertEqual(default.step(task_id)['status'], 'blocked_policy')
        original, _ = self.controller([action('k.txt')], state='persisted')
        task_id = original.start('Persistent finite repair budget')['task_id']
        original.step(task_id)
        reopened, proposer = self.controller([action('missing.txt')], limits=AutonomyLimits(max_argument_repairs=64), state='persisted')
        denied = reopened.step(task_id)
        self.assertEqual((denied['status'], denied['argument_repairs'], denied['failures']), ('blocked_policy', 1, 2))
        self.assertEqual(denied['limits']['max_argument_repairs'], 1)
        reopened.step(task_id)
        self.assertEqual(len(proposer.contexts), 1)
        original, _ = self.controller([action('k.txt')], state='binding')
        task_id = original.start('No changed whitelist after repair')['task_id']
        original.step(task_id)
        changed, proposer = self.controller([action('uno.txt')], tools=[self.tool(policy_id='different-whitelist')], state='binding')
        self.assertEqual(changed.step(task_id)['status'], 'blocked_policy')
        self.assertEqual(proposer.contexts, [])
        self.assertEqual(self.effects, [])

    def test_invalid_json_and_unknown_tool_cannot_consume_argument_repairs(self):
        cases = [('```json\n{}\n```', 'blocked_model'), (action('uno.txt', tool='unregistered'), 'blocked_policy')]
        for index, (proposal, status) in enumerate(cases):
            with self.subTest(status=status):
                controller, proposer = self.controller([proposal], state='invalid-' + str(index))
                task_id = controller.start('Fail closed outside READ argument correction')['task_id']
                result = controller.step(task_id)
                self.assertEqual((result['status'], result['argument_repairs']), (status, 0))
                self.assertFalse(any(event['kind'] == 'argument_repair_allowed' for event in result['journal']))
                controller.step(task_id)
                self.assertEqual(len(proposer.contexts), 1)
        self.assertEqual(self.effects, [])

    def test_write_and_sensitive_effects_remain_terminal_without_approval_or_handler(self):
        for effect in ('write', 'destructive', 'publication', 'expense'):
            with self.subTest(effect=effect):
                controller, _ = self.controller([action('k.txt')], tools=[self.tool(effect=effect)], state=effect)
                result = controller.step(controller.start('No effect correction escalation')['task_id'])
                self.assertEqual((result['status'], result['argument_repairs']), ('blocked_policy', 0))
                self.assertIsNone(result['pending_approval'])
                self.assertFalse(any(event['kind'] in ('effect_started', 'approval_requested', 'argument_repair_allowed')
                                     for event in result['journal']))
        self.assertEqual(self.effects, [])

    def test_cancel_during_validator_and_after_allowed_repair_prevents_new_effect(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def denied_after_wait(arguments):
            entered.set()
            release.wait(2)
            return False
        controller, _ = self.controller([action('k.txt')], tools=[self.tool(validator=denied_after_wait)])
        task_id = controller.start('Cancellation during READ rejection')['task_id']
        results = []
        thread = threading.Thread(target=lambda: results.append(controller.step(task_id)))
        thread.start()
        self.addCleanup(lambda: thread.join(2))
        self.assertTrue(entered.wait(1))
        controller.cancel(task_id)
        release.set()
        thread.join(2)
        self.assertEqual((results[0]['status'], results[0]['argument_repairs']), ('cancelled', 0))
        repaired, proposer = self.controller([action('k.txt'), action('uno.txt')], state='cancel-ready')
        task_id = repaired.start('Cancellation before correction proposal')['task_id']
        repaired.step(task_id)
        repaired.cancel(task_id)
        self.assertEqual(repaired.step(task_id)['status'], 'cancelled')
        self.assertEqual(len(proposer.contexts), 1)
        self.assertEqual(self.effects, [])

    def test_repair_cannot_extend_failure_step_time_or_host_integer_limits(self):
        for value in (-1, 65, True, 1.0):
            with self.assertRaises(BrainError):
                AutonomyLimits(max_argument_repairs=value)
        limits = [('failures', AutonomyLimits(max_argument_repairs=1, max_failures=1)),
                  ('steps', AutonomyLimits(max_argument_repairs=1, max_steps=1)),
                  ('seconds', AutonomyLimits(max_argument_repairs=1, max_seconds=2))]
        for name, value in limits:
            with self.subTest(resource=name):
                clock = [1000.0]
                def validator(arguments):
                    if name == 'seconds':
                        clock[0] += 3
                    return False
                with patch('ia_propia.brain.autonomy.time.time', side_effect=lambda: clock[0]):
                    controller, _ = self.controller([action('k.txt')], limits=value,
                                                   tools=[self.tool(validator=validator)], state='budget-' + name)
                    result = controller.step(controller.start('Budget applies after rejection')['task_id'])
                self.assertEqual((result['status'], result['argument_repairs'], result['failures']), ('limit_reached', 0, 1))
                self.assertFalse(any(event['kind'] == 'effect_started' for event in result['journal']))
        self.assertEqual(self.effects, [])


if __name__ == '__main__':
    unittest.main()
