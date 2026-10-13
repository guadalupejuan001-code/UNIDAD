"""Real temporary IO tests of controller contracts, not learned planning skill."""

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from ia_propia.brain.autonomy import AutonomyController, AutonomyLimits, AutonomyTool
from ia_propia.brain.protocol import BrainError


class FixtureProposer:
    origin = "test_fixture"
    proposer_id = "controller-contract-fixture"

    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.contexts = []
        self.last_metadata = {"fixture": True}

    def __call__(self, context):
        self.contexts.append(context)
        value = next(self.proposals)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if type(value) is dict else value


def action(tool, arguments=None):
    return {"action": "tool", "tool": tool, "arguments": arguments or {}, "reason": "Fixture exercises controller"}


def empty_arguments(arguments):
    return arguments == {}


class AutonomyControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.destination = self.directory / "report.txt"

    def controller(self, proposer, tools=(), **kwargs):
        return AutonomyController(self.directory / "state", proposer, tools, **kwargs)

    def save_validator(self, arguments):
        return set(arguments) == {"text"} and type(arguments["text"]) is str and len(arguments["text"]) < 1000

    def save(self, arguments):
        self.destination.write_text(arguments["text"], encoding="utf-8")
        return {"sha256": hashlib.sha256(self.destination.read_bytes()).hexdigest()}

    def saved(self, arguments, result):
        verified = self.destination.is_file() and self.destination.read_text(encoding="utf-8") == arguments["text"]
        return {"verified": verified, "result": result}

    def complete(self, snapshot):
        return self.destination.is_file() and self.destination.read_text(encoding="utf-8") == "verified report"

    def save_tool(self, *, handler=None, verifier=True, effect="write"):
        return AutonomyTool("save", handler or self.save, self.save_validator, effect=effect,
                            verifier=self.saved if verifier else None)

    def test_real_effect_observation_and_host_completion_persist(self):
        proposer = FixtureProposer([action("save", {"text": "verified report"})])
        controller = self.controller(proposer, [self.save_tool()], completion_verifier=self.complete)
        task = controller.start("Write a report", task_inputs={"note": "explicit host input"})
        result = controller.run(task["task_id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.destination.read_text(), "verified report")
        observations = [e for e in result["journal"] if e["kind"] == "tool_result"]
        self.assertEqual(observations[0]["arguments"], {"text": "verified report"})
        self.assertTrue(observations[0]["verification"]["verified"])
        self.assertEqual(result["source"], "test_fixture")
        reopened = self.controller(FixtureProposer([]), [self.save_tool()], completion_verifier=self.complete)
        self.assertEqual(reopened.status(task["task_id"]), result)
        self.assertEqual(len(proposer.contexts), 1)

    def test_finish_model_claim_does_not_complete(self):
        controller = self.controller(FixtureProposer([{"action": "finish", "reason": "I say finished"}]))
        result = controller.step(controller.start("Create report")["task_id"])
        self.assertEqual(result["status"], "blocked_verification")
        self.assertFalse(self.destination.exists())

    def test_closed_json_protocol_blocks_invalid_raw_and_preserves_evidence(self):
        proposals = ['```json\n{"action":"finish","reason":"done"}\n```',
                     '{"action":"finish","action":"stop","reason":"done"}',
                     '{"action":"tool","tool":"save","arguments":{"x":NaN},"reason":"r"}',
                     '{"action":"finish","reason":"done","approved":true}',
                     '{"action":"tool","tool":"save","arguments":[],"reason":"r"}',
                     '{"action":"finish","reason":"r"} trailing']
        for raw in proposals:
            with self.subTest(raw=raw):
                controller = self.controller(FixtureProposer([raw]), [self.save_tool()])
                result = controller.step(controller.start("Invalid JSON has no effects")["task_id"])
                self.assertEqual(result["status"], "blocked_model")
                self.assertEqual(result["steps"], 1)
                event = next(e for e in result["journal"] if e["kind"] == "proposal")
                self.assertEqual(event["raw"], raw)
                self.assertFalse(event["valid"])
                self.assertFalse(self.destination.exists())

    def test_unregistered_tool_and_rejected_arguments_have_no_effect(self):
        for proposal in [action("unavailable"), action("save", {"path": "/etc/passwd", "text": "unsafe"})]:
            controller = self.controller(FixtureProposer([proposal]), [self.save_tool()])
            result = controller.step(controller.start("Host registry only")["task_id"])
            self.assertEqual(result["status"], "blocked_policy")
            self.assertFalse(self.destination.exists())

    def test_effect_category_and_proposer_origin_are_explicit_host_configuration(self):
        with self.assertRaises(BrainError):
            AutonomyTool("opaque", lambda args: args, empty_arguments)
        with self.assertRaises(BrainError):
            self.controller(lambda context: json.dumps(action("save")))
        with self.assertRaises(BrainError):
            AutonomyTool("autonomy.approve", lambda args: args, empty_arguments, effect="read")

    def test_approval_is_separate_nonce_bound_and_consumed_once(self):
        effects = []
        tool = AutonomyTool("publish", lambda args: effects.append(args) or {"published": True},
                            lambda args: True, effect="publication")
        controller = self.controller(FixtureProposer([action("publish", {"approved": True, "operator_authorized": True})]), [tool])
        task_id = controller.start("Publication needs operator")["task_id"]
        result = controller.step(task_id)
        self.assertEqual(result["status"], "needs_approval")
        nonce = result["pending_approval"]["approval_id"]
        self.assertEqual(effects, [])
        for passed in [{}, {"operator_authorized": True}]:
            with self.assertRaises(BrainError):
                controller.approve(task_id, "wrong" if passed else nonce, True, **passed)
        controller.approve(task_id, nonce, True, operator_authorized=True)
        result = controller.step(task_id)
        self.assertEqual(len(effects), 1)
        self.assertIsNone(result["pending_approval"])
        with self.assertRaises(BrainError):
            controller.approve(task_id, nonce, True, operator_authorized=True)

    def test_operator_denial_and_policy_change_cannot_execute_pending_approval(self):
        tool = AutonomyTool("delete", lambda args: self.destination.unlink(), empty_arguments, effect="destructive")
        controller = self.controller(FixtureProposer([action("delete")]), [tool])
        self.destination.write_text("keep original")
        task_id = controller.start("Await approval")["task_id"]
        result = controller.step(task_id)
        nonce = result["pending_approval"]["approval_id"]
        replacement = AutonomyTool("delete", tool.handler, tool.validator, effect="expense")
        changed = self.controller(FixtureProposer([]), [replacement])
        with self.assertRaises(BrainError):
            changed.approve(task_id, nonce, True, operator_authorized=True)
        result = controller.approve(task_id, nonce, False, operator_authorized=True)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(controller.step(task_id)["status"], "stopped")
        self.assertEqual(self.destination.read_text(), "keep original")

    def launch(self, operation):
        results = []
        def run():
            try:
                results.append(operation())
            except BaseException as error:
                results.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        self.addCleanup(lambda: thread.join(2))
        return thread, results

    def test_cancel_during_model_call_is_nonblocking_and_prevents_effect(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        proposer = FixtureProposer([])
        def blocked(context):
            entered.set()
            release.wait(2)
            return json.dumps(action("save", {"text": "never"}))
        proposer.__class__ = type("BlockingFixture", (FixtureProposer,), {"__call__": lambda self, c: blocked(c)})
        controller = self.controller(proposer, [self.save_tool()])
        task_id = controller.start("Cancel while model is blocked")["task_id"]
        thread, results = self.launch(lambda: controller.step(task_id))
        self.assertTrue(entered.wait(1))
        self.assertEqual(controller.cancel(task_id)["status"], "cancelled")
        release.set()
        thread.join(2)
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertFalse(self.destination.exists())
        self.assertTrue(any(e["kind"] == "proposal" for e in results[0]["journal"]))

    def test_cancel_while_validator_blocked_prevents_tool(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def validator(arguments):
            entered.set()
            release.wait(2)
            return self.save_validator(arguments)
        tool = AutonomyTool("save", self.save, validator, effect="write")
        controller = self.controller(FixtureProposer([action("save", {"text": "never"})]), [tool])
        task_id = controller.start("Cancellation during permission validation")["task_id"]
        thread, results = self.launch(lambda: controller.step(task_id))
        self.assertTrue(entered.wait(1))
        controller.cancel(task_id)
        release.set()
        thread.join(2)
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertFalse(self.destination.exists())

    def test_concurrent_controller_cannot_repeat_inflight_step(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def handler(arguments):
            entered.set()
            release.wait(2)
            return self.save(arguments)
        tool = self.save_tool(handler=handler)
        controller = self.controller(FixtureProposer([action("save", {"text": "once"})]), [tool])
        other = self.controller(FixtureProposer([]), [tool])
        task_id = controller.start("Single effect")["task_id"]
        thread, results = self.launch(lambda: controller.step(task_id))
        self.assertTrue(entered.wait(1))
        with self.assertRaises(BrainError) as caught:
            other.step(task_id)
        self.assertEqual(caught.exception.code, "task_busy")
        controller.cancel(task_id)
        release.set()
        thread.join(2)
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertTrue(any(e["kind"] == "handler_returned" for e in results[0]["journal"]))
        self.assertEqual(self.destination.read_text(), "once")

    def test_interrupted_write_recovers_via_artifact_without_repeating_handler(self):
        calls = []
        def interrupted(arguments):
            calls.append(1)
            self.save(arguments)
            raise KeyboardInterrupt("simulate interrupted process after external effect")
        tool = self.save_tool(handler=interrupted)
        controller = self.controller(FixtureProposer([action("save", {"text": "verified report"})]), [tool],
                                     completion_verifier=self.complete)
        task_id = controller.start("Recover recorded interruption")["task_id"]
        with self.assertRaises(KeyboardInterrupt):
            controller.step(task_id)
        self.assertEqual(controller.status(task_id)["status"], "executing")
        reopened = self.controller(FixtureProposer([]), [tool], completion_verifier=self.complete)
        result = reopened.resume(task_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls, [1])
        self.assertTrue(any(e["kind"] == "recovery_verification" for e in result["journal"]))

    def test_uncertain_write_does_not_retry_and_has_no_rollback_claim(self):
        calls = []
        def failure(arguments):
            calls.append(1)
            self.save(arguments)
            raise RuntimeError("Connection lost after write")
        tool = self.save_tool(handler=failure, verifier=False)
        controller = self.controller(FixtureProposer([action("save", {"text": "keep effect"})]), [tool])
        task_id = controller.start("Unknown effect stays recorded")["task_id"]
        self.assertEqual(controller.step(task_id)["status"], "blocked_uncertain")
        self.assertEqual(controller.resume(task_id)["status"], "blocked_uncertain")
        self.assertEqual(controller.step(task_id)["status"], "blocked_uncertain")
        self.assertEqual(calls, [1])
        self.assertEqual(self.destination.read_text(), "keep effect")

    def test_same_nonidempotent_write_cannot_be_proposed_twice(self):
        calls = []
        def handler(arguments):
            calls.append(1)
            return self.save(arguments)
        proposal = action("save", {"text": "only once"})
        controller = self.controller(FixtureProposer([proposal, proposal]), [self.save_tool(handler=handler)])
        result = controller.run(controller.start("No duplicate writes")["task_id"])
        self.assertEqual(result["status"], "blocked_policy")
        self.assertEqual(calls, [1])

    def test_read_failures_stop_at_budget_and_proposer_failures_are_recorded(self):
        attempts = []
        def failing(arguments):
            attempts.append(1)
            raise RuntimeError("temporary read failure")
        tool = AutonomyTool("read", failing, empty_arguments, effect="read")
        controller = self.controller(FixtureProposer([action("read")] * 10), [tool],
                                     limits=AutonomyLimits(max_failures=2))
        result = controller.run(controller.start("Finite retry budget")["task_id"])
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(result["failures"], 2)
        self.assertEqual(len(attempts), 2)
        other = self.controller(FixtureProposer([RuntimeError("model unavailable")]), [],
                                limits=AutonomyLimits(max_failures=1))
        result = other.run(other.start("Record failed model")["task_id"])
        self.assertEqual(result["status"], "limit_reached")
        self.assertTrue(any(e["kind"] == "proposer_evidence" for e in result["journal"]))

    def test_step_state_and_time_budgets_stop_before_new_effect(self):
        calls = []
        tool = AutonomyTool("read", lambda args: calls.append(1) or {}, empty_arguments, effect="read")
        controller = self.controller(FixtureProposer([action("read")] * 4), [tool], limits=AutonomyLimits(max_steps=1))
        result = controller.run(controller.start("One bounded step")["task_id"])
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(calls, [1])
        controller = self.controller(FixtureProposer([action("read")]), [tool], limits=AutonomyLimits(max_state_bytes=16384))
        result = controller.run(controller.start("State space reserve")["task_id"])
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(result["steps"], 0)
        controller = self.controller(FixtureProposer([action("read")]), [tool], limits=AutonomyLimits(max_seconds=0.001))
        task_id = controller.start("Elapsed budget")["task_id"]
        time.sleep(0.005)
        self.assertEqual(controller.step(task_id)["status"], "limit_reached")
        self.assertEqual(calls, [1])

    def test_false_tool_verification_blocks_completion(self):
        tool = self.save_tool()
        controller = self.controller(FixtureProposer([action("save", {"text": "written"})]),
                                     [AutonomyTool("save", tool.handler, tool.validator, effect="write",
                                                   verifier=lambda args, result: {"verified": False})],
                                     completion_verifier=lambda state: True)
        result = controller.step(controller.start("Verifier must confirm effect")["task_id"])
        self.assertEqual(result["status"], "blocked_verification")
        self.assertTrue(self.destination.exists())

    def test_task_policy_change_blocks_even_ordinary_tool_without_effect(self):
        tool = self.save_tool()
        first = self.controller(FixtureProposer([]), [tool])
        task_id = first.start("Registry stays bound")["task_id"]
        changed = self.controller(FixtureProposer([action("save", {"text": "unsafe"})]),
                                  [self.save_tool(effect="read")])
        self.assertEqual(changed.step(task_id)["status"], "blocked_policy")
        self.assertFalse(self.destination.exists())

    def test_complex_real_result_keeps_receipt_when_state_node_budget_exhausts(self):
        def handler(arguments):
            self.save(arguments)
            return {"items": list(range(19980))}
        controller = self.controller(FixtureProposer([action("save", {"text": "real effect retained"})]),
                                     [self.save_tool(handler=handler, verifier=False)],
                                     limits=AutonomyLimits(max_result_bytes=262144))
        result = controller.step(controller.start("Bound large observations")["task_id"])
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(self.destination.read_text(), "real effect retained")
        receipt = next(e for e in result["journal"] if e["kind"] == "state_limit")
        self.assertTrue(receipt["outcome_may_exist"])
        self.assertFalse(receipt["effects_rolled_back"])
        self.assertEqual(receipt["omitted_events"][0]["kind"], "handler_returned")
        self.assertEqual(len(receipt["omitted_events"][0]["sha256"]), 64)

    def test_host_resource_binding_identity_is_private_and_prevents_rebind(self):
        original = self.save_tool()
        tool = AutonomyTool("save", original.handler, original.validator, effect="write", policy_id="resource-A")
        first = self.controller(FixtureProposer([]), [tool])
        task_id = first.start("Keep host resource identity")["task_id"]
        replacement = AutonomyTool("save", original.handler, original.validator, effect="write", policy_id="resource-B")
        changed = self.controller(FixtureProposer([action("save", {"text": "never"})]), [replacement])
        self.assertNotIn("policy_id", tool.metadata())
        self.assertEqual(changed.step(task_id)["status"], "blocked_policy")
        self.assertFalse(self.destination.exists())

    def test_completion_verifier_cannot_change_after_task_creation(self):
        tool = self.save_tool()
        original = self.controller(FixtureProposer([]), [tool], completion_verifier=self.complete)
        task_id = original.start("Keep original completion policy")["task_id"]
        changed = self.controller(FixtureProposer([action("save", {"text": "never"})]), [tool],
                                  completion_verifier=lambda snapshot: True)
        self.assertEqual(changed.resume(task_id)["status"], "blocked_policy")
        self.assertFalse(self.destination.exists())

    def test_elapsed_handler_budget_prevents_verifier_and_keeps_actual_result(self):
        calls = []
        def handler(arguments):
            calls.append("handler")
            time.sleep(0.02)
            return self.save(arguments)
        def verifier(arguments, result):
            calls.append("verifier")
            return self.saved(arguments, result)
        tool = AutonomyTool("save", handler, self.save_validator, effect="write", verifier=verifier)
        controller = self.controller(FixtureProposer([action("save", {"text": "effect remains"})]), [tool],
                                     limits=AutonomyLimits(max_seconds=0.01))
        result = controller.step(controller.start("Stop new callbacks after deadline")["task_id"])
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(calls, ["handler"])
        self.assertEqual(self.destination.read_text(), "effect remains")
        event = next(e for e in result["journal"] if e["kind"] == "handler_returned")
        self.assertEqual(event["result"]["sha256"], hashlib.sha256(self.destination.read_bytes()).hexdigest())

    def test_host_input_policy_identity_is_bound_without_exposing_descriptor(self):
        first = self.controller(FixtureProposer([]), [self.save_tool()], policy_id="input-validator-wide-v1")
        task_id = first.start("Task keeps original input policy", task_inputs={"resource": "A"})["task_id"]
        changed = self.controller(FixtureProposer([action("save", {"text": "never"})]), [self.save_tool()],
                                  policy_id="input-validator-narrow-v2")
        result = changed.step(task_id)
        self.assertEqual(result["status"], "blocked_policy")
        self.assertNotIn("input-validator-wide-v1", json.dumps(result))
        self.assertFalse(self.destination.exists())

    def test_host_observation_restore_failure_retries_on_explicit_resume(self):
        observation = self.directory / "read-evidence.txt"
        class HostAdapter:
            def restore_task_observations(self, snapshot):
                return {"sha256": hashlib.sha256(observation.read_bytes()).hexdigest()}
            def completion_verifier(self, snapshot):
                return False
        adapter = HostAdapter()
        proposer = FixtureProposer([action("save", {"text": "restored"})])
        controller = self.controller(proposer, [self.save_tool()], completion_verifier=adapter.completion_verifier)
        task_id = controller.start("Recover read cache before effect")["task_id"]
        result = controller.step(task_id)
        self.assertEqual(result["status"], "blocked_verification")
        self.assertEqual(proposer.contexts, [])
        self.assertFalse(self.destination.exists())
        observation.write_text("trusted host bytes")
        self.assertEqual(controller.resume(task_id)["status"], "ready")
        result = controller.step(task_id)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.destination.read_text(), "restored")
        event = next(e for e in result["journal"] if e["kind"] == "observations_restored")
        self.assertEqual(event["result"]["sha256"], hashlib.sha256(observation.read_bytes()).hexdigest())

    def test_cancel_during_host_restore_stops_before_proposer_and_effect(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        class HostAdapter:
            def restore_task_observations(self, snapshot):
                entered.set()
                release.wait(2)
                return {"restored_files": []}
            def completion_verifier(self, snapshot):
                return False
        adapter = HostAdapter()
        proposer = FixtureProposer([action("save", {"text": "never"})])
        controller = self.controller(proposer, [self.save_tool()], completion_verifier=adapter.completion_verifier)
        task_id = controller.start("Cancel during host-only restore")["task_id"]
        thread, results = self.launch(lambda: controller.step(task_id))
        self.assertTrue(entered.wait(1))
        self.assertEqual(controller.cancel(task_id)["status"], "cancelled")
        release.set()
        thread.join(2)
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertEqual(proposer.contexts, [])
        self.assertFalse(self.destination.exists())

    def test_model_context_capacity_failure_is_terminal_and_preserves_evidence(self):
        proposer = FixtureProposer([BrainError("context_too_large", "Prompt exceeds local model context")])
        proposer.last_metadata = {"promptTokens": 732, "contextTokens": 512, "contextRejected": True}
        controller = self.controller(proposer, [self.save_tool()])
        task_id = controller.start("Record local model context limit")["task_id"]
        result = controller.run(task_id)
        self.assertEqual(result["status"], "blocked_context")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["failures"], 1)
        evidence = next(e for e in result["journal"] if e["kind"] == "proposer_evidence")
        self.assertEqual(evidence["metadata"], proposer.last_metadata)
        self.assertEqual(controller.resume(task_id)["status"], "blocked_context")
        self.assertEqual(len(proposer.contexts), 1)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
