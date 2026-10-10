"""Portable service contracts; successful explicit work is not instruction skill."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ia_propia.brain import Brain, BrainError, Tool
from ia_propia.brain import models
from ia_propia.brain.protocol import MAX_REQUEST_BYTES, decode_json


class BrainCoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.brain = Brain(self.directory / "private")

    def request(self, body, operation, input_data, **extra):
        return body.request({"schemaVersion": 1, "requestId": "core-test", "operation": operation,
                             "input": input_data, **extra})

    def ok(self, response):
        self.assertEqual(response["status"], "ok", response)
        self.assertFalse(response["workerCapabilityValidated"])
        return response["result"]

    def step(self, step_id, operation, input_data):
        return {"id": step_id, "operation": operation, "input": input_data}

    def test_models_load_once_share_identity_and_keep_requests_independent(self):
        with patch.object(models, "_packaged_models", wraps=models._packaged_models) as loader:
            brain = Brain(self.directory / "shared-brain")
            first, second = brain.connect("python"), brain.connect("website")
            self.assertIs(first.models, second.models)
            self.assertIs(first.models, brain.models)
            values = {"prompt": "La red ", "length": 30, "seed": 7, "temperature": 0.8}
            expected = self.ok(self.request(first, "language.generate", values, context={"note": "first body"}))
            actual = self.ok(self.request(second, "language.generate", values, context={"note": "second body"}))
            self.assertEqual(expected, actual)
            self.assertEqual(loader.call_count, 1)
        capabilities = brain.capabilities()
        self.assertFalse(capabilities["naturalLanguagePlanner"])
        self.assertFalse(capabilities["automaticLearning"])
        self.assertFalse(capabilities["instructionFollowingValidated"])

    def test_tools_are_body_bound_and_cannot_mutate_the_callers_request(self):
        seen = []
        def handler(arguments):
            seen.append(copy.deepcopy(arguments))
            arguments["changed"] = True
            return arguments
        first = self.brain.connect("first", [Tool("echo", handler)])
        second = self.brain.connect("second")
        payload = {"schemaVersion": 1, "operation": "tool.call", "input": {"name": "echo", "arguments": {"value": 7}}}
        before = copy.deepcopy(payload)
        self.assertTrue(self.ok(first.request(payload))["changed"])
        self.assertEqual(payload, before)
        denied = second.request(payload)
        self.assertEqual(denied["error"]["code"], "tool_denied")
        self.assertEqual(len(seen), 1)
        self.assertEqual(second.capabilities()["tools"], [])

    def test_memory_is_private_per_body_and_shared_only_by_host_configuration(self):
        first, second = self.brain.connect("first"), self.brain.connect("second")
        task = self.ok(self.request(first, "memory.task.add", {"goal": "Synthetic private first-body task"}))
        other = self.request(second, "memory.task.show", {"task_id": task["id"]})
        self.assertEqual(other["status"], "error")
        listed = self.ok(self.request(second, "memory.task.list", {}, context={"memory_scope": "first", "workspace": str(self.directory)}))
        self.assertEqual(listed["tasks"], [])
        injection = self.request(second, "memory.task.show", {"task_id": task["id"], "path": str(self.directory)})
        self.assertEqual(injection["error"]["code"], "invalid_input")
        selected_body = second.request({"schemaVersion": 1, "operation": "memory.task.list", "input": {}, "bodyId": "first"})
        self.assertEqual(selected_body["status"], "error")
        self.assertEqual(selected_body["bodyId"], "second")
        shared_a = self.brain.connect("shared-a", memory_scope="explicit-project")
        shared_b = self.brain.connect("shared-b", memory_scope="explicit-project")
        shared_task = self.ok(self.request(shared_a, "memory.task.add", {"goal": "Explicitly shared synthetic task"}))
        self.assertEqual(self.ok(self.request(shared_b, "memory.task.show", {"task_id": shared_task["id"]}))["goal"], shared_task["goal"])
        self.assertTrue(first.capabilities()["memory"]["isolatedByBody"])
        self.assertFalse(shared_b.capabilities()["memory"]["isolatedByBody"])
        self.assertTrue(shared_b.capabilities()["memory"]["sharedScopeConfigured"])

    def test_real_generation_flows_into_a_bounded_product_save_tool(self):
        destination = self.directory / "draft.txt"
        def validate(arguments):
            if set(arguments) != {"text"} or type(arguments["text"]) is not str or len(arguments["text"]) > 500:
                raise ValueError("Invalid bounded draft")
        def save(arguments):
            destination.write_text(arguments["text"], encoding="utf-8")
            return {"saved": True, "characters": len(arguments["text"])}
        body = self.brain.connect("editor", [Tool("save-draft", save, validator=validate)])
        generated_input = {"prompt": "La red ", "length": 40, "seed": 5, "temperature": 0.8}
        result = self.ok(self.request(body, "workflow.run", {"steps": [
            self.step("draft", "language.generate", generated_input),
            self.step("save", "tool.call", {"name": "save-draft", "arguments": {"text": {"$ref": "draft.continuation"}}})]}))
        expected = body.models.execute("language.generate", generated_input)
        self.assertEqual(destination.read_text(encoding="utf-8"), expected["continuation"])
        self.assertEqual(result["completedCount"], 2)
        self.assertEqual(result["steps"][0]["result"], expected)
        self.assertTrue(result["steps"][1]["result"]["saved"])

    def test_invalid_complete_plan_is_rejected_before_any_tool_effect(self):
        calls = []
        body = self.brain.connect("bounded", [Tool("effect", lambda arguments: calls.append(arguments) or {"value": 1})])
        first = self.step("first", "tool.call", {"name": "effect", "arguments": {}})
        invalid_later = [
            self.step("later", "unknown.operation", {}),
            self.step("later", "tool.call", {"name": "unbound", "arguments": {}}),
            self.step("later", "language.generate", {"length": True}),
            self.step("later", "tool.call", {"name": "effect", "arguments": {"value": {"$ref": "future.value"}}}),
            self.step("later", "tool.call", {"name": "effect", "arguments": {"value": {"$ref": "first.value", "extra": 1}}}),
            self.step("later", "tool.call", {"name": "effect", "arguments": {"value": {"$ref": "first.items." + "9" * 5000}}}),
        ]
        for later in invalid_later:
            response = self.request(body, "workflow.run", {"steps": [first, later]})
            self.assertEqual(response["status"], "error")
            self.assertEqual(calls, [])
        response = self.request(body, "workflow.run", {"steps": [self.step(f"step-{number}", "tool.call", first["input"]) for number in range(17)]})
        self.assertEqual(response["status"], "error")
        self.assertEqual(calls, [])

    def test_runtime_failure_preserves_completed_memory_effect_and_stops_next_step(self):
        def fail(_arguments):
            raise RuntimeError("Synthetic adapter failure")
        body = self.brain.connect("partial", [Tool("fail", fail)])
        response = self.request(body, "workflow.run", {"steps": [
            self.step("created", "memory.task.add", {"goal": "Completed task persists"}),
            self.step("failed", "tool.call", {"name": "fail", "arguments": {}}),
            self.step("skipped", "memory.task.add", {"goal": "Never created"})]})
        self.assertEqual(response["error"]["code"], "workflow_failed")
        self.assertEqual(response["result"]["completedCount"], 1)
        self.assertEqual(response["result"]["steps"][0]["id"], "created")
        self.assertEqual(response["result"]["failedStep"]["id"], "failed")
        self.assertFalse(response["result"]["effectsRolledBack"])
        self.assertEqual(self.ok(self.request(body, "memory.task.status", {}))["taskCount"], 1)

    def test_references_traverse_only_json_fields_and_lists_and_report_runtime_path_failure(self):
        body = self.brain.connect("references", [Tool("capture", lambda _args: {"items": [{"value": 7, "flag": True}]}),
                                                  Tool("consume", lambda args: args)])
        first = self.step("capture", "tool.call", {"name": "capture", "arguments": {}})
        valid = self.step("consume", "tool.call", {"name": "consume", "arguments": {
            "nested": [{"value": {"$ref": "capture.items.0.value"}}], "flag": {"$ref": "capture.items.0.flag"}}})
        result = self.ok(self.request(body, "workflow.run", {"steps": [first, valid]}))
        self.assertEqual(result["steps"][1]["result"], {"nested": [{"value": 7}], "flag": True})
        for path in ("capture.items.9999999999999999999.value", "capture.items.0.__class__"):
            invalid = self.step("consume", "tool.call", {"name": "consume", "arguments": {"value": {"$ref": path}}})
            response = self.request(body, "workflow.run", {"steps": [first, invalid]})
            self.assertEqual(response["error"]["code"], "workflow_failed")
            self.assertEqual(response["result"]["completedCount"], 1)
            self.assertEqual(response["result"]["failedStep"]["error"]["code"], "reference_invalid")

    def test_aggregate_response_limit_retains_compact_completed_receipts(self):
        effects = []
        def large(_args):
            effects.append("completed")
            return {"blob": "x" * 400_000}
        body = self.brain.connect("budget", [Tool("large", large)])
        response = self.request(body, "workflow.run", {"steps": [
            self.step(f"step-{number}", "tool.call", {"name": "large", "arguments": {}}) for number in range(4)]})
        self.assertEqual(response["error"]["code"], "workflow_failed")
        self.assertEqual(response["result"]["completedCount"], 3)
        self.assertEqual(len(effects), 3)
        self.assertEqual([step["id"] for step in response["result"]["steps"]], ["step-0", "step-1", "step-2"])
        self.assertTrue(all(step["resultOmitted"] for step in response["result"]["steps"]))
        self.assertFalse(response["result"]["effectsRolledBack"])

    def test_compact_failure_fallback_keeps_receipts_at_json_element_limit(self):
        body = self.brain.connect("fallback")
        partial = {"status": "failed", "steps": [{"id": "finished", "operation": "tool.call", "result": [None] * 20_000}],
                   "completedCount": 1, "failedStep": {"id": "failed", "operation": "tool.call", "error": {"code": "operation_failed", "message": "fixture"}},
                   "effectsRolledBack": False}
        with patch("ia_propia.brain.core.run_plan", side_effect=BrainError("workflow_failed", "Fixture reporting overflow", result=partial)):
            response = self.request(body, "workflow.run", {"steps": [self.step("noop", "memory.task.status", {})]})
        self.assertEqual(response["result"]["completedCount"], 1)
        self.assertEqual(response["result"]["steps"], [{"id": "finished", "operation": "tool.call", "resultOmitted": True}])
        self.assertFalse(response["result"]["effectsRolledBack"])

    def test_json_protocol_rejects_coercion_duplicates_and_invalid_unicode_without_effects(self):
        for raw in ('{"schemaVersion":1,"schemaVersion":1}', '{"value":NaN}', '{"value":1e999}', '{"value":' + '9' * 5000 + '}'):
            with self.assertRaises(BrainError):
                decode_json(raw)
        body = self.brain.connect("protocol")
        payloads = [
            {"schemaVersion": True, "operation": "memory.task.list", "input": {}},
            {"schemaVersion": 1, "operation": "memory.task.list", "input": {}, "requestId": "\ud800"},
            {"schemaVersion": 1, "operation": "\ud800", "input": {}},
            {"schemaVersion": 1, "operation": "memory.task.add", "input": {"goal": "x" * MAX_REQUEST_BYTES}},
            {"schemaVersion": 1, "operation": "memory.task.add", "input": {"goal": float("nan")}},
        ]
        for payload in payloads:
            response = body.request(payload)
            self.assertEqual(response["status"], "error")
            json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.assertEqual(self.ok(self.request(body, "memory.task.status", {}))["taskCount"], 0)


if __name__ == "__main__":
    unittest.main()
