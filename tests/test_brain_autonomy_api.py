"""Actual authenticated transport with declared fixtures and real local file IO.

Scripted proposals here test controller/API mechanics, never learned planning.
The real own-model acceptance experiment is a separate recorded execution.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from ia_propia.brain import Brain, Tool
from ia_propia.brain.autonomy import AutonomyLimits, AutonomyTool
from ia_propia.brain.autonomy_model import ModelActionProposer
from ia_propia.brain.file_tools import SafeFileTools
from tests.test_brain_http import exchange, running_server


OPERATOR_TOKEN = "separate-test-operator-token-0000000000000000"
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


def prepare_mock_context(engine, context_length=4096):
    from ia_propia.conversation.tokenizer import fit_tokenizer
    engine._tokenizer = fit_tokenizer([{"messages": [
        {"role": "user", "content": "Contrato de una prueba técnica declarada."},
        {"role": "assistant", "content": "Este fixture no acredita planificación aprendida."}]}], vocab_size=384)
    engine._model = SimpleNamespace(config={"context_length": context_length})
    return engine


class ScriptedTestProposer:
    origin = "test_fixture"
    proposer_id = "api-scripted-test-fixture-not-learned"

    def __init__(self, actions):
        self.actions = iter(actions)

    def __call__(self, _context):
        return json.dumps(next(self.actions))


def action(tool, arguments):
    return {"action": "tool", "tool": tool, "arguments": arguments, "reason": "Declared test fixture step"}


def payload(operation, values, **extra):
    return {"schemaVersion": 1, "requestId": "autonomy-api-test", "operation": operation, "input": values, **extra}


class BrainAutonomyApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs, self.reports = self.root / "inputs", self.root / "reports"
        self.inputs.mkdir()
        self.reports.mkdir()
        (self.inputs / "uno.txt").write_text("Primera observación.\nDos líneas.\n", encoding="utf-8")
        (self.inputs / "dos.txt").write_text("Trabajo verificable en español.\n", encoding="utf-8")
        self.originals = {path.name: path.read_bytes() for path in self.inputs.iterdir()}
        self.brain = Brain(self.root / "state")
        self.files = SafeFileTools(self.inputs, self.reports)
        self.addCleanup(self.files.close)

    def connect(self, body, actions, **extra):
        return self.brain.connect(body, autonomy_tools=self.files.tools,
                                  autonomy_proposer=ScriptedTestProposer(actions),
                                  autonomy_limits=AutonomyLimits(max_steps=8, max_seconds=30),
                                  autonomy_completion_verifier=self.files.completion_verifier,
                                  autonomy_input_validator=self.files.validate_task_inputs, **extra)

    def request(self, server, operation, values, **extra):
        status, _, response = exchange(server, payload=payload(operation, values, **extra))
        return status, response

    def start(self, server, report="informe.json"):
        status, response = self.request(server, "autonomy.start", {
            "goal": "Analiza dos archivos y guarda un informe sin modificar los originales.",
            "task_inputs": {"files": ["uno.txt", "dos.txt"], "report": report}})
        self.assertEqual(status, 200, response)
        return response["result"]["task_id"]

    def assert_originals(self):
        for name, raw in self.originals.items():
            self.assertEqual((self.inputs / name).read_bytes(), raw)

    def test_opt_in_is_required_and_legacy_tools_keep_working(self):
        session = self.brain.connect("disabled", [Tool("echo", lambda arguments: arguments)])
        with running_server(session) as server:
            status, response = self.request(server, "autonomy.start", {"goal": "Any goal"})
            self.assertEqual(status, 503)
            self.assertEqual(response["error"]["code"], "capability_unavailable")
            status, response = self.request(server, "tool.call", {"name": "echo", "arguments": {"value": 7}})
            self.assertEqual(status, 200)
            self.assertEqual(response["result"], {"value": 7})
        self.assertFalse(session.capabilities()["autonomy"]["enabled"])
        self.assertNotIn("autonomy.start", session.capabilities()["operations"])

    def test_authenticated_api_real_observations_report_and_original_hashes(self):
        session = self.connect("analysis", [action("inspect_file", {"filename": "uno.txt"}),
                                           action("read_file", {"filename": "dos.txt"}),
                                           action("save_report", {"filename": "informe.json", "files": ["uno.txt", "dos.txt"]})])
        with running_server(session) as server:
            status, _, response = exchange(server, payload=payload("autonomy.start", {"goal": "Hidden"}),
                                            headers={"Authorization": None})
            self.assertEqual(status, 401)
            task_id = self.start(server)
            status, response = self.request(server, "autonomy.step", {"task_id": task_id})
            self.assertEqual(status, 200)
            self.assertEqual(response["result"]["steps"], 1)
            status, response = self.request(server, "autonomy.run", {"task_id": task_id})
            self.assertEqual(status, 200, response)
            state = response["result"]
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["source"], "test_fixture")
            self.assertFalse(response["workerCapabilityValidated"])
            status, readback = self.request(server, "autonomy.status", {"task_id": task_id})
            self.assertEqual(readback["result"], state)
        report_raw = (self.reports / "informe.json").read_bytes()
        report = json.loads(report_raw)
        self.assertEqual(report["kind"], "unidad.file-report")
        self.assertEqual({row["filename"] for row in report["files"]}, set(self.originals))
        for row in report["files"]:
            self.assertEqual(row["sha256"], hashlib.sha256(self.originals[row["filename"]]).hexdigest())
        saved = next(event["result"] for event in state["journal"] if event["kind"] == "tool_result" and event["tool"] == "save_report")
        self.assertEqual(saved["sha256"], hashlib.sha256(report_raw).hexdigest())
        self.assert_originals()
        capabilities = session.capabilities()
        self.assertFalse(capabilities["naturalLanguagePlanner"])
        self.assertFalse(capabilities["autonomy"]["modelProposalValidated"])

    def test_tasks_isolate_by_body_even_with_shared_memory_and_requests_cannot_bind_policy(self):
        first = self.connect("first", [], memory_scope="same-memory")
        second = self.connect("second", [], memory_scope="same-memory")
        with running_server(first) as a, running_server(second) as b:
            task_id = self.start(a)
            status, response = self.request(b, "autonomy.status", {"task_id": task_id},
                                           context={"bodyId": "first", "memory_scope": "same-memory"})
            self.assertEqual(response["error"]["code"], "task_not_found")
            for fields in ({"input_root": str(self.root)}, {"tools": ["shell"]},
                           {"model": "alternate"}, {"max_steps": 10000}):
                status, response = self.request(a, "autonomy.start", {"goal": "Read", **fields})
                self.assertEqual(status, 400, response)
            status, response = self.request(a, "autonomy.start", {
                "goal": "Read", "task_inputs": {"files": ["../secrets"], "report": "new.json"}})
            self.assertEqual(status, 400)
        self.assertTrue(second.capabilities()["autonomy"]["isolatedByBody"])

    def test_actor_cannot_complete_without_artifacts_or_escalate_tools(self):
        finished = self.connect("false-complete", [{"action": "finish", "reason": "Unverified actor claim"}])
        escaped = self.connect("escaped", [action("read_file", {"filename": "../uno.txt"})])
        for session, expected in ((finished, "blocked_verification"), (escaped, "blocked_policy")):
            with running_server(session) as server:
                task_id = self.start(server)
                status, response = self.request(server, "autonomy.run", {"task_id": task_id},
                                               context={"approved": True, "tools": ["shell"], "root": str(self.root)})
                self.assertEqual(status, 200)
                self.assertEqual(response["result"]["status"], expected)
        self.assertEqual(list(self.reports.iterdir()), [])
        self.assert_originals()

    def test_cancellation_prevents_later_write_through_existing_api(self):
        session = self.connect("cancelled", [action("inspect_file", {"filename": "uno.txt"}),
                                            action("save_report", {"filename": "informe.json", "files": ["uno.txt"]})])
        with running_server(session) as server:
            task_id = self.start(server)
            self.request(server, "autonomy.step", {"task_id": task_id})
            status, response = self.request(server, "autonomy.cancel", {"task_id": task_id})
            self.assertEqual(response["result"]["status"], "cancelled")
            status, response = self.request(server, "autonomy.run", {"task_id": task_id})
            self.assertEqual(response["result"]["status"], "cancelled")
        self.assertFalse((self.reports / "informe.json").exists())
        self.assert_originals()

    def test_sensitive_effect_requires_separate_operator_and_exact_pending_nonce(self):
        calls = []
        sensitive = AutonomyTool("sensitive", lambda args: calls.append(args) or {"observed": True},
                                 lambda args: set(args) == {"target"}, effect="destructive",
                                 verifier=lambda args, result: {"verified": result == {"observed": True}})
        session = self.brain.connect("operator", autonomy_tools=[sensitive],
                                     autonomy_proposer=ScriptedTestProposer([action("sensitive", {"target": "declared-fixture"})]),
                                     autonomy_operator_token=OPERATOR_TOKEN)
        with running_server(session) as server:
            status, response = self.request(server, "autonomy.start", {"goal": "Declared operator approval fixture"})
            task_id = response["result"]["task_id"]
            status, response = self.request(server, "autonomy.step", {"task_id": task_id}, context={"approved": True})
            self.assertEqual(response["result"]["status"], "needs_approval")
            nonce = response["result"]["pending_approval"]["approval_id"]
            values = {"task_id": task_id, "approval_id": nonce, "approved": True, "operator_token": "wrong"}
            status, response = self.request(server, "autonomy.approve", values, context={"approved": True})
            self.assertEqual(status, 403)
            values.update(operator_token=OPERATOR_TOKEN, approval_id="wrong-nonce")
            status, response = self.request(server, "autonomy.approve", values)
            self.assertEqual(response["error"]["code"], "approval_denied")
            self.assertEqual(calls, [])
            values["approval_id"] = nonce
            status, response = self.request(server, "autonomy.approve", values)
            self.assertEqual(status, 200, response)
            status, response = self.request(server, "autonomy.step", {"task_id": task_id})
            self.assertEqual(calls, [{"target": "declared-fixture"}])
            self.assertNotIn(OPERATOR_TOKEN, json.dumps(response))

    @unittest.skipUnless(HAS_TOKENIZERS, "conversation tokenizer is optional")
    def test_model_adapter_uses_shared_engine_and_preserves_raw_json_failure(self):
        class MockEngineForAdapterTest:
            model_id = "explicit-mock-engine-not-model-evidence"
            def reply(engine, **request):
                engine.last_request = request
                return {"text": '```json\n{"action":"finish","reason":"fixture"}\n```',
                        "modelId": engine.model_id, "discardedPromptTokens": 0}
        mock_engine = prepare_mock_context(MockEngineForAdapterTest())
        self.brain.models._conversation = mock_engine
        proposer = ModelActionProposer(self.brain.models)
        self.assertIs(proposer.engine, self.brain.models.conversation_engine)
        session = self.brain.connect("model-adapter", autonomy_tools=[], autonomy_proposer=proposer)
        with running_server(session) as server:
            status, response = self.request(server, "autonomy.start", {"goal": "Adapter contract only"})
            task_id = response["result"]["task_id"]
            status, response = self.request(server, "autonomy.step", {"task_id": task_id})
            state = response["result"]
        self.assertEqual(state["status"], "blocked_model")
        evidence = next(event for event in state["journal"] if event["kind"] == "proposal")
        self.assertEqual(evidence["raw"], evidence["metadata"]["rawReply"])
        self.assertTrue(evidence["raw"].startswith("```json"))
        self.assertEqual(evidence["metadata"]["modelId"], mock_engine.model_id)
        self.assertEqual(evidence["metadata"]["request"], mock_engine.last_request)
        self.assertFalse(session.capabilities()["naturalLanguagePlanner"])

    @unittest.skipUnless(HAS_TOKENIZERS, "conversation tokenizer is optional")
    def test_parallel_model_receipts_remain_on_the_correct_request_thread(self):
        class MockParallelEngineForAdapterTest:
            model_id = "parallel-adapter-mock-not-model-evidence"
            barrier = threading.Barrier(2)
            def reply(engine, **request):
                goal = json.loads(request["messages"][-1]["content"])["task"]["goal"]
                engine.barrier.wait(timeout=2)
                return {"text": json.dumps({"action": "stop", "reason": goal}), "modelId": engine.model_id}
        self.brain.models._conversation = prepare_mock_context(MockParallelEngineForAdapterTest())
        proposer = ModelActionProposer(self.brain.models)
        both_proposed = threading.Barrier(2)
        def proposed(goal):
            raw = proposer({"task": {"goal": goal}, "tools": []})
            both_proposed.wait(timeout=2)
            metadata = proposer.last_metadata
            return raw, json.loads(metadata["request"]["messages"][-1]["content"])["task"]["goal"], metadata["rawReply"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(proposed, ["first declared test", "second declared test"]))
        for raw, goal, recorded in results:
            self.assertEqual(json.loads(raw)["reason"], goal)
            self.assertEqual(raw, recorded)

    @unittest.skipUnless(HAS_TOKENIZERS, "conversation tokenizer is optional")
    def test_model_context_tokens_are_checked_before_generation_without_dropping_goal(self):
        class MockFiniteEngineForAdapterTest:
            model_id = "finite-token-context-mock-not-model-evidence"
            calls = 0
            def reply(engine, **request):
                engine.calls += 1
                return {"text": '{"action":"stop","reason":"fixture"}', "modelId": engine.model_id}
        engine = prepare_mock_context(MockFiniteEngineForAdapterTest(), context_length=32)
        self.brain.models._conversation = engine
        proposer = ModelActionProposer(self.brain.models)
        goal = "Conserva cada palabra del objetivo. " * 12
        with self.assertRaisesRegex(ValueError, "discard prompt tokens"):
            proposer({"task": {"goal": goal}, "tools": []})
        self.assertEqual(engine.calls, 0)
        metadata = proposer.last_metadata
        self.assertEqual(metadata["contextRejected"], "model_token_limit")
        self.assertGreater(metadata["discardedPromptTokens"], 0)
        self.assertEqual(json.loads(metadata["request"]["messages"][-1]["content"])["task"]["goal"], goal)

    def test_restart_rejects_changed_explicit_task_input_validator_policy(self):
        class HostValidator:
            def __init__(self, policy_id):
                self.policy_id = policy_id
            def __call__(self, inputs):
                return self.policy_id == "broad-v1" or inputs == {}
        first = self.brain.connect("validator-policy", autonomy_tools=[],
                                   autonomy_proposer=ScriptedTestProposer([]),
                                   autonomy_input_validator=HostValidator("broad-v1"))
        with running_server(first) as server:
            status, response = self.request(server, "autonomy.start", {
                "goal": "Persisted validator policy fixture", "task_inputs": {"extension": "broad"}})
            self.assertEqual(status, 200)
            task_id = response["result"]["task_id"]
        restarted = Brain(self.root / "state").connect("validator-policy", autonomy_tools=[],
                                                       autonomy_proposer=ScriptedTestProposer([]),
                                                       autonomy_input_validator=HostValidator("narrow-v2"))
        with running_server(restarted) as server:
            status, response = self.request(server, "autonomy.step", {"task_id": task_id})
        self.assertEqual(status, 200, response)
        self.assertEqual(response["result"]["status"], "blocked_policy")

    def test_large_persisted_journal_returns_bounded_truthful_task_receipt(self):
        observation = AutonomyTool("large_observation", lambda args: {"bytes": "x" * 200000},
                                   lambda args: args == {}, effect="read", retry_safe=True,
                                   verifier=lambda args, result: True)
        session = self.brain.connect("large-journal", autonomy_tools=[observation],
                                     autonomy_proposer=ScriptedTestProposer([
                                         *[action("large_observation", {}) for _ in range(6)],
                                         {"action": "finish", "reason": "No verified completion fixture"}]),
                                     autonomy_limits=AutonomyLimits(max_steps=10, max_result_bytes=262144,
                                                                    max_state_bytes=4194304))
        with running_server(session) as server:
            status, response = self.request(server, "autonomy.start", {"goal": "Bounded transport fixture"})
            task_id = response["result"]["task_id"]
            status, response = self.request(server, "autonomy.run", {"task_id": task_id})
            self.assertEqual(status, 200, response)
            receipt = response["result"]
            self.assertEqual(receipt["status"], "blocked_verification")
            self.assertEqual(receipt["steps"], 7)
            self.assertTrue(receipt["snapshotOmitted"])
            self.assertGreater(receipt["journalEntries"], 6)
            self.assertLess(len(json.dumps(response).encode()), 1048576)
        stored = session._autonomy["controller"].status(task_id)
        self.assertEqual(stored["status"], receipt["status"])
        self.assertEqual(sum(event["kind"] == "tool_result" for event in stored["journal"]), 6)


if __name__ == "__main__":
    unittest.main()
