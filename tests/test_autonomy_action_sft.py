"""Declared fixtures record real tool effects; these tests do not train a model."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if importlib.util.find_spec("tokenizers") is None:
    raise unittest.SkipTest("Action SFT preparation requires optional tokenizers")

from ia_propia.brain.autonomy_model import ACTION_INSTRUCTION, ModelActionProposer, _schema_constraints
from ia_propia.brain.file_tools import SafeFileTools
from ia_propia.conversation import training
from ia_propia.conversation.tokenizer import (
    ASSISTANT_ID, EOS_ID, IGNORE_INDEX, encode_prompt, fit_tokenizer,
)


HELPER_PATH = Path(__file__).resolve().parents[1] / "examples" / "prepare_autonomy_action_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_autonomy_action_sft", HELPER_PATH)
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
SPEC.loader.exec_module(helper)


def tool_action(name, arguments):
    return {"action": "tool", "tool": name, "arguments": arguments,
            "reason": "Paso de un fixture declarado."}


def task_spec():
    return {"split": "train", "groupId": "action-sft-test:two-files",
            "family": "inspect-read-report", "goal": "Analiza dos archivos sin alterar los originales.",
            "files": {"uno.txt": "La reunión será el martes.\nParticipan Lucía y Omar.\n",
                      "dos.txt": "Informe de prueba.\nSólo se leen los originales.\n"},
            "report": "informe.json",
            "actions": [tool_action("inspect_file", {"filename": "uno.txt"}),
                        tool_action("read_file", {"filename": "dos.txt"}),
                        tool_action("save_report", {"filename": "informe.json",
                                                    "files": ["uno.txt", "dos.txt"]})]}


def context_fixture():
    return {"task": {"goal": "Analiza uno.txt sin alterar los originales.",
                     "task_inputs": {"files": ["uno.txt"], "report": "informe.json"},
                     "task_id": "private-task-id", "policy": {"privateRoot": "/private/fixture"},
                     "journal": [{"kind": "failure", "code": "ignored-journal",
                                  "message": "An explicit journal takes precedence."}]},
            "tools": [{"name": "inspect_file", "effect": "read", "policy_id": "private-policy-id",
                       "description": "Never place host prose in the prompt.",
                       "schema": {"type": "object", "description": "Schema prose.",
                                  "properties": {"filename": {"type": "string", "minLength": 1,
                                                               "description": "Filename prose."},
                                                 "description": {"type": "string", "enum": ["description"]}},
                                  "required": ["filename"], "additionalProperties": False}}],
            "journal": [{"kind": "proposal", "raw": "Private previous proposal."},
                        {"kind": "effect_started", "tool": "inspect_file",
                         "arguments": {"filename": "uno.txt"}},
                        {"kind": "handler_returned", "tool": "inspect_file",
                         "result": {"filename": "uno.txt", "size_bytes": 4}},
                        {"kind": "tool_result", "tool": "inspect_file",
                         "result": {"filename": "uno.txt", "size_bytes": 4},
                         "verification": {"verified": True, "result": {"duplicate": "omit"}}},
                        {"kind": "recovery_verification", "tool": "inspect_file",
                         "observation": {"verified": True}},
                        {"kind": "failure", "code": "file_changed", "message": "Observed failure."}]}


class AutonomyActionSftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Own tiny tokenizer, fitted on fixture text only; no weights are loaded.
        texts = [json.dumps(context_fixture(), ensure_ascii=False),
                 json.dumps(task_spec(), ensure_ascii=False), ACTION_INSTRUCTION,
                 'name schema effect type object properties filename files required additionalProperties '
                 'string minLength maxLength array items minItems maxItems uniqueItems observations '
                 'inspect_file read_file save_report arguments result verified size_bytes sha256 statistics '
                 'lines utf8 characters words encoding content task task_inputs goal report']
        cls.tokenizer = fit_tokenizer([{"messages": [
            {"role": "system", "content": ACTION_INSTRUCTION},
            {"role": "user", "content": " ".join(texts)},
            {"role": "assistant", "content": json.dumps(task_spec()["actions"][-1], ensure_ascii=False)}]}] * 3,
            vocab_size=1024)

    def test_projection_is_the_exact_shared_model_prompt_without_private_state(self):
        context = context_fixture()
        original = copy.deepcopy(context)
        requests = []

        def reply(**request):
            requests.append(copy.deepcopy(request))
            return {"text": '{"action":"stop","reason":"Prueba declarada."}'}

        engine = SimpleNamespace(model_id="test-engine-without-weights", reply=reply,
                                 _tokenizer=self.tokenizer, _model=SimpleNamespace(config={"context_length": 2048}))
        proposer = ModelActionProposer(SimpleNamespace(conversation_engine=engine), max_new_tokens=128)
        proposer(context)
        messages = helper.compact_projection(context)
        self.assertEqual(messages, requests[0]["messages"])
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], ACTION_INSTRUCTION)
        projected = json.loads(messages[1]["content"])
        self.assertEqual(set(projected), {"task", "tools", "observations"})
        self.assertEqual(projected["task"], {key: context["task"][key] for key in ("goal", "task_inputs")})
        schema = projected["tools"][0]["schema"]
        self.assertNotIn("description", schema)
        self.assertNotIn("description", schema["properties"]["filename"])
        self.assertEqual(schema["properties"]["description"]["enum"], ["description"])
        self.assertEqual(projected["observations"], [
            {"tool": "inspect_file", "arguments": {"filename": "uno.txt"},
             "result": {"filename": "uno.txt", "size_bytes": 4}, "verified": True,
             "recovery": {"verified": True}},
            {"failure": "file_changed", "message": "Observed failure."}])
        self.assertNotIn("private", messages[1]["content"])
        self.assertEqual(context, original)

    def test_declared_fixture_records_raw_action_and_an_independent_context_snapshot(self):
        context = context_fixture()
        action = {"action": "stop", "reason": "Fixture explícito."}
        raw = json.dumps(action, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        fixture = helper.DeclaredActionFixture(self.tokenizer, [action])
        action["reason"] = "Changed after fixture creation."
        returned = fixture(context)
        self.assertEqual(fixture.origin, "test_fixture")
        self.assertEqual(returned, raw)
        self.assertEqual(len(fixture.captures), 1)
        capture = fixture.captures[0]
        self.assertEqual(capture["rawAction"], raw)
        self.assertEqual(capture["messages"], helper.compact_projection(context))
        self.assertEqual(capture["context"], context)
        context["task"]["goal"] = "Changed after generation."
        self.assertNotEqual(capture["context"]["task"]["goal"], context["task"]["goal"])

    def test_small_context_control_invokes_the_real_adapter_without_generation(self):
        context = context_fixture()
        while True:
            messages = helper.compact_projection(context)
            prompt, discarded = encode_prompt(messages, self.tokenizer, 2048)
            self.assertEqual(discarded, 0)
            if len(prompt) >= 1100:
                break
            context["task"]["goal"] += " medida" * 25
        self.assertLessEqual(len(prompt), 1500)
        fixture = helper.DeclaredActionFixture(self.tokenizer, [{"action": "stop", "reason": "Control declarado."}])
        fixture(context)
        original_captures = copy.deepcopy(fixture.captures)
        adapter_calls = []
        real_call = ModelActionProposer.__call__

        def observe_adapter_call(adapter, observed_context):
            adapter_calls.append((adapter.engine._model.config["context_length"], copy.deepcopy(observed_context)))
            return real_call(adapter, observed_context)

        with patch.object(ModelActionProposer, "__call__", observe_adapter_call):
            controls = helper.check_small_context_rejections(fixture.captures, self.tokenizer)
        self.assertEqual(adapter_calls, [(1024, context)])
        self.assertEqual(fixture.captures, original_captures)
        self.assertEqual(len(controls), 1)
        control = controls[0]
        self.assertEqual(control["code"], "context_too_large")
        self.assertEqual(control["contextLength"], 1024)
        self.assertTrue(control["actualAdapterInvoked"])
        self.assertEqual(control["generationCalls"], 0)
        self.assertEqual(control["promptTokens"], len(prompt))
        self.assertEqual(control["discardedPromptTokens"], len(prompt) - 1024)
        self.assertEqual(control["contextSha256"],
                         hashlib.sha256(fixture.captures[0]["messages"][1]["content"].encode("utf-8")).hexdigest())
        self.assertFalse(control["positiveTrainingRecordCreated"])
        self.assertFalse(control["modelWeightsLoaded"])

    def test_real_file_task_records_verified_effects_and_a_physically_written_report(self):
        # Use the corpus builder's registry configuration, with Unicode inputs
        # whose measurements we can compute independently.
        spec = helper.build_task_specs()[1]
        spec["files"] = dict(zip(spec["files"], task_spec()["files"].values()))
        before = copy.deepcopy(spec)
        installed = []
        original_registries = []
        real_init = SafeFileTools.__init__
        real_install = SafeFileTools._install_report

        def observe_init(files, *args, **kwargs):
            real_init(files, *args, **kwargs)
            original_registries.append([{"name": tool.name, "schema": copy.deepcopy(tool.schema),
                                         "effect": tool.effect} for tool in files.tools])

        def observe_install(files, filename, raw):
            real_install(files, filename, raw)
            installed.append(((files.report_root / filename).read_bytes(), str(files.report_root)))
            self.assertEqual([{"name": tool.name, "schema": tool.schema, "effect": tool.effect}
                              for tool in files.tools], original_registries[0])

        with tempfile.TemporaryDirectory(prefix="action-sft-contract-") as directory:
            with patch.object(SafeFileTools, "__init__", observe_init), \
                    patch.object(SafeFileTools, "_install_report", observe_install):
                records, receipt = helper.execute_task(spec, self.tokenizer, workspace_parent=directory)
        self.assertEqual(spec, before)
        self.assertEqual(receipt["terminalStatus"], "completed")
        self.assertEqual(receipt["source"], "test_fixture")
        expected_hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                           for name, text in spec["files"].items()}
        self.assertEqual(receipt["originalHashesBefore"], expected_hashes)
        self.assertEqual(receipt["originalHashesAfter"], expected_hashes)
        self.assertTrue(receipt["originalsUnchanged"])
        self.assertFalse(receipt["learnedPlanningProven"])
        self.assertEqual(len(installed), 1)
        report = json.loads(installed[0][0])
        self.assertEqual(report, receipt["reportDocument"])
        self.assertEqual(receipt["reportBytesSha256"], hashlib.sha256(installed[0][0]).hexdigest())
        self.assertEqual(report["kind"], "unidad.file-report")
        self.assertEqual({item["filename"] for item in report["files"]}, set(spec["files"]))
        for measured in report["files"]:
            text = spec["files"][measured["filename"]]
            self.assertEqual(measured["sha256"], expected_hashes[measured["filename"]])
            self.assertEqual(measured["size_bytes"], len(text.encode("utf-8")))
            self.assertEqual(measured["statistics"]["characters"], len(text))
        proposals = [event for event in receipt["journal"] if event["kind"] == "proposal"]
        results = [event for event in receipt["journal"] if event["kind"] == "tool_result"]
        self.assertEqual(len(records), 3)
        self.assertEqual(len(proposals), 3)
        self.assertEqual([event["tool"] for event in results], ["inspect_file", "read_file", "save_report"])
        self.assertTrue(all(event["source"] == "test_fixture" and event["valid"] for event in proposals))
        self.assertTrue(all(event["verification"]["verified"] for event in results))
        generic_registry = [{**tool, "schema": _schema_constraints(tool["schema"])}
                            for tool in original_registries[0]]
        self.assertEqual({tool["name"] for tool in generic_registry}, {"inspect_file", "read_file", "save_report"})
        for record, proposal in zip(records, proposals):
            self.assertEqual(record["split"], spec["split"])
            self.assertEqual(record["groupId"], spec["groupId"])
            self.assertEqual(record["source"]["kind"], "authored-synthetic")
            self.assertEqual(record["source"]["fixtureOrigin"], "test_fixture")
            self.assertEqual(record["source"]["supervisionPolicy"], "assistant_last_reply")
            self.assertTrue(record["source"]["actualToolExecution"])
            self.assertFalse(record["source"]["modelResponsesUsed"])
            self.assertEqual([message["role"] for message in record["messages"]], ["system", "user", "assistant"])
            self.assertEqual(record["messages"][-1]["content"], proposal["raw"])
            self.assertEqual(record["messages"][:-1], proposal["metadata"]["request"]["messages"])
            self.assertFalse(proposal["metadata"]["modelGeneration"])
            self.assertEqual(proposal["metadata"]["contextTokens"], 2048)
            registered_tools = json.loads(record["messages"][1]["content"])["tools"]
            self.assertEqual(registered_tools, generic_registry)
            self.assertNotIn('"const":', json.dumps(registered_tools))
            self.assertNotIn('"enum":', json.dumps(registered_tools))
            self.assertNotIn(directory, json.dumps(record["messages"], ensure_ascii=False))
            self.assertNotIn(installed[0][1], json.dumps(record["messages"], ensure_ascii=False))
        self.assertEqual(json.loads(records[0]["messages"][1]["content"])["observations"], [])
        observed = json.loads(records[2]["messages"][1]["content"])["observations"]
        self.assertEqual([item["result"]["sha256"] for item in observed], list(expected_hashes.values()))
        self.assertEqual(observed[1]["result"]["content"],
                         spec["files"][spec["actions"][1]["arguments"]["filename"]])

    def test_token_audit_uses_native_last_reply_masks_and_teaches_eos_once(self):
        prior = '{"action":"finish","reason":"Respuesta anterior sin evidencia."}'
        target = '{"action":"stop","reason":"No hay evidencia de finalización."}'
        row = {"id": "mask-fixture", "groupId": "mask-fixture", "split": "train",
               "source": {"dataset": "declared-test-fixture"},
               "messages": [{"role": "system", "content": ACTION_INSTRUCTION},
                            {"role": "user", "content": "Analiza los archivos registrados."},
                            {"role": "assistant", "content": prior},
                            {"role": "user", "content": "El informe aún no existe."},
                            {"role": "assistant", "content": target}]}
        expected = self.tokenizer.encode(target, add_special_tokens=False).ids + [EOS_ID]
        torch_was_loaded = "torch" in sys.modules
        with patch.object(training, "_torch", side_effect=AssertionError("Mask audit must not load ML weights")):
            audit = helper.audit_tokens([row], self.tokenizer)
            for policy in ("contiguous_nonoverlapping", "sliding_context"):
                windows = list(training._record_windows(row, self.tokenizer, 2048, policy))
                self.assertEqual(len(windows), 1)
                inputs, labels = windows[0]
                last = max(index for index, token in enumerate(inputs) if token == ASSISTANT_ID)
                self.assertEqual(labels[:last], [IGNORE_INDEX] * last)
                self.assertEqual([label for label in labels if label != IGNORE_INDEX], expected)
        self.assertEqual(audit["recordCount"], 1)
        self.assertEqual(audit["contextLength"], 2048)
        self.assertEqual(audit["targetTokensIncludingEos"], len(expected))
        self.assertTrue(audit["eachFinalReplyTargetExactlyOnce"])
        self.assertTrue(audit["historicalAssistantTargetsIgnored"])
        self.assertTrue(audit["promptAndTargetFit"])
        self.assertEqual(audit["torchLoaded"], torch_was_loaded)
        self.assertEqual("torch" in sys.modules, torch_was_loaded)
        self.assertFalse(audit["modelWeightsLoaded"])

    def test_task_specs_are_deterministic_and_keep_complete_worlds_in_one_split(self):
        specs = helper.build_task_specs()
        self.assertEqual(specs, helper.build_task_specs())
        self.assertEqual(sum(spec["split"] == "train" for spec in specs), 64)
        self.assertEqual(sum(spec["split"] == "validation" for spec in specs), 8)
        self.assertEqual(len({spec["groupId"] for spec in specs}), len(specs))
        split_groups = {split: {spec["groupId"] for spec in specs if spec["split"] == split}
                        for split in ("train", "validation")}
        self.assertTrue(split_groups["train"].isdisjoint(split_groups["validation"]))
        for key in ("family", "world"):
            self.assertTrue({spec[key] for spec in specs if spec["split"] == "train"}.isdisjoint(
                {spec[key] for spec in specs if spec["split"] == "validation"}))
        train_worlds = {json.dumps(spec["files"], ensure_ascii=False, sort_keys=True)
                        for spec in specs if spec["split"] == "train"}
        validation_worlds = {json.dumps(spec["files"], ensure_ascii=False, sort_keys=True)
                             for spec in specs if spec["split"] == "validation"}
        self.assertTrue(train_worlds.isdisjoint(validation_worlds))
        self.assertTrue({name for spec in specs if spec["split"] == "train" for name in spec["files"]}.isdisjoint(
            {name for spec in specs if spec["split"] == "validation" for name in spec["files"]}))
        for spec in specs:
            self.assertTrue(spec["files"])
            self.assertEqual(len(spec["actions"]), 3)
            self.assertTrue(all(action["tool"] in {"inspect_file", "read_file"} for action in spec["actions"][:-1]))
            self.assertEqual({action["arguments"]["filename"] for action in spec["actions"][:-1]}, set(spec["files"]))
            self.assertEqual(spec["actions"][-1]["tool"], "save_report")
            self.assertEqual(set(spec["actions"][-1]["arguments"]["files"]), set(spec["files"]))


if __name__ == "__main__":
    unittest.main()
