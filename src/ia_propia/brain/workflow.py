"""Validate explicit step graphs and resolve JSON paths; no model plans or eval."""

from copy import deepcopy
import re

from .protocol import BrainError, MAX_RESPONSE_BYTES, MAX_WORKFLOW_STEPS, identifier, validate_json


def is_reference(value):
    return type(value) is dict and set(value) == {"$ref"}


def reference_parts(value):
    if type(value) is not str or not value or len(value) > 1024:
        raise BrainError("reference_invalid", "A reference must name a previous step and JSON path")
    parts = value.split(".")
    # Step IDs cannot contain dots, because dots delimit JSON paths.
    if len(parts) > 16 or any(not part or len(part) > 128 or re.fullmatch(r"[A-Za-z0-9_-]+", part) is None for part in parts):
        raise BrainError("reference_invalid", "Reference paths support JSON object fields and list indices only")
    return parts[0], parts[1:]


def references(value):
    if type(value) is dict:
        if "$ref" in value:
            if not is_reference(value):
                raise BrainError("reference_invalid", "Reference objects must contain only $ref")
            yield reference_parts(value["$ref"])
        else:
            for child in value.values():
                yield from references(child)
    elif type(value) is list:
        for child in value:
            yield from references(child)


def has_references(value):
    return next(references(value), None) is not None


def validate_plan(input_data, validate_operation):
    if type(input_data) is not dict or set(input_data) != {"steps"}:
        raise BrainError("invalid_input", "Workflow input must contain only steps")
    steps = input_data["steps"]
    if type(steps) is not list or not 1 <= len(steps) <= MAX_WORKFLOW_STEPS:
        raise BrainError("invalid_input", f"Workflow requires 1 to {MAX_WORKFLOW_STEPS} explicit steps")
    previous = set()
    for step in steps:
        if type(step) is not dict or set(step) != {"id", "operation", "input"}:
            raise BrainError("invalid_input", "Each workflow step needs id, operation and input")
        step_id = identifier(step["id"], "step id")
        if "." in step_id or step_id in previous:
            raise BrainError("invalid_input", "Step identities must be unique and contain no dots")
        if step["operation"] == "workflow.run":
            raise BrainError("invalid_input", "Nested workflows are not supported")
        if type(step["input"]) is not dict:
            raise BrainError("invalid_input", "Workflow step input must be an object")
        for dependency, _path in references(step["input"]):
            if dependency not in previous:
                raise BrainError("reference_invalid", "References must target an earlier step in this workflow")
        validate_operation(step["operation"], step["input"], allow_refs=True)
        previous.add(step_id)
    return deepcopy(steps)


def resolve(value, results):
    if is_reference(value):
        step_id, path = reference_parts(value["$ref"])
        if step_id not in results:
            raise BrainError("reference_invalid", "Reference result is unavailable")
        current = results[step_id]
        for segment in path:
            if type(current) is dict and segment in current:
                current = current[segment]
            elif type(current) is list and segment.isascii() and segment.isdecimal() and len(segment) <= 19 and int(segment) < len(current):
                current = current[int(segment)]
            else:
                raise BrainError("reference_invalid", "Reference JSON path does not exist")
        return deepcopy(current)
    if type(value) is dict:
        return {key: resolve(child, results) for key, child in value.items()}
    if type(value) is list:
        return [resolve(child, results) for child in value]
    return value


def run_plan(steps, execute):
    completed, results = [], {}
    for step in steps:
        try:
            input_data = resolve(step["input"], results)
            validate_json(input_data)
            result = execute(step["operation"], input_data)
            validate_json(result, max_bytes=MAX_RESPONSE_BYTES)
        except BrainError as error:
            partial = {"status": "failed", "steps": completed, "completedCount": len(completed),
                       "failedStep": {"id": step["id"], "operation": step["operation"],
                                      "error": {"code": error.code, "message": error.message}}, "effectsRolledBack": False}
            raise BrainError("workflow_failed", "Explicit workflow stopped after a failed step", result=partial) from error
        results[step["id"]] = result
        completed.append({"id": step["id"], "operation": step["operation"], "result": result})
        try:
            # Reserve room for the envelope and a failed-step explanation.
            validate_json({"schemaVersion": 1, "requestId": "x" * 128, "bodyId": "x" * 128,
                           "operation": "workflow.run", "status": "ok", "workerCapabilityValidated": False,
                           "result": {"status": "completed", "steps": completed, "completedCount": len(completed)}},
                          max_bytes=MAX_RESPONSE_BYTES - 4096)
        except BrainError as error:
            summary = [{"id": item["id"], "operation": item["operation"], "resultOmitted": True} for item in completed]
            partial = {"status": "failed", "steps": summary, "completedCount": len(completed),
                       "failedStep": {"id": step["id"], "operation": step["operation"], "phase": "response_encoding",
                                      "error": {"code": "response_too_large", "message": "Completed results exceed the response budget"}},
                       "effectsRolledBack": False}
            raise BrainError("workflow_failed", "Workflow stopped after completed effects reached the response limit", result=partial) from error
    return {"status": "completed", "steps": completed, "completedCount": len(completed)}
