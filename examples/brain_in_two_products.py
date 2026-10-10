"""One Unidad instance, two product bodies; one Python client and one HTTP client.

All text and tools here are demonstrations. No external account or AI is used.
Run after installing Unidad: python examples/brain_in_two_products.py
"""

import argparse
from http.client import HTTPConnection
import json
from pathlib import Path
import secrets
import tempfile
import threading

from ia_propia.brain import Brain, Tool
from ia_propia.brain.http import create_server


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def writer(filename):
        def save(arguments):
            if set(arguments) != {"text"} or not isinstance(arguments["text"], str):
                raise ValueError("The product expects one text string")
            raw = arguments["text"].encode("utf-8")
            (directory / filename).write_bytes(raw)
            return {"document": filename, "bytes": len(raw)}
        return save

    brain = Brain(state_dir=directory / "memory")
    editor = brain.connect("editor", tools=[
        Tool("draft.save", writer("editor-demo.txt"), "Save this product's demo draft")])
    dashboard = brain.connect("dashboard", tools=[
        Tool("panel.publish", writer("dashboard-demo.txt"), "Save this product's demo panel text")])

    def workflow(tool_name):
        return {"schemaVersion": 1, "operation": "workflow.run", "input": {"steps": [
            {"id": "draft", "operation": "language.generate", "input": {
                "prompt": "Una tarea ", "length": 160, "seed": 42, "temperature": 0.8}},
            {"id": "store", "operation": "tool.call", "input": {
                "name": tool_name, "arguments": {"text": {"$ref": "draft.continuation"}}}}]}}

    python_response = editor.request(workflow("draft.save"))
    token = secrets.token_urlsafe(32)
    with create_server(dashboard, token=token) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPConnection(*server.server_address, timeout=10)
            try:
                raw = json.dumps(workflow("panel.publish")).encode("utf-8")
                client.request("POST", "/v1/requests", body=raw, headers={
                    "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
                response = client.getresponse()
                http_status = response.status
                http_response = json.loads(response.read())
            finally:
                client.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)

    for result in (python_response, http_response):
        if result["status"] != "ok":
            raise RuntimeError("Demo workflow failed: " + json.dumps(result))
    if http_status != 200:
        raise RuntimeError("HTTP demo did not succeed")
    python_model = python_response["result"]["steps"][0]["result"]
    http_model = http_response["result"]["steps"][0]["result"]
    return {"schemaVersion": 1, "bodies": ["editor", "dashboard"],
            "interfaces": ["python", "http"], "httpStatus": http_status,
            "sameModelId": python_model["modelId"] == http_model["modelId"],
            "sameGeneratedText": python_model == http_model,
            "files": ["editor-demo.txt", "dashboard-demo.txt"],
            "responses": [python_response, http_response],
            "experimental": True, "workerCapabilityValidated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Keep demo output here; otherwise use and remove a temporary directory")
    args = parser.parse_args()
    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="unidad-bodies-") as temporary:
            result = run(temporary)
    else:
        result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
