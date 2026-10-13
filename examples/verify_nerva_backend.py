"""Regenerate own-weight char-RNN parity evidence for the optional NERVA backend.

This does not test C7 conversation quality or benchmark device performance.
Times include validation, temporary JSON transfer and Node subprocess startup.
No checkpoint is trained, changed or promoted.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import resources
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

from ia_propia.language.model import generate, validate_artifact
from ia_propia.language.nerva_backend import generate_nerva, NERVA_VERSION, RUNTIME_FILES


PROMPTS = ("Una tarea ", "Hola", "Aprender ", "IA", "El futuro ", "", "🧠未知")


def verify():
    assets = resources.files("ia_propia.assets")
    raw = assets.joinpath("language-demo.json").read_bytes()
    artifact = validate_artifact(json.loads(raw))
    cases = []
    numpy_times, nerva_times = [], []
    for prompt in PROMPTS:
        start = time.perf_counter()
        expected = generate(artifact, prompt, length=64, temperature=0, seed=42)
        numpy_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        actual = generate_nerva(artifact, prompt, length=64, seed=42)
        nerva_times.append(time.perf_counter() - start)
        backend = actual.pop("executionBackend")
        cases.append({"prompt": prompt, "effectivePrompt": actual["effectivePrompt"],
                      "continuation": actual["continuation"], "generatedCharacters": 64,
                      "sameOriginalResultFields": actual == expected,
                      "executionBackend": backend})
    sha = lambda content: hashlib.sha256(content).hexdigest()
    report = {
        "schemaVersion": 1, "status": "passed" if all(case["sameOriginalResultFields"] for case in cases) else "failed",
        "scope": "Complete greedy inference of the packaged own-weight character RNN in NERVA; not C7 conversation",
        "modelId": artifact["modelId"], "artifactSha256": artifact["artifactSha256"],
        "artifactFileSha256": sha(raw), "artifactUnchanged": raw == assets.joinpath("language-demo.json").read_bytes(),
        "nervaVersion": NERVA_VERSION, "dtype": "f64", "temperature": 0, "seed": 42,
        "checkedGeneratedCharacters": len(PROMPTS) * 64,
        "programSha256": sha(assets.joinpath("nerva", "language_rnn.nv").read_bytes()),
        "canonicalRuntimeSha256": {name: sha(assets.joinpath("nerva", name).read_bytes()) for name in RUNTIME_FILES},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "node": subprocess.check_output(["node", "--version"], text=True).strip()},
        "observedEndToEndTimesSeconds": {"numpyCalls": numpy_times, "nervaCalls": nerva_times,
                                         "numpyMedian": statistics.median(numpy_times),
                                         "nervaMedian": statistics.median(nerva_times),
                                         "includesNervaSubprocessStartup": True,
                                         "performanceImprovementEstablished": False},
        "predictions": cases,
        "limitations": [
            "A small character predictor is not an instruction-following or commercially competitive conversational AI.",
            "Parity is empirical for these prompts; JS accumulation and tanh can differ from NumPy near an argmax tie.",
            "Sampling and training through this adapter are not implemented; runtime limits and errors remain explicit.",
            "Timing observations are seven sequential calls on this host, including transfer and startup, not a controlled speed benchmark.",
            "No own or external model weights were trained, changed or promoted by this verification.",
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = verify()
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checkedGeneratedCharacters": report["checkedGeneratedCharacters"],
                      "artifactUnchanged": report["artifactUnchanged"], "report": str(arguments.report)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" and report["artifactUnchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
