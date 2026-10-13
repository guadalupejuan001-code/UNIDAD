#!/usr/bin/env python3
"""Measure random-weight compute and artifact IO; never train or grade an AI.

Each operation runs in a fresh CPU worker. The legacy IO reference is an explicit
trusted source snapshot imported with the current architecture validator; this
does not claim that the old 20M release supported the larger configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
CONFIG = {"vocab_size": 8192, "d_model": 512, "n_layers": 14, "n_heads": 8,
          "context_length": 1024, "dropout": 0.0, "ffn_multiplier": 4,
          "seed": 42, "parameter_budget": 50_000_000}
EXPECTED_PARAMETERS = 48_852_992
C7_WEIGHTS_SHA = "666527de240a7fc9b082f233699a1c808c6bdd8f929651824ce4975377c7c836"
C7_TOKENIZER_SHA = "fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d"
TRAINING = {"fixtureOnly": True, "rootOrigin": "random_weights", "trainedFrom": "random_weights",
            "optimizerSteps": 0, "trainingCorpusRead": False, "externalWeightsUsed": False,
            "tokenizerSource": "own_C7_tokenizer_reused_for_technical_fixture"}
EVALUATION = {"fixtureOnly": True, "conversationAccepted": False,
              "commercialSubstitutionValidated": False}
DEPENDENCIES = ("configuration.py", "model.py", "artifacts.py", "tokenizer.py", "contract.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    root = REPOSITORY / "src/ia_propia/conversation"
    return {name: digest(root / name) for name in DEPENDENCIES}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def memory():
    values = {}
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, amount, unit = line.split()
                values[name[:-1] + "Bytes"] = int(amount) * 1024
    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    values["ruMaxRssBytes"] = int(high if sys.platform == "darwin" else high * 1024)
    return values


def tensor_digest(tensor):
    return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B")).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(json.dumps(list(tensor.shape)).encode("ascii"))
        h.update(bytes.fromhex(tensor_digest(tensor)))
    return h.hexdigest()


def legacy_module(path):
    name = "ia_propia.conversation._artifacts_pre_stream_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    import torch
    from tokenizers import Tokenizer
    from ia_propia.conversation import artifacts
    from ia_propia.conversation.model import ConversationModel
    from ia_propia.conversation.configuration import count_config_parameters

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert count_config_parameters(CONFIG) == EXPECTED_PARAMETERS
    result = {"phase": args.worker, "torchVersion": str(torch.__version__),
              "device": "cpu", "precision": "float32", "torchThreads": torch.get_num_threads(),
              "torchInteropThreads": torch.get_num_interop_threads(),
              "sourceSha256": source_hashes(), "baselineMemoryAfterImports": memory(),
              "optimizerSteps": 0, "trainingCorpusRead": False, "qualityEvaluationPerformed": False}
    start = time.perf_counter()
    operation_start = start
    if args.worker == "c7_roundtrip":
        before = {name: digest(args.reference / name) for name in
                  ("config.json", "tokenizer.json", "weights.safetensors", "manifest.json")}
        assert before["weights.safetensors"] == C7_WEIGHTS_SHA
        model, tokenizer, manifest = artifacts.load_artifact(args.reference)
        destination = args.output / "c7-roundtrip"
        artifacts.save_artifact(destination, model, tokenizer,
                                training=manifest["training"], evaluation=manifest["evaluation"])
        after = {name: digest(destination / name) for name in before}
        assert after == before
        assert before == {name: digest(args.reference / name) for name in before}
        result.update({"originalC7AllFourFilesPreserved": True, "roundtripAllFourFilesByteExact": True,
                       "weightsSha256": after["weights.safetensors"], "artifactFilesSha256": after,
                       "parameterCount": model.parameter_count})
    else:
        fixture = torch.randint(6, 8192, (1, 65), generator=torch.Generator().manual_seed(7))
        inputs, labels = fixture[:, :-1], fixture[:, 1:]
        if args.worker in ("load_stream", "load_legacy"):
            api = artifacts if args.worker == "load_stream" else legacy_module(args.legacy_artifacts)
            load_start = time.perf_counter()
            model, tokenizer, manifest = api.load_artifact(args.output / "large-stream")
            result["loadSeconds"] = time.perf_counter() - load_start
            result["memoryImmediatelyAfterLoad"] = memory()
            result["loadPeakRuMaxRssBytes"] = memory()["ruMaxRssBytes"]
            expected = json.loads((args.output / "init_backward.result.json").read_text())
            assert state_digest(model) == expected["initialStateDigest"]
            with torch.inference_mode():
                logits = model(inputs)
            assert tensor_digest(logits) == expected["fixtureLogitsSha256"]
            result.update({"stateRoundtripExact": True, "fixtureLogitsRoundtripExact": True,
                           "modelId": manifest["modelId"], "parameterCount": model.parameter_count})
        else:
            tokenizer_path = args.reference / "tokenizer.json"
            assert digest(tokenizer_path) == C7_TOKENIZER_SHA
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            init_start = time.perf_counter()
            model = ConversationModel(CONFIG)
            result["initializationSeconds"] = time.perf_counter() - init_start
            assert model.parameter_count == EXPECTED_PARAMETERS
            assert len(model.state_dict()) == 172
            initial = state_digest(model)
            result.update({"parameterCount": model.parameter_count, "tensorCount": 172,
                           "initialStateDigest": initial, "weightPayloadBytesFloat32": EXPECTED_PARAMETERS * 4,
                           "memoryAfterInitialization": memory()})
            if args.worker == "init_backward":
                operation_start = time.perf_counter()
                logits = model(inputs)
                assert tuple(logits.shape) == (1, 64, 8192) and torch.isfinite(logits).all()
                result["forwardSeconds"] = time.perf_counter() - operation_start
                result["fixtureLogitsSha256"] = tensor_digest(logits)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8192), labels.reshape(-1))
                assert torch.isfinite(loss)
                backward_start = time.perf_counter()
                loss.backward()
                result["backwardSeconds"] = time.perf_counter() - backward_start
                gradients = [p.grad for p in model.parameters()]
                assert all(g is not None and torch.isfinite(g).all() for g in gradients)
                norm = math.sqrt(sum(float(g.norm()) ** 2 for g in gradients))
                assert math.isfinite(norm) and norm > 0
                assert state_digest(model) == initial
                result.update({"fixtureLoss": float(loss.detach()), "gradientTensors": len(gradients),
                               "allGradientsFinite": True, "globalGradientNorm": norm,
                               "weightsUnchangedByBackward": True, "fixtureTokens": 64,
                               "batchSize": 1, "trainingCorpusRead": False})
            else:
                api = artifacts if args.worker == "export_stream" else legacy_module(args.legacy_artifacts)
                destination = args.output / ("large-stream" if args.worker == "export_stream" else "large-legacy")
                operation_start = time.perf_counter()
                manifest = api.save_artifact(destination, model, tokenizer,
                                             training=TRAINING, evaluation=EVALUATION)
                result["exportSeconds"] = time.perf_counter() - operation_start
                result["exportPeakRuMaxRssBytes"] = memory()["ruMaxRssBytes"]
                result.update({"modelId": manifest["modelId"], "files": manifest["files"],
                               "weightsUnchangedByExport": state_digest(model) == initial})
    result["workerWallSecondsAfterImports"] = time.perf_counter() - start
    result["finalMemory"] = memory()
    result["sourceUnmodifiedDuringWorker"] = source_hashes() == result["sourceSha256"]
    assert result["sourceUnmodifiedDuringWorker"]
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def orchestrate(args):
    if args.output.exists():
        raise FileExistsError("Use a new output directory; prior evidence is immutable")
    args.output.mkdir(parents=True, mode=0o700)
    source_before = source_hashes()
    legacy_sha = digest(args.legacy_artifacts)
    c7_before = {name: digest(args.reference / name) for name in
                 ("config.json", "tokenizer.json", "weights.safetensors", "manifest.json")}
    assert c7_before["weights.safetensors"] == C7_WEIGHTS_SHA
    write_json(args.output / "design.json", {
        "purpose": "random-weight-scaling-and-artifact-io-only-not-learned-capability",
        "profile": CONFIG, "expectedParameters": EXPECTED_PARAMETERS, "fixtureTokens": 64,
        "cpuThreads": 1, "optimizerSteps": 0, "trainingCorpusRead": False,
        "legacyReferenceSourceSha256": legacy_sha, "sourceSha256": source_before,
        "c7OriginalFilesSha256": c7_before, "runnerSha256": digest(Path(__file__)),
        "peakMeasurement": "Fresh processes; Linux ru_maxrss and VmHWM, not subtraction of cumulative peaks",
        "legacyAdaptation": "Old artifact IO functions import current architecture validation accepting explicit parameter_budget; old 20M release did not support this profile."})
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    phases = ("init_backward", "export_stream", "export_legacy", "load_stream", "load_legacy", "c7_roundtrip")
    results = {}
    for phase in phases:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                   "--output", str(args.output), "--reference", str(args.reference),
                   "--legacy-artifacts", str(args.legacy_artifacts)]
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=args.timeout)
        (args.output / f"{phase}.stderr.txt").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{phase} failed; inspect its saved stderr: {completed.stderr[-2000:]}")
        result = json.loads(completed.stdout)
        assert result["sourceSha256"] == source_before
        results[phase] = result
        write_json(args.output / f"{phase}.result.json", result)
        print(json.dumps({"phase": phase, "parameters": result["parameterCount"],
                          "peakRssBytes": result["finalMemory"]["ruMaxRssBytes"],
                          "secondsAfterImports": result["workerWallSecondsAfterImports"]}), flush=True)
    assert digest(args.legacy_artifacts) == legacy_sha
    assert source_hashes() == source_before
    assert c7_before == {name: digest(args.reference / name) for name in c7_before}
    assert all(results[p]["initialStateDigest"] == results["init_backward"]["initialStateDigest"]
               for p in ("export_stream", "export_legacy"))
    equality = {name: digest(args.output / "large-stream" / name) == digest(args.output / "large-legacy" / name)
                for name in ("config.json", "tokenizer.json", "weights.safetensors", "manifest.json")}
    assert all(equality.values())
    report = {"schemaVersion": 1, "purpose": "random-weight-scaling-and-artifact-io-only-not-learned-capability",
              "profile": CONFIG, "parameterCount": EXPECTED_PARAMETERS,
              "optimizerSteps": 0, "trainingCorpusRead": False, "qualityEvaluationPerformed": False,
              "conversationAccepted": False, "commercialSubstitutionValidated": False,
              "sourceSha256": source_before, "legacySourceSha256": legacy_sha,
              "runnerSha256": digest(Path(__file__)), "largeOldNewFilesByteExact": equality,
              "c7OriginalPreserved": True, "largeRandomWeightsLocation": str(args.output / "large-stream"),
              "results": results,
              "limits": ["One observation per phase, fixed order, one CPU thread; no universal performance claim.",
                         "Peak RSS is process memory, not total system memory or file-cache consumption.",
                         "64-token batch-one compute fixture does not measure 1024-token training feasibility.",
                         "Backpropagation without optimizer steps does not teach language or improve C7.",
                         "Random 195MB artifact is local; no upload to GitHub as a >100MB file."]}
    write_json(args.output / "report.json", report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REPOSITORY / "models/conversation-reference-c7")
    parser.add_argument("--legacy-artifacts", type=Path, required=True,
                        help="Trusted, unchanged pre-stream artifacts.py source snapshot")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", choices=("init_backward", "export_stream", "export_legacy",
                                            "load_stream", "load_legacy", "c7_roundtrip"))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.reference = args.reference.resolve()
    args.legacy_artifacts = args.legacy_artifacts.resolve()
    if args.worker:
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
