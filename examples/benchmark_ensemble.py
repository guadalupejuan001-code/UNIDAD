"""Finite CPU profiling of real small Transformers under bounded residency.

    python examples/benchmark_ensemble.py --models 2 4 --repeats 5 \
        --max-new-tokens 16 --output docs/sovengi-ensemble-20261013/benchmark.json

Fixtures start from declared random own weights. Their tokenizers and context
lengths differ; the same artifacts are reused for every residency mode. Each
measurement uses a fresh child process and one Torch thread. Outputs and their
hashes are retained; this measures software behavior, not conversation quality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1 if sys.platform == "darwin" else 1024
    return {"bytes": int(raw * unit), "raw": raw, "rawUnit": "bytes" if unit == 1 else "KiB"}


def latency(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"calls": len(values), "p50Milliseconds": statistics.median(values) * 1000,
            "p95Milliseconds": p95 * 1000, "minMilliseconds": min(values) * 1000,
            "maxMilliseconds": max(values) * 1000}


def fixtures(directory, count):
    import torch
    from ia_propia.conversation.artifacts import save_artifact
    from ia_propia.conversation.model import ConversationModel
    from ia_propia.conversation.tokenizer import fit_tokenizer

    torch.set_num_threads(1)
    records = []
    for index in range(count):
        vocab_size = (320, 384, 448, 512)[index]
        context = (512, 768, 1024, 512)[index]
        seed = (7, 11, 19, 23)[index]
        dialogue = [{"role": "user", "content": (
            f"Ensayo ficticio {index}. El taller inventado usa papel azul y abre el jueves. "
            + ("Un documento del laboratorio conserva objetivos y correcciones. " * (index + 2)))},
            {"role": "assistant", "content": "El taller ficticio abre el jueves."}]
        tokenizer = fit_tokenizer([{"messages": dialogue}], vocab_size=vocab_size)
        model = ConversationModel({"vocab_size": tokenizer.get_vocab_size(), "d_model": 16,
            "n_layers": 1, "n_heads": 2, "context_length": context, "ffn_multiplier": 2, "seed": seed})
        target = directory / f"fixture-{index}"
        manifest = save_artifact(target, model, tokenizer,
            training={"fixtureOnly": True, "trained": False, "trainedFrom": "random_weights", "personalData": False})
        records.append({"name": target.name, "modelId": manifest["modelId"],
                        "parameterCount": model.parameter_count, "vocabularySize": tokenizer.get_vocab_size(),
                        "contextLength": context, "seed": seed,
                        "files": {path.name: sha256_file(path) for path in target.iterdir() if path.is_file()}})
    (directory / "fixtures.json").write_text(json.dumps(records), encoding="utf-8")
    return records


def worker(args):
    import torch
    from ia_propia.conversation import artifacts
    from ia_propia.conversation.ensemble import ConversationEnsemble, EnsembleBudget, ModelSpec
    from ia_propia.conversation.runtime import ConversationEngine

    torch.set_num_threads(1)
    records = json.loads((args.fixtures / "fixtures.json").read_text())[:args.worker_models]
    specs = [ModelSpec(name=row["name"], model_dir=args.fixtures / row["name"],
                       expected_model_id=row["modelId"]) for row in records]
    budget = EnsembleBudget(max_candidates=args.worker_models, min_candidates=2,
        max_resident=args.max_resident, max_seconds=60, total_new_tokens=args.worker_models * args.max_new_tokens,
        max_score_calls=args.worker_models * 2, critic_count=2, max_scoring_tokens=256)
    generations, scores = [], []

    original_reply, original_score = ConversationEngine.reply, ConversationEngine.score_reply
    def reply(engine, *messages, **controls):
        result = original_reply(engine, *messages, **controls)
        generations.append({"modelId": engine.model_id, "text": result["text"],
            "sha256": text_hash(result["text"]), "generatedTokens": result["generatedTokens"],
            "discardedPromptTokens": result["discardedPromptTokens"]})
        return result
    def score(engine, *messages, **controls):
        result = original_score(engine, *messages, **controls)
        scores.append({"modelId": engine.model_id, "result": result})
        return result

    baseline = peak_rss()
    messages = [{"role": "system", "content": "Eres SOVENGI. Responde en español con los hechos disponibles."},
                {"role": "user", "content": "¿Cuándo abre el taller ficticio?"}]
    trials = []
    with patch("ia_propia.conversation.artifacts.load_artifact", wraps=artifacts.load_artifact) as loading, \
            patch.object(ConversationEngine, "reply", reply), patch.object(ConversationEngine, "score_reply", score):
        started = time.perf_counter()
        ensemble = ConversationEnsemble(specs, budget=budget, primary_name=specs[0].name)
        construction_seconds = time.perf_counter() - started
        construction_loads = loading.call_count
        try:
            for repeat in range(args.repeats + 1):
                previous_loads, previous_generations, previous_scores = loading.call_count, len(generations), len(scores)
                started = time.perf_counter()
                result = ensemble.reply(messages, max_new_tokens=args.max_new_tokens, temperature=0, seed=42)
                elapsed = time.perf_counter() - started
                executed = generations[previous_generations:]
                generated_ids = {row["modelId"] for row in executed}
                raw_output = result["text"]
                trials.append({"phase": "cold" if repeat == 0 else "warm", "seconds": elapsed,
                    "weightLoads": loading.call_count - previous_loads,
                    "generations": executed, "generationCalls": len(executed),
                    "distinctGeneratingModels": len(generated_ids), "scoreCallsObserved": len(scores) - previous_scores,
                    "selectedText": raw_output, "selectedTextSha256": text_hash(raw_output),
                    "selectedIsExactGeneratedText": any(raw_output == row["text"] for row in executed),
                    "ensemble": result["ensemble"]})
                if len(executed) != args.worker_models or len(generated_ids) != args.worker_models:
                    raise AssertionError("Each configured fixture must generate exactly once per request")
                if result["ensemble"]["cache"]["peakResident"] > args.max_resident:
                    raise AssertionError("Resident engines exceeded the configured limit")
                if not trials[-1]["selectedIsExactGeneratedText"]:
                    raise AssertionError("Ensemble must return unchanged real generation from at least two contributing models")
            measured_peak = peak_rss()
        finally:
            ensemble.close()
    result = {"models": args.worker_models, "maxResident": args.max_resident,
              "generationControls": {"temperature": 0, "maxNewTokens": args.max_new_tokens, "seed": 42},
              "constructionSeconds": construction_seconds, "constructionWeightLoads": construction_loads,
              "coldMilliseconds": trials[0]["seconds"] * 1000,
              "warmLatency": latency([row["seconds"] for row in trials[1:]]),
              "allSelectedOutputsIdentical": len({row["selectedTextSha256"] for row in trials}) == 1,
              "totalWeightLoads": sum(row["weightLoads"] for row in trials) + construction_loads,
              "baselinePeakRss": baseline, "measuredPeakRss": measured_peak,
              "additionalPeakBytes": measured_peak["bytes"] - baseline["bytes"],
              "nativeFactoryAndRegistryPreflight": True,
              "trials": trials}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-models", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-resident", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--fixtures", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any(count not in (2, 3, 4) for count in args.models) or not 3 <= args.repeats <= 20 or not 1 <= args.max_new_tokens <= 32:
        parser.error("Use 2..4 models, 3..20 repetitions and 1..32 generated tokens")
    if args.worker_models is not None:
        worker(args)
        return 0
    if args.output is None or args.output.exists():
        parser.error("Provide a new --output path")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="unidad-ensemble-benchmark-") as temporary:
        directory = Path(temporary)
        fixture_records = fixtures(directory, max(args.models))
        samples = []
        for count in args.models:
            for resident in (1, count):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker-models", str(count),
                           "--max-resident", str(resident), "--fixtures", str(directory),
                           "--repeats", str(args.repeats), "--max-new-tokens", str(args.max_new_tokens)]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
                if completed.returncode:
                    raise RuntimeError(f"Ensemble benchmark worker failed:\n{completed.stderr}")
                sample = json.loads(completed.stdout)
                samples.append(sample)
                print(json.dumps({"models": count, "maxResident": resident,
                    "coldMilliseconds": sample["coldMilliseconds"], "warmLatency": sample["warmLatency"],
                    "totalWeightLoads": sample["totalWeightLoads"]}), flush=True)
    equality = all(len({sample["trials"][0]["selectedTextSha256"] for sample in samples if sample["models"] == count}) == 1
                   for count in args.models)
    critic_equality = all(len({tuple(sorted(sample["trials"][0]["ensemble"]["sharedCritics"]))
                              for sample in samples if sample["models"] == count}) == 1 for count in args.models)
    if not equality:
        raise AssertionError("Residency limits changed selected deterministic output")
    if not critic_equality:
        raise AssertionError("Residency limits changed the set of cross-model judges")
    report = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
        "scope": "actual_small_random_transformer_ensemble_cpu_bounded_residency",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torchThreads": 1},
        "fixture": {"randomOwnWeights": True, "trained": False, "personalData": False, "models": fixture_records},
        "durationSeconds": time.perf_counter() - started, "residencyPreservesDeterministicSelectedOutput": equality,
        "residencyPreservesCriticSet": critic_equality,
        "samples": samples, "conversationQualityValidated": False,
        "limitations": ["Random fixtures demonstrate software integration and output preservation, not Spanish conversation ability.",
                        "Small models do not predict memory or throughput for the full production catalog.",
                        "RSS is a fresh-worker high-water mark including Torch; additionalPeakBytes subtracts its imported baseline.",
                        "Auxiliary character RNN and trend MLP are covered by integration checks, outside these timings."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
