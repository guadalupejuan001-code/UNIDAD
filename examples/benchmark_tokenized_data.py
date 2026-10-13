r"""Compare in-memory and mmap data preparation on fictional Spanish documents.

Run from the repository root, for example::

    python examples/benchmark_tokenized_data.py --records 200 2000 \
        --min-text-bytes 2048 --max-text-bytes 4096 --repeats 3 \
        --output docs/tokenized-data-20261013/benchmark.json

Each measurement runs in a fresh child process. One TRAIN-only tokenizer is
fitted once and frozen for every size and mode. Every input and label window in
TRAIN and VALIDATION is hashed in order to verify equivalence. This does not
allocate a model, train parameters, or measure total training/system memory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATASET = "unidad-tokenized-data-benchmark-v1"
WORDS = (
    "casa calle plaza barrio ciudad pueblo bosque lago río montaña viaje tren "
    "puerta ventana mesa silla libro papel lápiz mochila jardín huerto cocina "
    "pan fruta agua café comida trabajo paseo música película historia dibujo "
    "lunes martes miércoles jueves viernes sábado domingo mañana tarde noche "
    "amigo vecina familia escuela oficina taller tienda mercado parque puente "
    "verde azul rojo amarillo blanco negro suave pequeño grande rápido lento "
    "leer escribir escuchar preguntar contar recordar explicar revisar ordenar "
    "caminar correr buscar encontrar pensar elegir guardar abrir cerrar cambiar "
    "antes después cerca lejos dentro fuera arriba abajo juntos siempre quizá "
    "tranquilo amable claro nuevo antiguo sencillo distinto suficiente posible "
    "una el con para desde hasta durante porque cuando mientras aunque también"
).split()
SOURCE_FILES = (
    "src/ia_propia/conversation/data.py",
    "src/ia_propia/conversation/language_data.py",
    "src/ia_propia/conversation/tokenizer.py",
    "src/ia_propia/conversation/training.py",
    "src/ia_propia/conversation/disk_data.py",
    "src/ia_propia/conversation/contract.py",
    "src/ia_propia/conversation/configuration.py",
    "examples/benchmark_tokenized_data.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixture(path, count, minimum, maximum, seed):
    """Write one record at a time, never collecting corpus text in memory."""
    digest = hashlib.sha256()
    split_counts = {"train": 0, "validation": 0}
    total_text_bytes = 0
    observed_minimum, observed_maximum = maximum, 0
    with path.open("xb") as stream:
        for index in range(count):
            rng = random.Random(seed + index * 1_000_003)
            target = rng.randint(minimum, maximum)
            prefix = f"Documento ficticio de rendimiento {index:08d}.\n"
            parts, size = [prefix], len(prefix.encode("utf-8"))
            while size < target:
                part = rng.choice(WORDS) + (".\n" if rng.randrange(16) == 0 else " ")
                parts.append(part)
                size += len(part.encode("utf-8"))
            text = "".join(parts).encode("utf-8")[:target].decode("utf-8", errors="ignore")
            text_size = len(text.encode("utf-8"))
            split = "validation" if index % 5 == 0 else "train"
            identifier = f"synthetic-doc-{index:08d}"
            record = {"id": identifier, "groupId": identifier, "split": split,
                      "source": {"dataset": DATASET, "kind": "synthetic-performance-fixture",
                                 "generator": "examples/benchmark_tokenized_data.py", "seed": seed},
                      "text": text}
            raw = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(raw)
            digest.update(raw)
            split_counts[split] += 1
            total_text_bytes += text_size
            observed_minimum = min(observed_minimum, text_size)
            observed_maximum = max(observed_maximum, text_size)
    return {"sha256": digest.hexdigest(), "fileBytes": path.stat().st_size,
            "records": count, "splits": split_counts, "textBytes": total_text_bytes,
            "observedMinTextBytes": observed_minimum, "observedMaxTextBytes": observed_maximum}


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. The raw value remains in evidence.
    unit_bytes = 1 if sys.platform == "darwin" else 1024
    return {"bytes": int(raw * unit_bytes), "raw": raw,
            "rawUnit": "bytes" if unit_bytes == 1 else "KiB"}


def window_digests(examples):
    """Read every window sequentially with at most one window of scratch data."""
    result, combined = {}, hashlib.sha256()
    for split in ("train", "validation"):
        windows = examples[split]
        digest = hashlib.sha256()
        digest.update(split.encode("ascii"))
        digest.update(struct.pack("<Q", len(windows)))
        input_count, supervised_count = 0, 0
        for inputs, labels in windows:
            if len(inputs) != len(labels):
                raise AssertionError("Input and label window lengths differ")
            digest.update(struct.pack("<Q", len(inputs)))
            digest.update(struct.pack(f"<{len(inputs)}q", *inputs))
            digest.update(struct.pack(f"<{len(labels)}q", *labels))
            input_count += len(inputs)
            supervised_count += sum(int(label) != -100 for label in labels)
        combined.update(digest.digest())
        result[split] = {"windows": len(windows), "inputTokens": input_count,
                         "supervisedTokens": supervised_count, "sha256": digest.hexdigest()}
    return {"splits": result, "sha256": combined.hexdigest(),
            "format": "train then validation; split ASCII, uint64 LE count/length, int64 LE inputs then labels"}


def worker(args):
    # Import the same packages in every child before the baseline. No torch or
    # model is imported; package import overhead is visible in baseline RSS.
    from tokenizers import Tokenizer
    from ia_propia.conversation.disk_data import StreamingCorpus, build_or_load_windows, close_windows
    from ia_propia.conversation.language_data import load_text_corpus
    from ia_propia.conversation.training import _encoded_data

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cache_existed = args.cache.exists() if args.worker_mode != "memory" else None
    if args.worker_mode == "mmap_build" and cache_existed:
        raise AssertionError("Fresh cache build requires an absent cache directory")
    if args.worker_mode == "mmap_reload" and not cache_existed:
        raise AssertionError("Cache reload requires a previously built cache directory")
    baseline = peak_rss()
    started = time.perf_counter()
    records = None
    examples = None
    try:
        if args.worker_mode == "memory":
            records = load_text_corpus(args.corpus)
            examples = _encoded_data(records, tokenizer, args.context_length, args.window_policy,
                                     objective="causal_lm", partition_policy="train_validation_only")[0]
            cache_metadata = None
        else:
            records = StreamingCorpus(args.corpus, {"objective": "causal_lm",
                                                    "partition_policy": "train_validation_only"})
            examples = build_or_load_windows(args.cache, records, tokenizer, args.context_length,
                                             args.window_policy, objective="causal_lm",
                                             partition_policy="train_validation_only",
                                             corpus_sha256=args.corpus_sha256, include_test=False)
            cache_metadata = examples.cache_metadata
        preparation_seconds = time.perf_counter() - started
        prepared_peak = peak_rss()
        verification_started = time.perf_counter()
        digests = window_digests(examples)
        verification_seconds = time.perf_counter() - verification_started
        verified_peak = peak_rss()
        result = {"mode": args.worker_mode, "preparationSeconds": preparation_seconds,
                  "verificationSeconds": verification_seconds, "baselinePeakRss": baseline,
                  "preparedPeakRss": prepared_peak, "verifiedPeakRss": verified_peak,
                  "additionalPreparationPeakBytes": prepared_peak["bytes"] - baseline["bytes"],
                  "additionalVerifiedPeakBytes": verified_peak["bytes"] - baseline["bytes"],
                  "windowDigests": digests, "torchImported": "torch" in sys.modules,
                  "cacheExistedBeforePreparation": cache_existed,
                  "cacheMetadata": cache_metadata}
        if result["torchImported"]:
            raise AssertionError("This data-only benchmark must not import torch")
        print(json.dumps(result, sort_keys=True))
    finally:
        if examples is not None and args.worker_mode != "memory":
            close_windows(examples)


def run_child(mode, corpus, corpus_sha256, tokenizer, cache, args):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-mode", mode,
               "--corpus", str(corpus), "--corpus-sha256", corpus_sha256,
               "--tokenizer", str(tokenizer), "--cache", str(cache),
               "--context-length", str(args.context_length), "--window-policy", args.window_policy]
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(command, check=True, capture_output=True, text=True,
                               env=environment, timeout=args.child_timeout_seconds)
    return json.loads(completed.stdout)


def summarize(measurements):
    keys = ("preparationSeconds", "verificationSeconds", "additionalPreparationPeakBytes",
            "additionalVerifiedPeakBytes")
    result = {key: {"median": statistics.median(item[key] for item in measurements),
                    "min": min(item[key] for item in measurements),
                    "max": max(item[key] for item in measurements)} for key in keys}
    for label in ("baselinePeakRss", "preparedPeakRss", "verifiedPeakRss"):
        values = [item[label]["bytes"] for item in measurements]
        result[label + "Bytes"] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=int, nargs="+", default=[200, 2000], help="Document counts to compare")
    parser.add_argument("--min-text-bytes", type=int, default=2048)
    parser.add_argument("--max-text-bytes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--window-policy", choices=("contiguous_nonoverlapping", "sliding_context"),
                        default="contiguous_nonoverlapping")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--child-timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, help="New JSON evidence file; existing files are never overwritten")
    parser.add_argument("--worker-mode", choices=("memory", "mmap_build", "mmap_reload"), help=argparse.SUPPRESS)
    parser.add_argument("--corpus", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--corpus-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_mode:
        worker(args)
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new file; existing evidence is never overwritten")
    if any(count < 5 or count > 1_000_000 for count in args.records):
        parser.error("--records values must be between 5 and 1000000")
    if not 128 <= args.min_text_bytes <= args.max_text_bytes <= 65536:
        parser.error("Text byte bounds must satisfy 128 <= min <= max <= 65536")
    if not 262 <= args.vocab_size <= 16384 or not 2 <= args.context_length <= 4096:
        parser.error("Vocabulary must be 262..16384 and context length 2..4096")
    if args.repeats < 1 or args.child_timeout_seconds < 1:
        parser.error("Repeats and child timeout must be positive")

    from ia_propia.conversation.data import _json_lines
    from ia_propia.conversation.tokenizer import fit_text_tokenizer

    sources = {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}
    cases = []
    with tempfile.TemporaryDirectory(prefix="unidad-tokenized-data-benchmark-") as temporary:
        directory = Path(temporary)
        counts = sorted(set(args.records))
        fixtures = {}
        for count in counts:
            path = directory / f"synthetic-{count}.jsonl"
            fixtures[count] = (path, fixture(path, count, args.min_text_bytes, args.max_text_bytes, args.seed))
        fitting_started = time.perf_counter()
        fit_path, fit_fixture = fixtures[counts[0]]
        tokenizer = fit_text_tokenizer((record for record in _json_lines(fit_path)
                                        if record["split"] == "train"), args.vocab_size)
        tokenizer_path = directory / "fixed-train-only-tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        tokenizer_info = {"sha256": sha256_file(tokenizer_path),
                          "canonicalSha256": hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest(),
                          "serialization": json.loads(tokenizer.to_str()), "vocabSize": tokenizer.get_vocab_size(),
                          "trainingCorpusSha256": fit_fixture["sha256"],
                          "trainingRecords": fit_fixture["splits"]["train"], "split": "train",
                          "fittingSecondsExcludedFromMeasurements": time.perf_counter() - fitting_started,
                          "policy": "One frozen tokenizer fitted only on TRAIN of the smallest fixture; identical for all children."}
        for count in counts:
            corpus, corpus_info = fixtures[count]
            measurements = {"memory": [], "mmap_build": [], "mmap_reload": []}
            cache_sizes = []
            for repeat in range(args.repeats):
                cache = directory / f"cache-{count}-{repeat}"
                # Alternate order to expose ordinary filesystem/cache timing
                # differences. A reload always follows its matching fresh build.
                modes = ("memory", "mmap_build", "mmap_reload") if repeat % 2 == 0 else (
                    "mmap_build", "mmap_reload", "memory")
                for mode in modes:
                    result = run_child(mode, corpus, corpus_info["sha256"], tokenizer_path, cache, args)
                    result["repeat"] = repeat + 1
                    result["executionOrder"] = list(modes)
                    measurements[mode].append(result)
                cache_sizes.append(sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()))
            expected = measurements["memory"][0]["windowDigests"]
            if any(item["windowDigests"] != expected for samples in measurements.values() for item in samples):
                raise AssertionError("Memory, mmap build and reload produced different ordered input/label windows")
            summaries = {mode: summarize(samples) for mode, samples in measurements.items()}
            memory_peak = summaries["memory"]["verifiedPeakRssBytes"]["median"]
            build_peak = summaries["mmap_build"]["verifiedPeakRssBytes"]["median"]
            reload_peak = summaries["mmap_reload"]["verifiedPeakRssBytes"]["median"]
            cases.append({"fixture": corpus_info, "allWindowsEquivalent": True, "windowDigests": expected,
                          "measurements": measurements, "summaries": summaries,
                          "cacheFileBytes": cache_sizes,
                          "verifiedPeakSavingsVsMemory": {
                              "mmapBuildBytes": memory_peak - build_peak,
                              "mmapBuildFraction": (memory_peak - build_peak) / memory_peak,
                              "mmapReloadBytes": memory_peak - reload_peak,
                              "mmapReloadFraction": (memory_peak - reload_peak) / memory_peak}})

    if sources != {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}:
        raise RuntimeError("Benchmark or data implementation changed during measurement; repeat with stable sources")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    report = {"schemaVersion": 1, "measuredAtUtc": datetime.now(timezone.utc).isoformat(),
              "scope": "Loading, validation and tokenized-window preparation only; no model allocation or training.",
              "fixture": {"dataset": DATASET, "kind": "synthetic-performance-fixture", "containsPrivateData": False,
                          "generation": "Deterministic independent per-document RNG; cases are prefixes of one synthetic corpus.",
                          "validationSplitPolicy": "Every fifth document is VALIDATION; all other documents TRAIN; no TEST."},
              "config": {"records": sorted(set(args.records)), "minTextBytes": args.min_text_bytes,
                         "maxTextBytes": args.max_text_bytes, "seed": args.seed, "requestedVocabSize": args.vocab_size,
                         "contextLength": args.context_length, "windowPolicy": args.window_policy,
                         "objective": "causal_lm", "partitionPolicy": "train_validation_only", "includeTest": False,
                         "repeats": args.repeats, "childTimeoutSeconds": args.child_timeout_seconds,
                         "childEnvironmentOverrides": {"TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}},
              "environment": {"python": sys.version, "implementation": platform.python_implementation(),
                              "platform": platform.platform(), "machine": platform.machine(),
                              "cpuCount": os.cpu_count(),
                              "cpuAffinityCount": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "tokenizers")},
                              "gitCommitBeforeChanges": revision.stdout.strip() if revision.returncode == 0 else None},
              "sourceSha256": sources, "tokenizer": tokenizer_info, "cases": cases,
              "limits": [
                  "Fresh subprocess per sample; ru_maxrss is process peak RSS, not system memory or Python heap size.",
                  "Baseline includes identical imports and loading the fixed tokenizer; increments subtract that existing peak.",
                  "Preparation time includes corpus loading/validation and encoding or cache loading; tokenizer fitting is excluded.",
                  "The memory mode retains the loaded records and all encoded windows, matching its training preparation path.",
                  "Complete sequential verification touches all mmap pages; mapped resident pages contribute to verified peak RSS.",
                  "Build and cache reload are measured separately; OS filesystem caches are uncontrolled and can affect both times.",
                  "Cache storage and system filesystem page cache are not a claim of eliminated memory; mmap shifts storage to files.",
                  "No model, optimizer, batches, training throughput or conversation quality is measured.",
                  "Synthetic documents, fixed vocabulary and window length do not establish a universal RAM or speed guarantee."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
