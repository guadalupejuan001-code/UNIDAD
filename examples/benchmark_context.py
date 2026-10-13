"""Finite CPU measurements of context ingestion, indexed retrieval and ranking.

Run from the repository root::

    python examples/benchmark_context.py --records 1000 10000 100000 \
        --repeats 30 --output docs/context-model-20261013/benchmark.json

Each size runs in a fresh child process. Inputs are fictional, batches contain
at most 64 sources and ranking receives at most 48 candidates. No conversation
model is allocated and no user messages are used for training. These timings
measure the context component, not conversation quality or model generation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import re
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BATCH_SIZE = 64
CANDIDATES = 48
SELECTED = 8
MATERIALS = ("madera", "barro", "vidrio", "papel", "algodón", "metal")


class CountingEngine:
    """Explicit deterministic cost fixture: no learned model and no BPE."""

    model_id = "synthetic-context-counting-engine-v1"

    def prompt_usage(self, messages):
        count = sum(2 + len(re.findall(r"\w+|[^\w\s]", message["content"])) for message in messages) + 1
        return {"tokens": count, "capacity": 4096, "discarded": max(0, count - 4096)}


def source(index):
    return {"role": "user", "content": (
        f"El proyectoficticio{index:08d} usa {MATERIALS[index % len(MATERIALS)]}. "
        f"Material del lote de prueba {index:08d}; todos estos datos son ficticios."
    )}


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit_bytes = 1 if sys.platform == "darwin" else 1024
    return {"bytes": int(raw * unit_bytes), "raw": raw,
            "rawUnit": "bytes" if unit_bytes == 1 else "KiB"}


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latencies(values):
    return {"calls": len(values), "p50Milliseconds": statistics.median(values) * 1000,
            "p95Milliseconds": percentile(values, .95) * 1000,
            "minMilliseconds": min(values) * 1000, "maxMilliseconds": max(values) * 1000}


def candidates_from(values):
    return [value if isinstance(value, dict) else value.as_dict() for value in values]


def database_bytes(path):
    return {suffix or "database": candidate.stat().st_size
            for suffix in ("", "-wal", "-shm")
            if (candidate := Path(str(path) + suffix)).exists()}


def measure(args):
    from ia_propia.context.manager import ContextManager, eligible_source
    from ia_propia.context.model import load_ranker
    from ia_propia.context.store import ContextStore

    ranker = load_ranker()
    baseline = peak_rss()
    path = args.database
    store = ContextStore(path, product_id="context-benchmark", session_id="synthetic-session")
    try:
        started = time.perf_counter()
        for start in range(0, args.worker_records, BATCH_SIZE):
            store.append([source(index) for index in range(start, min(start + BATCH_SIZE, args.worker_records))])
        ingestion_seconds = time.perf_counter() - started
        count = store.count()
        if count != args.worker_records:
            raise AssertionError(f"Expected {args.worker_records} sources, stored {count}")
        ingestion_peak = peak_rss()
        queries = [f"proyectoficticio{(round * (count - 1) // max(1, args.repeats - 1)):08d} material"
                   for round in range(args.repeats)]
        # One unmeasured warmup before measuring independent search/rank stages.
        ranker.score_candidates(queries[-1], candidates_from(store.search(queries[-1], top_k=CANDIDATES, max_chars=16384)))
        search_times, rank_times, total_times, hits = [], [], [], []
        candidate_counts = []
        for query in queries:
            started = time.perf_counter()
            values = store.search(query, top_k=CANDIDATES, max_chars=16384)
            after_search = time.perf_counter()
            candidates = candidates_from(values)
            scored = ranker.score_candidates(query, candidates)
            selected = [item for item in sorted(scored, key=lambda item: item["score"], reverse=True)
                        if eligible_source(item)][:SELECTED]
            ended = time.perf_counter()
            search_times.append(after_search - started)
            rank_times.append(ended - after_search)
            total_times.append(ended - started)
            candidate_counts.append(len(candidates))
            hits.append(any(query.split()[0] in item["text"] for item in selected))
        statistics_value = store.stats()
        manager = ContextManager(store, ranker)
        counter = CountingEngine()
        manager_times, manager_hits, restored_hits = [], [], []
        for repeat in range(args.repeats):
            index = repeat * max(0, count - 17) // max(1, args.repeats - 1)
            marker = f"proyectoficticio{index:08d}"
            matching = store.search(marker, top_k=1, max_chars=16384)
            expected_id = matching[0].id if matching else None
            messages = [{"role": "system", "content": "Eres SOVENGI. Usa fuentes originales y conserva sus roles."},
                        {"role": "user", "content": marker + " material"}]
            started = time.perf_counter()
            prepared = manager.prepare(messages, counter, 32)
            manager_times.append(time.perf_counter() - started)
            manager_hits.append(expected_id is not None and expected_id in prepared.metadata["sourceIds"])
            restored_hits.append(any(message["role"] == "user" and message["content"] == source(index)["content"]
                                     for message in prepared.messages))
        manager_measurement = {"latency": latencies(manager_times),
            "engine": "deterministic_regex_count_fixture_excludes_BPE_and_generation",
            "capacityCountingUnits": 4096, "reservedReplyCountingUnits": 32,
            "relevantSourcesSelectedById": sum(manager_hits), "relevantSourcesPreservedInLiveHistory": sum(restored_hits),
            "relevantSourcesAvailable": sum(hit or restored for hit, restored in zip(manager_hits, restored_hits)),
            "queries": len(manager_times), "storedMessagesAfter": store.count(),
            "includesIngestionRetrievalRankingSerializationAndCountFixture": True}
        live_files = database_bytes(path)
        final_peak = peak_rss()
    finally:
        store.close()
    result = {
        "records": count, "batchSize": BATCH_SIZE, "candidateLimit": CANDIDATES,
        "selectionLimit": SELECTED, "rankerModelId": ranker.model_id,
        "rankerParameters": ranker.parameter_count, "ingestionSeconds": ingestion_seconds,
        "ingestionSourcesPerSecond": count / ingestion_seconds,
        "search": latencies(search_times), "rankAndSelect": latencies(rank_times),
        "searchRankAndSelect": latencies(total_times),
        "queriesPerSecond": len(total_times) / sum(total_times),
        "selectedRelevantSources": sum(hits), "queries": len(hits),
        "candidateCountRange": [min(candidate_counts), max(candidate_counts)],
        "managerPrepare": manager_measurement,
        "store": statistics_value, "baselinePeakRss": baseline,
        "ingestionPeakRss": ingestion_peak, "finalPeakRss": final_peak,
        "additionalPeakRssBytes": final_peak["bytes"] - baseline["bytes"],
        "databaseLiveFilesBytes": live_files, "databaseClosedFilesBytes": database_bytes(path),
        "torchImported": "torch" in sys.modules,
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def fallback_equivalence(directory):
    """Require the relevant source in both retrieval paths, not arbitrary order."""
    from ia_propia.context.manager import eligible_source
    from ia_propia.context.model import load_ranker
    from ia_propia.context.store import ContextStore

    ranker = load_ranker()
    modes = []
    for use_fts in (True, False):
        path = directory / ("fts.sqlite3" if use_fts else "scan.sqlite3")
        store = ContextStore(path, product_id="context-equivalence", session_id="synthetic-session", use_fts=use_fts)
        try:
            for start in range(0, 256, BATCH_SIZE):
                store.append([source(index) for index in range(start, start + BATCH_SIZE)])
            relevant = []
            for index in (0, 7, 64, 127, 255):
                query = f"proyectoficticio{index:08d}"
                candidates = candidates_from(store.search(query, top_k=CANDIDATES, max_chars=16384))
                selected = [item for item in sorted(ranker.score_candidates(query, candidates),
                                  key=lambda item: item["score"], reverse=True) if eligible_source(item)][:SELECTED]
                relevant.append([entry["text"] for entry in selected if query in entry["text"]])
            modes.append({"requestedFts": use_fts, "store": store.stats(), "relevant": relevant})
        finally:
            store.close()
    equivalent = modes[0]["relevant"] == modes[1]["relevant"] and all(modes[0]["relevant"])
    if not equivalent:
        raise AssertionError("Indexed and fallback retrieval disagree on relevant synthetic sources")
    return {"records": 256, "queries": 5, "relevantSourcesEquivalent": True,
            "orderingEquivalenceRequired": False,
            "modes": [{key: value for key, value in mode.items() if key != "relevant"} for mode in modes]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-records", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--database", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 5 <= args.repeats <= 200 or any(not 1 <= count <= 1_000_000 for count in args.records):
        parser.error("Use 1..1,000,000 sources per size and 5..200 query repetitions")
    if args.worker_records is not None:
        if not 1 <= args.worker_records <= 1_000_000 or args.database is None:
            parser.error("Worker requires a database and a finite source count")
        measure(args)
        return 0
    if args.output is None or args.output.exists():
        parser.error("Provide a new --output path")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="unidad-context-benchmark-") as temporary:
        directory = Path(temporary)
        samples = []
        for count in args.records:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-records", str(count),
                       "--database", str(directory / f"{count}.sqlite3"), "--repeats", str(args.repeats)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f"Context benchmark worker for {count} sources failed:\n{result.stderr}")
            samples.append(json.loads(result.stdout))
            print(json.dumps({"records": count, "ingestionSourcesPerSecond": samples[-1]["ingestionSourcesPerSecond"],
                              "totalLatency": samples[-1]["searchRankAndSelect"],
                              "managerLatency": samples[-1]["managerPrepare"]["latency"],
                              "ftsEnabled": samples[-1]["store"]["fts_enabled"]}), flush=True)
        equivalence = fallback_equivalence(directory)
    report = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
              "scope": "context_store_retrieval_ranker_no_conversation_generation",
              "fixture": {"kind": "fictional-performance-sources", "generator": "examples/benchmark_context.py",
                          "personalData": False, "trainedOnBenchmark": False},
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "durationSeconds": time.perf_counter() - started,
              "samples": samples, "fallbackEquivalence": equivalence,
              "limitations": ["Single-process CPU measurements; no concurrency or generation throughput claim.",
                              "RSS is a high-water mark from a fresh worker for each corpus size.",
                              "Manager timings use deterministic regex counting units and exclude BPE tokenization and generation.",
                              "Fallback correctness comparison uses 256 sources within its bounded scan window.",
                              "Retrieval of distant sources at scale requires the optional SQLite FTS index."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
