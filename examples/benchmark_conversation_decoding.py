"""Finite read-only comparison of the original path, KV cache and NERVA.

This measures execution and preserves real continuations. It does not train,
grade language competence or overwrite an existing report. Run from a checkout
with `PYTHONPATH=src python examples/benchmark_conversation_decoding.py --help`.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of {id, messages}; no reference answers are read")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--torch-threads", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new file")
    raw = args.cases.read_bytes()
    if len(raw) > 1048576:
        parser.error("Cases exceed 1 MiB")
    cases = json.loads(raw)
    if type(cases) is not list or not 1 <= len(cases) <= 40:
        parser.error("Supply between 1 and 40 cases")
    from ia_propia.conversation.runtime import validate_reply_request
    for case in cases:
        if type(case) is not dict or set(case) != {"id", "messages"} or type(case["id"]) is not str:
            parser.error("Each case requires exactly id and messages")
        validate_reply_request({"messages": case["messages"], "max_new_tokens": args.max_new_tokens})
    import torch
    from ia_propia.conversation.artifacts import load_artifact
    from ia_propia.conversation.model import chat
    torch.set_num_threads(args.torch_threads)
    model, tokenizer, manifest = load_artifact(args.model)
    before = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest() for name in manifest["files"]}
    partial = args.output.with_suffix(".partial.jsonl")
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for directory in (Path("src/ia_propia/conversation"), Path("src/ia_propia/assets/nerva"))
                     for path in sorted(directory.iterdir()) if path.suffix in (".py", ".mjs", ".nv")}
    identity = {"modelId": manifest["modelId"], "artifactHashes": before, "sourceHashes": source_hashes,
                "casesSha256": hashlib.sha256(raw).hexdigest(), "maxNewTokens": args.max_new_tokens,
                "repeats": args.repeats, "torchThreads": args.torch_threads,
                "python": platform.python_version(), "torch": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    # Claim a fresh receipt before generating, so concurrent runs cannot share
    # an output. A failed run keeps its completed replies instead of losing them.
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(identity, ensure_ascii=False) + "\n")
    modes = [("torch_reference", "torch", False), ("torch_cache", "torch", True), ("nerva_cache", "nerva", True)]
    # Warm every path equally; timing includes NERVA startup and cleanup per reply.
    for _, backend, cached in modes:
        chat(model, tokenizer, cases[0]["messages"], max_new_tokens=2, temperature=0,
             decoder_backend=backend, use_cache=cached)
    for repeat in range(args.repeats):
        # Rotate order to reduce a systematic warmup/order bias.
        order = modes[repeat % 3:] + modes[:repeat % 3]
        for case in cases:
            for name, backend, cached in order:
                start = time.perf_counter()
                result = chat(model, tokenizer, case["messages"], max_new_tokens=args.max_new_tokens,
                              temperature=0, seed=42, decoder_backend=backend, use_cache=cached)
                row = {"caseId": case["id"], "repeat": repeat, "mode": name,
                       "seconds": time.perf_counter() - start, "result": result}
                records.append(row)
                with partial.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"{repeat + 1}/{args.repeats} {case['id']} {name}: {row['seconds']:.3f}s", flush=True)
    after = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest() for name in manifest["files"]}
    groups = {(row["repeat"], row["caseId"]): row["result"]["generatedTokenIds"] for row in records if row["mode"] == "torch_reference"}
    summary = {}
    for name, _, _ in modes:
        selected = [r for r in records if r["mode"] == name]
        times = [sum(r["seconds"] for r in selected if r["repeat"] == repeat) for repeat in range(args.repeats)]
        summary[name] = {"replies": len(selected), "medianSuiteSeconds": statistics.median(times),
            "suiteSeconds": times, "allTokenIdsMatchReference": all(r["result"]["generatedTokenIds"] == groups[(r["repeat"], r["caseId"])] for r in selected),
            "tokenBudgetTerminations": sum(r["result"]["stopReason"] == "token_budget" for r in selected)}
    for item in summary.values():
        item["speedupVsReference"] = summary["torch_reference"]["medianSuiteSeconds"] / item["medianSuiteSeconds"]
    report = {"schemaVersion": 1, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "purpose": "execution_comparison_not_semantic_acceptance", "modelId": manifest["modelId"],
        "modelManifest": manifest, "casesSha256": hashlib.sha256(raw).hexdigest(), "cases": cases,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                        "node": subprocess.check_output(["node", "--version"], text=True).strip(), "torchThreads": args.torch_threads},
        "settings": {"temperature": 0, "seed": 42, "maxNewTokens": args.max_new_tokens, "repeats": args.repeats},
        "timingScope": "Loaded model; complete reply including NERVA worker startup, IPC, sampling and cleanup",
        "sourceHashes": source_hashes,
        "artifactHashesBefore": before, "artifactHashesAfter": after, "artifactsUnchanged": before == after,
        "summary": summary, "outputs": records, "generalConversationValidated": False, "weightsUpdated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    return 0 if before == after and all(s["allTokenIdsMatchReference"] for s in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
