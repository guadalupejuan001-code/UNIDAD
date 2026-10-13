#!/usr/bin/env python3
"""Rebuild the attributed Wikcionario corpus from its safe parsed snapshot.

No network, tokenizer, model, private conversation, or evaluation suite is used.
Quoted underlying works excluded by the original extractor are not needed for
this replay. Original revision identifiers/hashes remain descriptive provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time
from urllib.parse import quote

from prepare_spanish_wiktionary import family_key
from ia_propia.conversation.data import assign_split
from ia_propia.conversation.language_data import validate_text_record


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-parsed-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    manifest_path = args.source_dir / "manifest.json"
    parsed_path = args.source_dir / "acquired-pages.jsonl"
    if manifest_path.stat().st_size > 1 << 20 or parsed_path.stat().st_size > 8 << 20:
        raise ValueError("frozen source exceeds replay byte limits")
    for path, expected in ((manifest_path, args.expected_manifest_sha256), (parsed_path, args.expected_parsed_sha256)):
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or file_sha(path) != expected:
            raise ValueError("frozen source identity differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"]["dataset"] != "Wikcionario/es" or manifest["source"]["rightsinfo"]["url"] != "https://creativecommons.org/licenses/by-sa/4.0/deed.es":
        raise ValueError("snapshot is not the recorded original Spanish source")
    pages = [json.loads(line) for line in parsed_path.open()]
    if len(pages) != manifest["records"]:
        raise ValueError("parsed source count differs")
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    for page in pages:
        key = family_key(page["title"])
        if key not in page["links"]:
            raise ValueError("parsed source omits its own lemma relationship")
        for linked in page["links"]:
            union(key, linked)
    records, byte_count, texts, families = [], 0, set(), {}
    for page in pages:
        if time.monotonic() - started > 60:
            raise RuntimeError("offline replay time limit reached")
        group = "wiktionary:es:lexical-family:" + find(family_key(page["title"]))
        record = {"id": f"wiktionary:es:{page['pageId']}:{page['revisionId']}", "groupId": group, "split": assign_split(group),
                  "source": {"dataset": "Wikcionario/es", "kind": "public-wiki-collaborative", "language": "es", "license": "CC-BY-SA-4.0", "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/deed.es",
                             "title": page["title"], "pageId": page["pageId"], "revisionId": page["revisionId"], "revisionTimestamp": page["revisionTimestamp"],
                             "url": "https://es.wiktionary.org/wiki/" + quote(page["title"].replace(" ", "_"), safe=""), "revisionUrl": "https://es.wiktionary.org/w/index.php?oldid=" + str(page["revisionId"]),
                             "historyUrl": "https://es.wiktionary.org/w/index.php?title=" + quote(page["title"].replace(" ", "_"), safe="") + "&action=history",
                             "attribution": "Wikcionario contributors; see the retained revision and page history", "wikitextSha256": page["wikitextSha256"], "spanishSectionSha256": page["sectionSha256"],
                             "extraction": "Spanish grammatical definition lines and uncredited source examples only; conservative inline cleanup; quoted references, translations and unknown templates excluded", "statistics": page["statistics"]}, "text": page["text"]}
        validate_text_record(record)
        digest = hashlib.sha256(record["text"].encode()).hexdigest()
        if digest in texts:
            continue
        raw_bytes = len((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        if byte_count + raw_bytes > manifest["limits"]["corpusBytes"]:
            continue
        texts.add(digest)
        byte_count += raw_bytes
        if group in families and families[group] != record["split"]:
            raise ValueError("lexical family crosses folds")
        families[group] = record["split"]
        records.append(record)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"schemaVersion": 1, "sourceManifestSha256": args.expected_manifest_sha256, "parsedSourceSha256": args.expected_parsed_sha256,
               "networkBytes": 0, "modelLoaded": False, "tokenizerFitted": False, "testUsedForTraining": False, "files": {}}
    for key, filename, selected in (("trainValidation", "wiktionary-es.train-validation.jsonl", [r for r in records if r["split"] != "test"]),
                                    ("test", "wiktionary-es.test.jsonl", [r for r in records if r["split"] == "test"])):
        path = args.output / filename
        with path.open("xb") as stream:
            for record in selected:
                stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode())
        expected = manifest["files"][key]
        actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path), "records": len(selected)}
        if any(actual[item] != expected[item] for item in ("bytes", "sha256", "records")):
            raise ValueError("replayed corpus differs from the frozen source receipt")
        receipt["files"][key] = actual
    receipt.update({"elapsedSeconds": time.monotonic() - started, "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "scriptSha256": file_sha(Path(__file__))})
    (args.output / "replay-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
