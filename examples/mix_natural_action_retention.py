"""Byte-preserving rehearsal mix; merging old rows is not new knowledge."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def prepare(output_dir, natural_dir, action_dir):
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Preserve earlier versions; output must be new")
    sources = [(Path(natural_dir), "natural-latam.manifest.admission-v1.json"),
               (Path(action_dir), "manifest.json")]
    blocks, counts, ids, folds, content, source_metadata = [], Counter(), set(), {}, hashlib.sha256(), []
    for directory, manifest_name in sources:
        manifest_path = directory / manifest_name
        manifest = json.loads(manifest_path.read_text())
        path = directory / manifest["file"]["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["corpusSha256"] or not raw.endswith(b"\n"):
            raise ValueError("Pinned input corpus mismatch or missing final newline")
        for line in raw.splitlines():
            row = json.loads(line)
            if row["split"] not in {"train", "validation"} or row["id"] in ids:
                raise ValueError("TEST, unknown fold or duplicate record identity")
            group = row.get("groupId", row["id"])
            if group in folds and folds[group] != row["split"]:
                raise ValueError("Conversation group crosses folds")
            folds[group] = row["split"]
            ids.add(row["id"])
            counts[row["split"]] += 1
            content.update((json.dumps(row["messages"], ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")) + "\n").encode())
        blocks.append(raw)
        source_metadata.append({"path": str(path), "sha256": manifest["corpusSha256"],
                                "manifestPath": str(manifest_path),
                                "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                "sources": manifest["sources"], "role": "rehearsal_not_new_knowledge"})
    output.mkdir(parents=True)
    corpus = output / "natural-action-retention.train-validation.jsonl"
    corpus.write_bytes(b"".join(blocks))
    manifest = {"schemaVersion": 1, "status": "prepared-not-trained-not-accepted",
                "corpusSha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "contentSha256": content.hexdigest(),
                "file": {"path": corpus.name, "bytes": corpus.stat().st_size},
                "splitCounts": {**counts, "test": 0}, "testUsedForTraining": False,
                "sources": source_metadata, "originalRowsByteExact": True,
                "contextLength": 2048, "objective": "assistant_last_reply",
                "newKnowledgeClaim": False, "weightsProduced": False,
                "helperSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--natural-dir", required=True)
    parser.add_argument("--action-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.natural_dir, args.action_dir), ensure_ascii=False))
