"""Select a small first curriculum without changing family evaluation splits.

Authored examples are explicitly synthetic. Optional OASST input must already be
imported into Unidad's canonical JSONL format. This script never downloads data.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ia_propia.conversation.data import load_corpus


def prepare(demo_path, output_path, *, oasst_path=None, human_groups=500):
    records = load_corpus(demo_path)
    public = load_corpus(oasst_path) if oasst_path else []
    candidates = {}
    for record in public:
        messages = record["messages"]
        if len(messages) > 4 or len(messages[-1]["content"].encode("utf-8")) > 500:
            continue
        if sum(len(message["content"].encode("utf-8")) for message in messages) > 1400:
            continue
        if any("```" in message["content"] or "$$" in message["content"] for message in messages):
            continue
        if any(name in messages[-1]["content"].casefold() for name in
               ("chatgpt", "openai", "openassistant", "open assistant")):
            continue
        candidates.setdefault(record["groupId"], []).append(record)
    selected_groups = sorted(candidates, key=lambda group: hashlib.sha256(group.encode()).digest())[:human_groups]
    for group in selected_groups:
        records.extend(candidates[group])

    # Exact conversations across different families may be copied public text.
    # Keep one, without moving its split or consulting any evaluation scores.
    seen, unique, duplicates = set(), [], 0
    for record in records:
        text = json.dumps(record["messages"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        unique.append(record)
    raw = b"".join((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for record in unique)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError("Choose a new corpus output; an existing training corpus is immutable")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with destination.open("xb") as stream:
        stream.write(raw)
    destination.chmod(0o600)
    verified = load_corpus(destination)
    result = {"schemaVersion": 1, "records": len(verified), "groups": len({row["groupId"] for row in verified}),
              "splits": dict(Counter(row["split"] for row in verified)),
              "sources": dict(Counter(row["source"]["kind"] for row in verified)),
              "sha256": hashlib.sha256(raw).hexdigest(), "exactDuplicatesRemoved": duplicates,
              "selection": {"humanGroupsRequested": human_groups, "humanGroupsSelected": len(selected_groups),
                            "maxPublicTurns": 4, "maxPublicFinalReplyBytes": 500,
                            "maxPublicDialogueBytes": 1400, "codeFencesExcluded": True},
              "splitPolicy": "Preserve existing whole-family splits; never select by evaluation score"}
    if any(not result["splits"].get(split) for split in ("train", "validation", "test")):
        destination.unlink()
        raise ValueError("The first curriculum needs examples in all three evaluation splits")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=Path("examples/conversation-demo.jsonl"))
    parser.add_argument("--oasst-corpus", type=Path)
    parser.add_argument("--human-groups", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.human_groups < 0:
        parser.error("human-groups must be nonnegative")
    print(json.dumps(prepare(args.demo, args.output, oasst_path=args.oasst_corpus,
                             human_groups=args.human_groups), ensure_ascii=False))


if __name__ == "__main__":
    main()
