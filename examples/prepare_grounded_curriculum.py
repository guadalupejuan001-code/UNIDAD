"""Merge fictional grounded examples with fixed-fold, quota-limited human data.

Selection uses provenance, complete-dialogue length and stable family hashes.
It never loads model weights, generates responses or scores held-out answers.
Every retained record is unchanged. The manifest is published last as the receipt
for a new, immutable JSONL file; both paths are created exclusively.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer

from ia_propia.conversation.data import CorpusError, SPLITS, load_corpus
from ia_propia.conversation.tokenizer import (dialogue_tokens, integer,
                                             prepare_messages, validate_tokenizer)

KINDS = {"authored-synthetic", "public-human-reviewed"}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(path):
    before = _sha(path)
    records = load_corpus(path)
    if _sha(path) != before:
        raise CorpusError("An input corpus changed while loading")
    return records, {"name": Path(path).name, "sha256": before, "records": len(records)}


def _key(messages):
    return json.dumps(prepare_messages(messages), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _union(original, curriculum):
    """Check the complete union before any length/quota filters can hide conflicts."""
    ids, families, dialogues, prompts = {}, {}, {}, defaultdict(list)
    duplicate_ids = 0
    for record in original + curriculum:
        if record["source"]["kind"] not in KINDS:
            raise CorpusError("Only authored synthetic and reviewed human records are allowed")
        identifier, group, split = record["id"], record["groupId"], record["split"]
        if group in families and families[group] != split:
            raise CorpusError("A conversation family occurs across input splits")
        families[group] = split
        if identifier in ids:
            if ids[identifier] != record:
                raise CorpusError("A record ID has conflicting content or provenance")
            duplicate_ids += 1
            continue
        dialogue = _key(record["messages"])
        if dialogue in dialogues and dialogues[dialogue] != split:
            raise CorpusError("An exact normalized dialogue occurs across splits")
        dialogues[dialogue] = split
        prompts[_key(record["messages"][:-1])].append(record)
        ids[identifier] = record
    ambiguous = [rows for rows in prompts.values() if len({row["split"] for row in rows}) > 1]
    if any(row["source"]["kind"] == "authored-synthetic" for rows in ambiguous for row in rows):
        raise CorpusError("An authored normalized prompt occurs across splits")
    ambiguous_records = {row["id"] for rows in ambiguous for row in rows}
    excluded_groups = {row["groupId"] for rows in ambiguous for row in rows}
    return list(ids.values()), duplicate_ids, ambiguous_records, excluded_groups


def _rank(group):
    return hashlib.sha256(("unidad-grounded-human-quota-v1\0" + group).encode("utf-8")).hexdigest()


def _distribution(values):
    values = sorted(values)
    return ({"count": len(values), "min": values[0], "median": values[len(values) // 2],
             "p95": values[math.ceil(.95 * len(values)) - 1], "max": values[-1]}
            if values else {"count": 0, "min": None, "median": None, "p95": None, "max": None})


def _development_keys(path):
    # The existing validator loads only authored case definitions; no model is loaded.
    from ia_propia.conversation.assessment import load_cases
    before = _sha(path)
    suite, _ = load_cases(path)
    if _sha(path) != before:
        raise CorpusError("Development cases changed while loading")
    return {_key(case["messages"]) for case in suite["cases"]}, before


def _publish(output, manifest, raw, report):
    """Publish each complete file with link(), never replacing existing evidence."""
    temporary_paths, published = [], []
    try:
        for destination, content in ((output, raw), (manifest, (json.dumps(report, ensure_ascii=False,
                sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))):
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, name = tempfile.mkstemp(prefix=".grounded-", dir=destination.parent)
            temporary = Path(name)
            temporary_paths.append(temporary)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if destination == output:
                load_corpus(temporary)
        for destination, temporary in zip((output, manifest), temporary_paths):
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise ValueError("An output was created concurrently; existing evidence is immutable") from exc
            published.append((destination, temporary.stat()))
        for directory in {output.parent, manifest.parent}:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException:
        # If receipt publication fails, remove only files this call actually created.
        for destination, owned in reversed(published):
            try:
                current = destination.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def prepare(input_path, output_path, *, curriculum_path, tokenizer_path, manifest_path=None,
            context_length=256, max_human_train=1000, max_human_validation=125,
            max_human_test=125, exclude_development_cases=None):
    """Preserve folds and records; caps count human records without splitting families.

    All full-fit authored records are retained. Human families are eligible only
    when every member fits, and no member has a prompt shared across folds.
    Hash-ranked eligible families are included only when the whole family fits
    the remaining quota. Later smaller families may fill unused quota space.
    """
    integer(context_length, "context_length", 2, 4096)
    limits = dict(zip(SPLITS, (max_human_train, max_human_validation, max_human_test)))
    for split, value in limits.items():
        integer(value, "max_human_" + split, 0, 1_000_000)
    output = Path(output_path)
    manifest = Path(manifest_path) if manifest_path is not None else Path(str(output) + ".manifest.json")
    inputs = [Path(input_path), Path(curriculum_path), Path(tokenizer_path)]
    if exclude_development_cases is not None:
        inputs.append(Path(exclude_development_cases))
    destinations = [output, manifest]
    if len({path.resolve() for path in destinations + inputs}) != len(destinations + inputs):
        raise ValueError("Inputs, output and manifest must use distinct paths")
    if any(os.path.lexists(path) for path in destinations):
        raise ValueError("Choose new output and manifest paths; existing evidence is immutable")
    original, original_info = _load(input_path)
    curriculum, curriculum_info = _load(curriculum_path)
    if any(record["source"]["kind"] != "authored-synthetic" for record in curriculum):
        raise CorpusError("The new procedural curriculum must contain authored synthetic records")
    records, duplicate_ids, ambiguous_records, excluded_groups = _union(original, curriculum)
    development_info = None
    if exclude_development_cases is not None:
        keys, digest = _development_keys(exclude_development_cases)
        original_ids = {row["id"] for row in original}
        if any(row["id"] not in original_ids and _key(row["messages"][:-1]) in keys for row in curriculum):
            raise CorpusError("A new authored prompt collides with a development case")
        development_info = {"name": Path(exclude_development_cases).name, "sha256": digest,
                            "promptCount": len(keys), "scoresUsed": False}
    tokenizer_file = Path(tokenizer_path)
    with tokenizer_file.open("rb") as stream:
        tokenizer_raw = stream.read(16_777_217)
    if len(tokenizer_raw) > 16_777_216:
        raise CorpusError("Tokenizer JSON exceeds the local byte limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(tokenizer_raw.decode("utf-8")), strict=True)
    tokenizer_hash = hashlib.sha256(tokenizer_raw).hexdigest()
    lengths = {row["id"]: len(dialogue_tokens(row["messages"], tokenizer)[0]) for row in records}
    selected, reasons = [], Counter()
    human = {split: defaultdict(list) for split in SPLITS}
    for row in records:
        if row["source"]["kind"] == "authored-synthetic":
            if lengths[row["id"]] <= context_length + 1:
                selected.append(row)
            else:
                reasons["authored_full_dialogue_exceeds_context"] += 1
        else:
            human[row["split"]][row["groupId"]].append(row)
    human_counts = Counter()
    for split in SPLITS:
        for group in sorted(human[split], key=lambda value: (_rank(value), value)):
            members = human[split][group]
            if group in excluded_groups:
                reasons["human_cross_split_prompt_family"] += len(members)
            elif any(lengths[row["id"]] > context_length + 1 for row in members):
                reasons["human_family_exceeds_context"] += len(members)
            elif human_counts[split] + len(members) > limits[split]:
                reasons["human_quota"] += len(members)
            else:
                selected.extend(sorted(members, key=lambda row: row["id"]))
                human_counts[split] += len(members)
    selected.sort(key=lambda row: (SPLITS.index(row["split"]), row["source"]["kind"], row["groupId"], row["id"]))
    splits = Counter(row["split"] for row in selected)
    if any(not splits[split] for split in SPLITS):
        raise CorpusError("Prepared corpus requires train, validation and test records")
    train_counts, train_targets = Counter(), Counter()
    for row in selected:
        if row["split"] == "train":
            source = row["source"]["dataset"]
            train_counts[source] += 1
            train_targets[source] += len(tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False).ids) + 1
    train_target_count = sum(train_targets.values())
    raw = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for row in selected)
    original_ids = {row["id"] for row in original}
    report = {"schemaVersion": 1, "records": len(selected), "sha256": hashlib.sha256(raw).hexdigest(),
              "splits": {split: splits[split] for split in SPLITS},
              "groupSplits": {split: len({row["groupId"] for row in selected if row["split"] == split}) for split in SPLITS},
              "sources": dict(sorted(Counter(row["source"]["kind"] for row in selected).items())),
              "inputs": [original_info, curriculum_info], "duplicateIdsSkipped": duplicate_ids,
              "oldInputSelectedRecords": sum(row["id"] in original_ids for row in selected),
              "newCurriculumSelectedRecords": sum(row["id"] not in original_ids for row in selected),
              "tokenizer": {"name": tokenizer_file.name, "sha256": tokenizer_hash, "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
              "selection": {"contextLength": context_length, "fullDialogueTokenLimit": context_length + 1,
                            "humanRecordQuotaBySplit": limits,
                            "humanRecordsSelectedBySplit": {split: human_counts[split] for split in SPLITS},
                            "humanFamiliesKeptComplete": True, "humanFamilyRanking": "sha256 unidad-grounded-human-quota-v1 + NUL + groupId",
                            "oldAssignmentsPreserved": True, "dialoguesTruncated": False,
                            "modelResponsesOrEvaluationMetricsUsed": False},
              "reasons": dict(sorted(reasons.items())),
              "crossSplitPromptAudit": {"ambiguousPrompts": len({_key(row["messages"][:-1]) for row in records if row["id"] in ambiguous_records}),
                                        "ambiguousPublicRecords": len(ambiguous_records), "excludedPublicFamilies": len(excluded_groups),
                                        "policy": "Exclude every ambiguous public family; fail on authored collisions; never move folds"},
              "fullDialogueTokensBySplit": {split: _distribution([lengths[row["id"]] for row in selected if row["split"] == split]) for split in SPLITS},
              "trainAudit": {"scope": "train_only", "assistantTargets": "last_reply_only_including_eos", "records": splits["train"],
                             "targetTokens": train_target_count,
                             "byDataset": {source: {"records": train_counts[source], "exampleShare": train_counts[source] / splits["train"],
                                            "targetTokens": train_targets[source], "targetTokenShare": train_targets[source] / train_target_count}
                                           for source in sorted(train_counts)}},
              "developmentExclusion": development_info,
              "limitations": ["Exact normalized dialogue/prompt guards do not identify semantic paraphrases",
                              "Prior corpora and their historical evaluation exposure are unchanged",
                              "Within-fold repeated prompts may remain with their original authored responses"]}
    for path, info in zip((input_path, curriculum_path), report["inputs"]):
        if _sha(path) != info["sha256"]:
            raise CorpusError("An input corpus changed during preparation")
    if _sha(tokenizer_path) != tokenizer_hash:
        raise CorpusError("Tokenizer changed during preparation")
    if development_info and _sha(exclude_development_cases) != development_info["sha256"]:
        raise CorpusError("Development cases changed during preparation")
    _publish(output, manifest, raw, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "curriculum", "tokenizer", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--max-human-train", type=int, default=1000)
    parser.add_argument("--max-human-validation", type=int, default=125)
    parser.add_argument("--max-human-test", type=int, default=125)
    parser.add_argument("--exclude-development-cases", type=Path)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.input, args.output, curriculum_path=args.curriculum, tokenizer_path=args.tokenizer,
                         manifest_path=args.manifest, context_length=args.context_length,
                         max_human_train=args.max_human_train, max_human_validation=args.max_human_validation,
                         max_human_test=args.max_human_test, exclude_development_cases=args.exclude_development_cases)
    except (CorpusError, ValueError, OSError) as exc:
        parser.exit(2, f"prepare_grounded_curriculum: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
