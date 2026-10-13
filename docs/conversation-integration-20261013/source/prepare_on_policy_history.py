"""Prepare an offline paired history experiment from reviewed fictional data.

Both arms retain the same users, gold final replies, families, folds and order.
Only earlier TRAIN assistant replies differ. Those replies come from the own
teacher with its actual generated history, never from evaluation examples.
Identity and style families remain identical clean anchors in both arms.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation import training


REVIEWED_CORPUS_SHA256 = "c1e9c68335d06ae4f695d327e944332a0923581b827742b2317c2413d39f1336"
C7_ID = "unidad-conversation-bd1975c26297d4665dd95b2ad7d36349c62404a2489b72894d1a67f354a228d2"
ANCHOR_SKILLS = frozenset({"model_identity", "style_language"})
FAMILY_RANK_PREFIX = "unidad-on-policy-history-v1"
TEACHER_FILES = ("manifest.json", "config.json", "tokenizer.json", "weights.safetensors")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def episode_key(row):
    source = row["source"]
    return row["groupId"], source["templateId"], source["counterfactualBranch"]


def select_families(rows, families_per_skill=4):
    if type(families_per_skill) is not int or families_per_skill < 1:
        raise ValueError("families_per_skill must be a positive integer")
    families = defaultdict(set)
    family_traits = {}
    for row in rows:
        if row["split"] == "test":
            continue  # TEST content is never a selection input.
        source = row["source"]
        if (source.get("dataset") != "unidad-grounded-conversations-v1"
                or source.get("kind") != "authored-synthetic"
                or source.get("fictional") is not True
                or source.get("privateConversationUsed") is not False
                or source.get("modelResponsesUsed") is not False):
            raise ValueError("Only the reviewed fictional grounded curriculum is accepted")
        trait = row["split"], source["skill"]
        if row["groupId"] in family_traits and family_traits[row["groupId"]] != trait:
            raise ValueError("A family changes fold or skill")
        family_traits[row["groupId"]] = trait
        families[trait].add(row["groupId"])
    selected = set()
    for (split, skill), groups in sorted(families.items()):
        if len(groups) < families_per_skill:
            raise ValueError(f"Not enough {split}/{skill} families for the declared selection")
        rank = lambda group: hashlib.sha256(
            (FAMILY_RANK_PREFIX + "\0" + split + "\0" + skill + "\0" + group).encode()).hexdigest()
        selected.update(sorted(groups, key=rank)[:families_per_skill])
    if {split for split, _ in families} != {"train", "validation"}:
        raise ValueError("The paired experiment requires TRAIN and validation")
    return [row for row in rows if row["split"] != "test" and row["groupId"] in selected]


def collect_episodes(rows):
    episodes = defaultdict(dict)
    for row in rows:
        if row["split"] != "train" or row["source"]["skill"] in ANCHOR_SKILLS:
            continue
        prefix = row["source"]["supervisedPrefixPairs"]
        if type(prefix) is not int or prefix not in (1, 2, 3) or len(row["messages"]) != 2 * prefix:
            raise ValueError("The reviewed three-exchange prefix contract changed")
        key = episode_key(row)
        if prefix in episodes[key]:
            raise ValueError("Duplicate episode prefix")
        episodes[key][prefix] = row
    for prefixes in episodes.values():
        if set(prefixes) != {1, 2, 3}:
            raise ValueError("Each selected episode must retain every supervised prefix")
        complete = prefixes[3]["messages"]
        if any(row["messages"] != complete[:2 * prefix] for prefix, row in prefixes.items()):
            raise ValueError("An episode's ideal prefixes disagree")
    return episodes


def replace_history(row, generated):
    """Keep gold labels and all users byte-exact; generated history is not a target."""
    if row["split"] != "train" or row["source"]["skill"] in ANCHOR_SKILLS:
        return json.loads(json.dumps(row))
    result = json.loads(json.dumps(row))
    prefix = row["source"]["supervisedPrefixPairs"]
    for exchange in range(prefix - 1):
        result["messages"][2 * exchange + 1]["content"] = generated[exchange]["text"]
    result["source"]["historyPolicy"] = "own_teacher_actual_previous_replies"
    validate_record(result)  # Empty/invalid generations fail, never get invented fillers.
    return result


def source_hashes(corpus, review):
    package = Path(training.__file__).resolve().parent
    return {"builder": training._sha(Path(__file__)), "corpus": training._sha(corpus),
            "review": training._sha(review),
            **{path.name: training._sha(path) for path in sorted(package.glob("*.py"))}}


def prepare(corpus, teacher_model, output, *, families_per_skill=4,
            generation_budget_seconds=600, expected_teacher_model_id=C7_ID):
    corpus, teacher, output = (Path(path).resolve() for path in (corpus, teacher_model, output))
    if (type(generation_budget_seconds) not in (int, float)
            or not math.isfinite(generation_budget_seconds)
            or not 0 < generation_budget_seconds <= 86400):
        raise ValueError("Invalid generation time budget")
    if output.exists():
        raise FileExistsError("Prepared experiments are immutable; choose a new output")
    if training._sha(corpus) != REVIEWED_CORPUS_SHA256:
        raise ValueError("Source differs from the reviewed grounded-v2 curriculum")
    review = corpus.with_suffix(".review.json")
    evidence = training._json_read(review)
    if (evidence.get("datasetSha256") != REVIEWED_CORPUS_SHA256
            or evidence.get("modelScored") is not False
            or evidence.get("scope", {}).get("authoredTemplatesManuallyReviewed") != 44):
        raise ValueError("The source's pre-model grounding review is missing or changed")
    # The inherited review covers all 44 templates: facts are supplied by users;
    # identity is the authored Unidad identity, and style follows user requests.
    # No template asks what an earlier assistant actually suggested. No semantic
    # exclusions or new gold answers are chosen after observing the teacher.
    rows = select_families(load_corpus(corpus), families_per_skill)
    episodes = collect_episodes(rows)
    model, tokenizer, model_config, lineage = training._initial_own_model(teacher, None)
    manifest = training._json_read(teacher / "manifest.json")
    if expected_teacher_model_id is not None and manifest["modelId"] != expected_teacher_model_id:
        raise ValueError("Teacher identity differs from the declared original C7")
    teacher_hashes = {name: training._sha(teacher / name) for name in TEACHER_FILES}
    frozen = source_hashes(corpus, review)
    from ia_propia.conversation.runtime import ConversationEngine
    from ia_propia.conversation.tokenizer import dialogue_tokens
    import torch
    del model
    torch.set_num_threads(1)
    engine = ConversationEngine(teacher)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    os.chmod(stage, 0o700)
    design = {"schemaVersion": 1, "purpose": "paired_offline_on_policy_history_training_not_acceptance",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(), "sourceSha256": frozen,
        "teacherModelId": manifest["modelId"], "teacherFiles": teacher_hashes, "teacherLineage": lineage,
        "familiesPerSkillPerFold": families_per_skill, "familySelection": "SHA256 over fold/skill/family",
        "selectedFamilyIds": sorted({row["groupId"] for row in rows}),
        "anchorSkills": sorted(ANCHOR_SKILLS), "recordSplits": dict(Counter(row["split"] for row in rows)),
        "generation": {"maxNewTokens": 32, "temperature": 0, "seed": 42, "torchThreads": 1,
                       "actualGeneratedHistory": True, "budgetSeconds": generation_budget_seconds},
        "plannedTeacherReplies": 2 * len(episodes), "teacherEvaluationSplitsRead": [],
        "goldGrounding": "Unchanged reviewed source targets; user facts and requests only, plus Unidad identity.",
        "groundingReview": str(review), "goldTargetsRewritten": False, "testEvaluated": False,
        "personalMessagesUsed": False, "externalTeacherUsed": False,
        "historyPolicyOnlyDifference": True, "supervision": "native_last_reply_only",
        "limitations": ["Reviewed procedural templates, not independent natural conversation samples.",
                        "On-policy histories are sampled once from fixed C7; training is offline.",
                        "Inherited manual review is not a new independent semantic audit of every target."]}
    training._atomic_json(stage / "design.json", design)
    calls, generated_cache = 0, {}
    started = time.monotonic()
    raw_path = stage / "teacher-replies-actual.jsonl"
    try:
        with raw_path.open("xb") as raw:
            os.chmod(raw_path, 0o600)
            for key, prefixes in episodes.items():
                ideal = prefixes[3]["messages"]
                actual_history, replies = [], []
                for exchange in range(2):
                    if time.monotonic() - started >= generation_budget_seconds:
                        raise TimeoutError("Generation invocation budget exhausted; actual replies preserved")
                    actual_history.append(dict(ideal[2 * exchange]))
                    request = [dict(message) for message in actual_history]
                    reply = engine.reply(request, max_new_tokens=32, temperature=0, seed=42)
                    calls += 1
                    raw.write(encoded({"episodeKey": list(key), "sourceRecordId": prefixes[exchange + 1]["id"],
                        "exchange": exchange + 1, "inputMessages": request, "actual": reply}))
                    raw.flush()
                    os.fsync(raw.fileno())
                    # This exact text, even when wrong or truncated, becomes
                    # historical context. Its token IDs are retained above.
                    actual_history.append({"role": "assistant", "content": reply["text"]})
                    replies.append(reply)
                    if not reply["text"].strip():
                        raise ValueError("Teacher emitted empty historical text; no substitute was authored")
                    if calls % 32 == 0:
                        training._atomic_json(stage / "progress.json", {"status": "generating",
                            "teacherReplies": calls, "plannedTeacherReplies": design["plannedTeacherReplies"],
                            "elapsedSeconds": time.monotonic() - started})
                        print(json.dumps({"event": "teacher_generation", "replies": calls,
                                          "planned": design["plannedTeacherReplies"]}), flush=True)
                generated_cache[key] = replies

        ideal_rows, generated_rows, changed = [], [], 0
        maxima = {"ideal": 0, "generated": 0}
        for row in rows:
            control = json.loads(json.dumps(row))
            if row["split"] == "train" and row["source"]["skill"] not in ANCHOR_SKILLS:
                control["source"]["historyPolicy"] = "original_reviewed_history"
                variant = replace_history(row, generated_cache[episode_key(row)])
            else:
                variant = json.loads(json.dumps(row))
            if variant["messages"] != row["messages"]:
                changed += 1
            for arm, item in (("ideal", control), ("generated", variant)):
                validate_record(item)
                tokens, _ = dialogue_tokens(item["messages"], tokenizer)
                maxima[arm] = max(maxima[arm], len(tokens) - 1)
                if len(tokens) - 1 > model_config["context_length"]:
                    raise ValueError("Generated context exceeds native context; no data silently discarded")
            # The only target-bearing difference is earlier assistant content.
            if (control["id"] != variant["id"] or control["split"] != variant["split"]
                    or control["messages"][-1] != variant["messages"][-1]
                    or control["messages"][::2] != variant["messages"][::2]):
                raise AssertionError("Paired user, target or identity invariant failed")
            ideal_rows.append(control)
            generated_rows.append(variant)

        if (source_hashes(corpus, review) != frozen
                or any(training._sha(teacher / name) != digest for name, digest in teacher_hashes.items())):
            raise ValueError("Frozen source, teacher or code changed during preparation")
        for name, arm_rows in (("ideal", ideal_rows), ("generated", generated_rows)):
            path = stage / f"{name}.train-validation.jsonl"
            with path.open("xb") as stream:
                for row in arm_rows:
                    stream.write(encoded(row))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(path, 0o600)
            load_corpus(path)
        validation_equal = [row for row in ideal_rows if row["split"] == "validation"] == [
            row for row in generated_rows if row["split"] == "validation"]
        if not validation_equal:
            raise AssertionError("Validation changed between arms")
        training._atomic_json(stage / "progress.json", {"status": "prepared", "teacherReplies": calls})
        report = {**design, "status": "prepared", "teacherReplies": calls,
            "generationSeconds": time.monotonic() - started, "changedTrainRecords": changed,
            "maximumInputTokens": maxima, "validationIdentical": True,
            "trainingFinalTargetsByteExact": True, "trainingUserMessagesByteExact": True,
            "files": {path.name: {"sha256": training._sha(path), "bytes": path.stat().st_size}
                      for path in stage.iterdir() if path.is_file()}}
        training._atomic_json(stage / "manifest.json", report)
        # Publication is one directory rename; failed generations stay in their
        # separate staging directory with all unedited teacher replies.
        import fcntl
        descriptor = os.open(output.parent / f".{output.name}.publish.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if output.exists():
                raise FileExistsError("Another process published this output; staged evidence preserved")
            os.rename(stage, output)
            directory = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.close(descriptor)
        return {"output": str(output), "manifestSha256": training._sha(output / "manifest.json"), **report}
    except BaseException as exc:
        training._atomic_json(stage / "failure.json", {"status": "failed", "reason": type(exc).__name__,
            "error": str(exc), "teacherReplies": calls, "elapsedSeconds": time.monotonic() - started,
            "rawRepliesPreserved": str(raw_path), "testEvaluated": False})
        raise RuntimeError(f"Preparation failed; actual evidence preserved at {stage}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("examples/conversation-grounded-curriculum-v2.jsonl"))
    parser.add_argument("--teacher-model", type=Path, default=Path("models/conversation-reference-c7"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--families-per-skill", type=int, default=4)
    parser.add_argument("--generation-budget-seconds", type=float, default=600)
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.teacher_model, args.output,
        families_per_skill=args.families_per_skill,
        generation_budget_seconds=args.generation_budget_seconds), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
