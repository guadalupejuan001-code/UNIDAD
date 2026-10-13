#!/usr/bin/env python3
"""Freeze short reviewed Spanish OASST dialogues for an own-model experiment.

This helper downloads only pinned public source files when requested. It reuses
the existing review/role/ancestor importer, preserves its historical tree folds,
and never loads model weights, fits a tokenizer, reads owner chats, or consults
model responses or evaluation scores. The caller supplies the existing own BPE
tokenizer solely to enforce declared context and generation length limits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ia_propia.conversation.data import assign_split, import_oasst, load_corpus
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer
from tokenizers import Tokenizer

SOURCES = (
    {"dataset": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
     "file": "2023-04-12_oasst_ready.trees.jsonl.gz", "sha256": "2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7"},
    {"dataset": "OpenAssistant/oasst2", "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
     "file": "2023-11-05_oasst2_ready.trees.jsonl.gz", "sha256": "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b"},
)
DOCUMENTS = (
    (0, "LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    (0, "README.md", "68483ac2fcc2f3f7779f453352363827678070ef35b6d746ccb2ca6958540fff"),
    (1, "README.md", "52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31"),
)
TOKENIZER_SHA256 = "fca6e470b737f1dd9b5f785b750be42ab13dd1b121aef1dda8b6b5f57d1ea60d"

IDENTITY_RULE_VERSION = "external-assistant-self-identity-v1"
_BRAND = r"(?:open[\s_-]*assistant|chat[\s_-]*gpt)"
_CREATOR = rf"(?:open[\s_-]*ai|{_BRAND})"
_MODEL = (r"(?:un(?:a)?\s+)?(?:modelo(?:\s+de\s+(?:lenguaje(?:\s+natural)?|inteligencia\s+artificial|ia))?"
          r"|asistente(?:\s+virtual)?|inteligencia\s+artificial|ia)"
          r"(?:\s+(?:ai|ia|de\s+ia))?(?:\s+de\s+(?:c[oó]digo\s+abierto|gran\s+escala))?")
_CREATED = r"(?:cread[oa]|desarrollad[oa]|entrenad[oa]|diseñad[oa])"
_CREATED_BY = rf"{_CREATED}(?:\s+y\s+{_CREATED})?\s+por\s+(?:(?:el\s+equipo|la\s+comunidad|la\s+empresa)\s+de\s+)?{_CREATOR}\b"
_FIRST_PERSON = r"(?:soy|estoy|puedo|tengo|ofrezco|respondo|me\s+llamo|mi\s+objetivo|no\s+(?:tengo|puedo))\b"
_SELF_PATTERNS = (
    ("named-external-assistant", re.compile(rf"(?:yo\s+)?(?:soy|me\s+llamo|mi\s+nombre\s+es)\s+(?:el\s+|la\s+)?{_BRAND}\b", re.I)),
    ("model-of-external-creator", re.compile(rf"(?:yo\s+)?soy\s+{_MODEL}\s+(?:(?:de|llamad[oa]|conocid[oa]\s+como)\s+{_CREATOR}\b|{_CREATED_BY})", re.I)),
    ("first-person-external-creator", re.compile(rf"(?:yo\s+)?(?:fui|he\s+sido|estoy|soy)\s+{_CREATED_BY}", re.I)),
    ("as-external-assistant", re.compile(rf"como\s+(?:{_BRAND}|{_MODEL}\s+(?:de\s+{_CREATOR}\b|{_CREATED_BY}))\s*,\s*{_FIRST_PERSON}", re.I)),
)
_GREETING = re.compile(r"^(?:hola(?:\s+a\s+todos)?|buenas(?:\s+tardes|\s+noches)?|lo\s+siento|claro(?:\s+que\s+s[ií])?|por\s+supuesto|pero|adem[aá]s|gracias)[\s,!¡:]+", re.I)
_QUOTED = re.compile(r'«[^»]*»|“[^”]*”|‘[^’]*’|"[^"]*"|\x27[^\x27]*\x27')
_FENCED_CODE = re.compile(r"(?ms)^[ \t]*(```|~~~)[^\n]*\n.*?(?:^[ \t]*\1[^\n]*(?:\n|$)|\Z)")


def foreign_self_identity(text):
    """Return an explicit external self-claim, leaving source text unchanged.

    Matching starts at a sentence (optionally after a listed greeting), not at
    an arbitrary brand mention. Quotes, Markdown quotations and code are masked
    for inspection only. Negated identity and explanatory comparisons are not
    positive claims. This finite Spanish rule does not understand every possible
    paraphrase, indirect assertion or roleplay context.
    """
    analysis = unicodedata.normalize("NFKC", text)
    mask = lambda match: " " * len(match.group())
    analysis = _FENCED_CODE.sub(mask, analysis)
    analysis = re.sub(r"(?m)^(?: {4}|\t)[^\n]*", mask, analysis)
    analysis = re.sub(r"(?m)^[ \t]*>[^\n]*", mask, analysis)
    analysis = re.sub(r"(`+)[^`]*\1", mask, analysis)
    def quote(match):
        value = match.group()[1:-1]
        # A quoted name in 'Me llamo "ChatGPT"' remains a direct name, while
        # an entire quoted claim such as 'Ella dijo: "Soy ChatGPT"' is masked.
        return value if re.fullmatch(_BRAND, value.strip(), re.I) else mask(match)
    analysis = _QUOTED.sub(quote, analysis)
    analysis = analysis.replace("**", "").replace("__", "")
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", analysis):
        clause = sentence.strip().lstrip("¡!")
        if not clause or clause.startswith("¿"):
            continue
        while match := _GREETING.match(clause):
            clause = clause[match.end():].lstrip("¡! ")
        for reason, pattern in _SELF_PATTERNS:
            if match := pattern.match(clause):
                # An identity question is not a positive assertion. A direct
                # claim followed by ', ¿cómo puedo ayudarte?' remains a claim.
                suffix = clause[match.end():].strip()
                if clause.endswith("?") and not suffix.startswith(","):
                    continue
                return reason
    return None


def filter_foreign_self_identity(selected_records, source_records=None):
    """Remove complete selected families implicated by a validated source claim.

    Full-source inspection also catches a long assistant branch excluded by the
    length limit. Applying this guard after the historical selection ensures the
    correction only removes baseline records; it never introduces replacement
    examples, changes folds, rewrites responses or ranks by model performance.
    """
    if source_records is None:
        source_records = selected_records
    families, messages, reasons = set(), set(), Counter()
    for record in source_records:
        for message_id, message in zip(record["source"].get("messageIds", range(len(record["messages"]))), record["messages"]):
            if message["role"] != "assistant":
                continue
            reason = foreign_self_identity(message["content"])
            if reason is not None:
                families.add(record["groupId"])
                key = (record["groupId"], message_id)
                if key not in messages:
                    messages.add(key)
                    reasons[reason] += 1
    kept = [record for record in selected_records if record["groupId"] not in families]
    removed = [record for record in selected_records if record["groupId"] in families]
    return kept, {"ruleVersion": IDENTITY_RULE_VERSION,
                  "policy": "Exclude whole families with an assistant's explicit external self-identity; keep explanations, negations and quoted statements; no text replacement",
                  "ruleMotivation": "Own-model development exposed a source/system identity conflict; the rule is fixed before this new training stage, without final acceptance cases or TEST scores",
                  "sourceFamiliesWithExplicitIdentity": len(families), "sourceAssistantMessagesWithExplicitIdentity": len(messages),
                  "sourceClaimReasons": dict(reasons), "excludedSourceFamilyIds": sorted(families),
                  "selectedBaselineRecords": len(selected_records), "selectedFamiliesExcluded": len({r["groupId"] for r in removed}),
                  "selectedRecordsExcluded": len(removed), "selectedRecordsExcludedBySplit": dict(Counter(r["split"] for r in removed)),
                  "selectedRecordIdsExcluded": [r["id"] for r in removed], "historicalFoldsMoved": 0,
                  "baselineOnlyExclusions": True, "modelScoresUsedToRankOrSelectFamilies": False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_path(directory, source, filename):
    return directory / (source["dataset"].split("/")[-1] + "-" + filename)


def pinned_file(directory, source, filename, expected, download):
    path = raw_path(directory, source, filename)
    url = f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}/{filename}"
    if not path.exists():
        if not download:
            raise ValueError(f"missing pinned source: {path}; pass --download")
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            request = Request(url, headers={"User-Agent": "Unidad-human-dialogues/1.0"})
            with urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
                size = 0
                while block := response.read(1 << 20):
                    size += len(block)
                    if size > 250_000_000:
                        raise ValueError("public source download exceeds byte limit")
                    stream.write(block)
            if sha(temporary) != expected:
                raise ValueError("pinned public source hash mismatch")
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha(path) != expected:
        raise ValueError(f"pinned public source hash mismatch: {path.name}")
    return path, {"dataset": source["dataset"], "revision": source["revision"],
                  "file": filename, "url": url, "sha256": expected, "bytes": path.stat().st_size}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    path.chmod(0o600)


def write_records(path, records):
    with path.open("xb") as stream:
        for record in records:
            stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    path.chmod(0o600)
    loaded = load_corpus(path)
    if loaded != records:
        raise ValueError("output canonical verification differs")
    return {"file": path.name, "sha256": sha(path), "bytes": path.stat().st_size,
            "records": len(records), "families": len({r["groupId"] for r in records}),
            "splits": dict(Counter(r["split"] for r in records))}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalized_messages(record):
    return [{"role": m["role"], "content": " ".join(m["content"].split())}
            for m in record["messages"]]


def crossfold_families(records):
    maps = {key: defaultdict(list) for key in ("prompt", "dialogue", "messageText")}
    for record in records:
        messages = normalized_messages(record)
        maps["prompt"][fingerprint(messages[:-1])].append(record)
        maps["dialogue"][fingerprint(messages)].append(record)
        for message in messages:
            maps["messageText"][fingerprint(message["content"])].append(record)
    groups, receipt = set(), {}
    for key, mapping in maps.items():
        collisions = [values for values in mapping.values() if len({r["split"] for r in values}) > 1]
        affected = {r["groupId"] for values in collisions for r in values}
        groups.update(affected)
        receipt[key] = {"classes": len(collisions), "families": len(affected)}
    return groups, receipt


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "median": values[len(values) // 2],
            "p90": values[int((len(values) - 1) * .9)], "max": values[-1],
            "mean": sum(values) / len(values)}


def _prepare(args):
    if args.max_sequence_tokens < 8 or args.max_assistant_tokens < 1 or args.max_pairs < 1:
        raise ValueError("invalid positive sequence/assistant/pair limits")
    if args.min_assistant_tokens < 1 or args.min_assistant_tokens > args.max_assistant_tokens:
        raise ValueError("invalid assistant token minimum")
    if args.max_records < 1:
        raise ValueError("max_records must be positive")
    if sha(args.tokenizer) != args.tokenizer_sha256:
        raise ValueError("own tokenizer identity differs")
    tokenizer = validate_tokenizer(Tokenizer.from_file(str(args.tokenizer)), strict=True)
    args.raw_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    imported, import_receipts, download_receipts = [], [], []
    for index, source in enumerate(SOURCES):
        raw, receipt = pinned_file(args.raw_directory, source, source["file"], source["sha256"], args.download)
        download_receipts.append(receipt)
        canonical = args.output / f"oasst{index + 1}-importer-intermediate.jsonl"
        imported_receipt = import_oasst(raw, canonical, language="es", revision=source["revision"])
        imported_receipt["actualSourceDataset"] = source["dataset"]
        records = load_corpus(canonical)
        # The historical importer writes oasst1 in descriptive provenance. Its
        # filters/tree IDs/folds are identical for oasst2; correct only dataset
        # metadata in the final union and retain the intermediate hash receipt.
        for record in records:
            record["source"]["dataset"] = source["dataset"]
        imported.append(records)
        import_receipts.append(imported_receipt)
        write_json(canonical.with_suffix(".manifest.json"), imported_receipt)
    for index, filename, expected in DOCUMENTS:
        _, receipt = pinned_file(args.raw_directory, SOURCES[index], filename, expected, args.download)
        download_receipts.append(receipt)

    merged, by_id, duplicate_ids = [], {}, 0
    for records in imported:
        for record in records:
            if record["split"] != assign_split(record["groupId"]):
                raise ValueError("historical family fold differs")
            if record["id"] in by_id:
                previous = by_id[record["id"]]
                if any(record[key] != previous[key] for key in ("groupId", "split", "messages")):
                    raise ValueError("same public source message ID changes content or family")
                if record["source"]["messageIds"] != previous["source"]["messageIds"]:
                    raise ValueError("same public source message ID changes ancestors")
                duplicate_ids += 1
                continue
            by_id[record["id"]] = record
            merged.append(record)
    excluded, before = crossfold_families(merged)
    guarded = [r for r in merged if r["groupId"] not in excluded]
    remaining, after = crossfold_families(guarded)
    if remaining:
        raise ValueError("exact normalized crossfold collision remains")

    encoded_text = {}
    def payload(text):
        if text not in encoded_text:
            encoded_text[text] = len(tokenizer.encode(text, add_special_tokens=False).ids)
        return encoded_text[text]

    eligible, filtered, metrics, seen_dialogues = [], Counter(), {}, set()
    for record in guarded:
        sizes = [payload(m["content"]) for m in record["messages"]]
        assistant = sizes[1::2]
        sequence, mask = dialogue_tokens(record["messages"], tokenizer)
        reason = ("more_than_max_pairs" if len(sizes) // 2 > args.max_pairs else
                  "full_sequence_over_limit" if len(sequence) > args.max_sequence_tokens else
                  "assistant_payload_over_limit" if max(assistant) > args.max_assistant_tokens else
                  "assistant_payload_below_minimum" if min(assistant) < args.min_assistant_tokens else None)
        if reason:
            filtered[reason] += 1
            continue
        key = fingerprint(normalized_messages(record))
        if key in seen_dialogues:
            filtered["exact_normalized_dialogue_duplicate"] += 1
            continue
        seen_dialogues.add(key)
        eligible.append(record)
        metrics[record["id"]] = {"sequenceTokens": len(sequence), "assistantPayloads": assistant,
                                  "supervisedTokens": sum(mask), "pairs": len(sizes) // 2}

    families = defaultdict(list)
    for record in eligible:
        families[record["groupId"]].append(record)
    rank = lambda group: hashlib.sha256(("unidad-human-short-es-v1\0" + group).encode()).hexdigest()
    selected = []
    for group in sorted(families, key=rank):
        selected.extend(families[group])
        if len(selected) >= args.max_records:
            break
    selected, identity_receipt = filter_foreign_self_identity(selected, merged)
    if any(not any(r["split"] == split for r in selected) for split in ("train", "validation", "test")):
        raise ValueError("selection must contain train, validation and reserved test")
    training = [r for r in selected if r["split"] != "test"]
    test = [r for r in selected if r["split"] == "test"]
    train_receipt = write_records(args.output / "human-short-es.train-validation.jsonl", training)
    test_receipt = write_records(args.output / "human-short-es.test-reserved.jsonl", test)
    train_groups, test_groups = ({r["groupId"] for r in records} for records in (training, test))
    if train_groups & test_groups:
        raise ValueError("training and reserved test share a family")
    if sha(args.tokenizer) != args.tokenizer_sha256:
        raise ValueError("tokenizer changed during preparation")
    for source in SOURCES:
        if sha(raw_path(args.raw_directory, source, source["file"])) != source["sha256"]:
            raise ValueError("source changed during preparation")

    by_split = {}
    for split in ("train", "validation", "test"):
        records = [r for r in selected if r["split"] == split]
        values = [metrics[r["id"]] for r in records]
        by_split[split] = {"records": len(records), "families": len({r["groupId"] for r in records}),
                           "byDataset": dict(Counter(r["source"]["dataset"] for r in records)),
                           "pairs": dict(Counter(v["pairs"] for v in values)),
                           "sequenceTokens": distribution([v["sequenceTokens"] for v in values]),
                           "assistantPayloadTokens": distribution([n for v in values for n in v["assistantPayloads"]]),
                           "supervisedTokens": sum(v["supervisedTokens"] for v in values)}
    manifest = {"schemaVersion": 1, "purpose": "bounded natural reviewed Spanish SFT experiment, not conversation acceptance",
                "modelWeightsLoaded": False, "tokenizerFitted": False, "ownerChatsRead": False,
                "modelEvaluationPerformed": False, "testScoresConsulted": False,
                "helperSha256": sha(Path(__file__)), "sources": download_receipts,
                "declaredLicense": "Apache-2.0", "sourceReviews": "Original public review_result=true; not project certification of every fact, grammar or contributor origin",
                "importReceipts": import_receipts,
                "tokenizer": {"sha256": args.tokenizer_sha256, "vocabSize": tokenizer.get_vocab_size(), "purpose": "length measurement only; existing own model BPE"},
                "union": {"records": len(merged), "duplicateSourceIdsRemoved": duplicate_ids, "oldestSourcePreferred": True,
                          "oasst2DatasetMetadataCorrectedOnly": True},
                "guards": {"before": before, "after": after, "wholeFamiliesExcluded": len(excluded),
                           "excludedFamilyIds": sorted(excluded), "recordsExcluded": len(merged) - len(guarded),
                           "historicalFoldsMoved": 0, "normalizedWhitespaceExactGuards": True},
                "selection": {"maxSequenceTokensIncludingBOSRolesSystemAndEOS": args.max_sequence_tokens,
                              "minAssistantPayloadTokens": args.min_assistant_tokens,
                              "maxAssistantPayloadTokens": args.max_assistant_tokens,
                              "maxDialoguePairs": args.max_pairs, "eligibleRecords": len(eligible),
                              "filteredRecords": dict(filtered), "maxRecordsSoftWholeEligibleFamilyLimit": args.max_records,
                              "rank": "sha256(unidad-human-short-es-v1 + NUL + groupId)", "selectedRecords": len(selected),
                              "contentRewritten": False, "branchesRankedByQualityOrModel": False},
                "externalSelfIdentityGuard": identity_receipt,
                "training": train_receipt, "reservedTest": test_receipt, "splits": by_split,
                "limitations": ["Most retained examples are single question/answer pairs, not everyday multi-turn dialogues",
                                "Length filtering favors short answers and excludes long explanations; no model output informed selection",
                                "Upstream positive review does not certify perfect grammar, facts or desired persona",
                                "Exact duplicate guards do not detect semantic paraphrases",
                                "External self-identity is a finite explicit Spanish rule, not exhaustive semantic detection; it does not certify factual quality or improve model behavior by itself",
                                "Heldout historical folds remain fixed; reservation is not a claim of no exposure in every prior experiment",
                                "Tokenizing TEST here measures only declared ingestion lengths; training reads the separate train/validation file",
                                "This corpus is public text and excludes owner messages; it does not prove conversation quality"]}
    write_json(args.output / "human-short-es.manifest.json", manifest)
    return manifest


def prepare(args):
    """Publish a complete directory; failure removes all prepared intermediates.

    A sibling lock serializes helper publications targeting the same output.
    Previously downloaded pinned raw files remain reusable after a failure.
    """
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = target.with_name("." + target.name + ".preparation-lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.close(descriptor)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output already exists: {target}")
        with tempfile.TemporaryDirectory(prefix=".human-dialogues-", dir=target.parent) as temporary:
            staged = argparse.Namespace(**vars(args))
            staged.output = Path(temporary) / "prepared"
            manifest = _prepare(staged)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"output appeared during preparation: {target}")
            staged.output.rename(target)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA256)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-sequence-tokens", type=int, default=513)
    parser.add_argument("--min-assistant-tokens", type=int, default=1)
    parser.add_argument("--max-assistant-tokens", type=int, default=95)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=5000)
    args = parser.parse_args()
    manifest = prepare(args)
    print(json.dumps({key: manifest[key] for key in ("training", "reservedTest", "splits", "selection")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
