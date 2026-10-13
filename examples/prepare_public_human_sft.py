#!/usr/bin/env python3
"""Curate pinned Spanish OASST SFT families; optional own-tokenizer lengths.

Pool mode does not load weights or a tokenizer. With --tokenizer, only TRAIN and
VALIDATION are length-measured. TEST stays a separately copied source holdout,
without tokenization or model evaluation. Source text and historical folds stay
unchanged; rules remove families or overlong prefixes, never invent targets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_human_dialogues as human
import prepare_spanish_causal_base as causal
from ia_propia.conversation.data import assign_split, load_corpus

ACTOR_RULE_VERSION = "unsupported-external-first-person-action-v1"
_EXTERNAL_SEND = r"enviar(?:te)?\s+(?:(?:un|el|tu|su)\s+)?(?:correo(?:\s+electr[oó]nico)?|email|e-mail|mensaje\s+(?:por\s+(?:whatsapp|telegram)|a\s+tu))"
_EXTERNAL_RESERVE = r"reservar\s+(?:(?:un|una|el|la|tu|su)\s+)?(?:vuelo|hotel|billete|boleto|mesa|cita|entrada|habitaci[oó]n|viaje)"
_EXTERNAL_REMINDER = r"programar\s+(?:(?:un|una|el|la|tu|su)\s+)?(?:alarma|recordatorio)"
_EXTERNAL_BROWSE = r"(?:navegar|buscar(?:\s+informaci[oó]n)?)\s+(?:en\s+|por\s+)?(?:internet|la\s+web)"
_EXTERNAL_ACTION = rf"(?:{_EXTERNAL_SEND}|{_EXTERNAL_RESERVE}|{_EXTERNAL_REMINDER}|{_EXTERNAL_BROWSE}|comprar\b|transferir\s+(?:dinero|fondos)|llamar\s+por\s+tel[eé]fono)"
_ACTOR_PATTERNS = (
    ("first-person-external-promise", re.compile(rf"(?:yo\s+)?(?:te\s+)?(?:puedo|voy\s+a|me\s+encargar[eé]\s+de)\s+{_EXTERNAL_ACTION}", re.I)),
    ("first-person-external-future", re.compile(r"(?:yo\s+)?(?:te\s+)?(?:enviar[eé]\s+(?:(?:un|el|tu)\s+)?(?:correo|email|e-mail)|reservar[eé]\s+(?:(?:un|una|el|la)\s+)?(?:vuelo|hotel|mesa|cita)|comprar[eé]\b|transferir[eé]\s+(?:dinero|fondos)|llamar[eé]\s+por\s+tel[eé]fono)", re.I)),
    ("first-person-external-completion", re.compile(r"(?:yo\s+)?(?:ya\s+)?(?:he\s+(?:enviado\s+(?:(?:un|el|tu)\s+)?(?:correo|email|e-mail)|reservado\s+(?:(?:un|una|el|la)\s+)?(?:vuelo|hotel|mesa|cita)|comprado\b|transferido\s+(?:dinero|fondos)|llamado\s+por\s+tel[eé]fono)|envi[eé]\s+(?:(?:un|el|tu)\s+)?(?:correo|email|e-mail)|reserv[eé]\s+(?:(?:un|una|el|la)\s+)?(?:vuelo|hotel|mesa|cita)|compr[eé]\b)", re.I)),
)


def unsupported_external_action(text):
    """Detect narrow positive actor claims, preserving explanations and quotes.

    Canonical source dialogues have no tool-result evidence. The rule excludes
    explicit first-person external operations, not writing help, instructions,
    negations, questions, citations or code. It is finite, not a factual oracle.
    """
    value = unicodedata.normalize("NFKC", text)
    mask = lambda match: " " * len(match.group())
    value = human._FENCED_CODE.sub(mask, value)
    value = re.sub(r"(?m)^(?: {4}|\t)[^\n]*", mask, value)
    value = re.sub(r"(?m)^[ \t]*>[^\n]*", mask, value)
    value = re.sub(r"(`+)[^`]*\1", mask, value)
    value = human._QUOTED.sub(mask, value).replace("**", "").replace("__", "")
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", value):
        clause = sentence.strip().lstrip("¡!")
        if not clause or clause.startswith("¿") or clause.endswith("?"):
            continue
        while match := human._GREETING.match(clause):
            clause = clause[match.end():].lstrip("¡! ")
        for reason, pattern in _ACTOR_PATTERNS:
            if pattern.match(clause):
                return reason
    return None


def curate(args):
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    merged, by_id, receipts = [], {}, []
    for dataset, revision, expected in causal.OASST_INPUTS:
        path = args.oasst_directory / f"{dataset}-importer-intermediate.jsonl"
        if human.sha(path) != expected:
            raise ValueError("pinned OASST canonical input differs")
        original_manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        if original_manifest["sha256"] != expected or original_manifest["source"]["revision"] != revision:
            raise ValueError("source revision/hash differs")
        receipts.append({"dataset": "OpenAssistant/" + dataset, "revision": revision, "license": "Apache-2.0",
                         "canonicalSha256": expected, "rawSourceSha256": original_manifest["sourceSha256"],
                         "sourceUrl": f"https://huggingface.co/datasets/OpenAssistant/{dataset}/tree/{revision}"})
        for record in load_corpus(path):
            if record["split"] != assign_split(record["groupId"]):
                raise ValueError("historical family fold changed")
            record["source"]["dataset"] = "OpenAssistant/" + dataset
            if record["id"] in by_id:
                previous = by_id[record["id"]]
                if any(previous[k] != record[k] for k in ("groupId", "split", "messages")):
                    raise ValueError("same public source ID changes text or family")
                continue
            by_id[record["id"]] = record
            merged.append(record)
    clone_groups, clone_receipt = human.crossfold_families(merged)
    identity_groups, actor_groups, seen_messages, reasons = set(), set(), set(), Counter()
    for record in merged:
        for mid, message in zip(record["source"]["messageIds"], record["messages"]):
            if message["role"] != "assistant" or mid in seen_messages:
                continue
            seen_messages.add(mid)
            identity = human.foreign_self_identity(message["content"])
            actor = unsupported_external_action(message["content"])
            if identity:
                identity_groups.add(record["groupId"])
                reasons["identity:" + identity] += 1
            if actor:
                actor_groups.add(record["groupId"])
                reasons["actor:" + actor] += 1
    removed_groups = clone_groups | identity_groups | actor_groups
    retained, seen_dialogues, duplicates = [], set(), 0
    for record in merged:
        if record["groupId"] in removed_groups:
            continue
        fingerprint = human.fingerprint(human.normalized_messages(record))
        if fingerprint in seen_dialogues:
            duplicates += 1
            continue
        seen_dialogues.add(fingerprint)
        retained.append(record)
    training = [r for r in retained if r["split"] != "test"]
    heldout = [r for r in retained if r["split"] == "test"]
    length_receipt = {"measured": False, "testTokenized": False}
    if args.tokenizer is not None:
        from tokenizers import Tokenizer
        from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer
        tokenizer = validate_tokenizer(Tokenizer.from_file(str(args.tokenizer)), strict=True)
        fingerprint = human.sha(args.tokenizer)
        eligible, rejected, pairs = [], Counter(), Counter()
        for record in training:
            ids, _ = dialogue_tokens(record["messages"], tokenizer)
            if len(record["messages"]) // 2 > args.max_pairs:
                rejected["above_max_pairs"] += 1
            elif len(ids) > args.max_sequence_tokens:
                rejected["above_full_sequence_limit"] += 1
            else:
                eligible.append(record)
                pairs[len(record["messages"]) // 2] += 1
        training = eligible
        if human.sha(args.tokenizer) != fingerprint:
            raise ValueError("own tokenizer changed during preparation")
        length_receipt = {"measured": True, "tokenizerSha256": fingerprint, "vocabSize": tokenizer.get_vocab_size(),
                          "maximumFullSequenceTokens": args.max_sequence_tokens, "maximumPairs": args.max_pairs,
                          "rejectedPrefixes": dict(rejected), "retainedPairs": dict(pairs), "testTokenized": False}
    files = {"trainValidation": human.write_records(args.output / "human-sft.train-validation.jsonl", training),
             "reservedTestSourcePool": human.write_records(args.output / "human-sft.test-reserved-source-pool.jsonl", heldout)}
    report = {"schemaVersion": 1, "purpose": "source-curated public Spanish human SFT, not model acceptance",
              "status": "context_filtered_training_data" if args.tokenizer else "source_pool_pending_new_tokenizer_lengths",
              "sources": receipts, "mergedSourceRecords": len(merged), "oldestSourceForRepeatedIds": True,
              "files": files, "splits": dict(Counter(r["split"] for r in training + heldout)),
              "sourcePairsBySplit": {split: dict(Counter(len(r["messages"]) // 2 for r in retained if r["split"] == split)) for split in ("train", "validation", "test")},
              "guards": {"exactClones": clone_receipt, "exactCloneFamilies": len(clone_groups),
                         "externalIdentityRuleVersion": human.IDENTITY_RULE_VERSION, "externalIdentityFamilies": len(identity_groups),
                         "externalActorRuleVersion": ACTOR_RULE_VERSION, "unsupportedActorFamilies": len(actor_groups),
                         "sourceClaimReasons": dict(reasons), "excludedFamilyIds": sorted(removed_groups),
                         "normalizedDialogueDuplicatesRemoved": duplicates, "wholeFamiliesExcluded": True,
                         "historicalFoldsMoved": 0, "responsesRewritten": False},
              "lengths": length_receipt, "helperSha256": human.sha(Path(__file__)),
              "identityHelperSha256": human.sha(Path(human.__file__)), "modelWeightsLoaded": False,
              "evaluationCasesRead": False, "modelEvaluationPerformed": False, "ownerMessagesRead": False,
              "testScoredOrTokenized": False, "license": "Apache-2.0 source declaration; upstream contributor origin and every fact not certified",
              "limitations": ["Most human source paths are one or two exchanges; availability of longer histories is limited",
                              "Upstream positive review and finite actor/identity rules do not certify every fact or instruction",
                              "Canonical source dialogues contain no tool-result provenance; explicit external self-actions are excluded conservatively",
                              "The separately reserved TEST source pool is not length-filtered or tokenized",
                              "No new personal messages, evaluation targets or external model-generated labels were used"]}
    human.write_json(args.output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasst-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-sequence-tokens", type=int, default=513)
    parser.add_argument("--max-pairs", type=int, default=4)
    args = parser.parse_args()
    report = curate(args)
    print(json.dumps({"status": report["status"], "files": report["files"], "splits": report["splits"],
                      "sourcePairsBySplit": report["sourcePairsBySplit"], "lengths": report["lengths"],
                      "sourceGuardFamilyCounts": {k: report["guards"][k] for k in ("exactCloneFamilies", "externalIdentityFamilies", "unsupportedActorFamilies")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
