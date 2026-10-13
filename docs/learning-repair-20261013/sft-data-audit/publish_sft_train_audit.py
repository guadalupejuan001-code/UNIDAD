"""Publish aggregate TRAIN audit with precise names; no tokenization or ML."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--measurement-source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite published aggregate audit")
    raw_hash = sha256(args.measurement)
    report = json.loads(args.measurement.read_text())
    if sha256(args.corpus) != report["corpusSha256"]:
        raise ValueError("Corpus differs from measurement")
    if sha256(args.measurement_source) != report["helperSha256"]:
        raise ValueError("Measurement source differs from frozen execution")
    # Extra family-ID accounting only; non-TRAIN messages are never decoded.
    header = re.compile(rb'(?<!\\)"split"\s*:\s*"(train|validation|test)"')
    families = Counter()
    for line in args.corpus.open("rb"):
        matches = list(header.finditer(line))
        if len(matches) != 1:
            raise ValueError("Ambiguous canonical split header")
        if matches[0].group(1) != b"train":
            continue
        record = json.loads(line)
        families[record["groupId"]] += 1
    if sum(families.values()) != report["scope"]["records"]:
        raise ValueError("TRAIN family accounting differs from token audit")
    def rename_keys(section):
        if "exactCompletePrompts" in section:
            section["exactProvidedMessagePrompts"] = section.pop("exactCompletePrompts")
        if "uniqueCompleteDialoguesExact" in section:
            section["uniqueRecordMessageSequencesExact"] = section.pop("uniqueCompleteDialoguesExact")
    rename_keys(report["repetition"])
    for values in report["datasets"].values():
        rename_keys(values)
    report["schemaVersion"] = 2
    report["measurementReceiptSha256"] = raw_hash
    report["publicationSourceSha256"] = sha256(Path(__file__))
    report["trainFamilyAccounting"] = {
        "uniqueFamilies": len(families),
        "familiesByNumberOfSupervisedPrefixes": dict(sorted(Counter(families.values()).items())),
        "sourceFamilyCountsMayOverlap": True,
    }
    report["definitions"] = {
        "exactProvidedMessagePrompts": "Exact messages[:-1], before inserting default system. All audited rows omit system and receive the same native default, so identity classes match native complete prompts here.",
        "uniqueRecordMessageSequencesExact": "Unique full sequences of each supervised-prefix record; these are not independent conversations or families.",
        "completePromptClassesWithDifferentTargets": "Same provided conditioning messages with multiple different replies. Human branches can contain valid alternatives; this count is not a contradiction or label-error diagnosis.",
        "generationPromptTokens": "Native full conditioning sequence including default system and final assistant role marker, before payload.",
        "teacherForcedInputTokens": "Native inputs=tokens[:-1], including final assistant payload during teacher forcing.",
        "supervisedTokenShareIncludingEos": "Corpus mass of positive last-assistant payload+EOS tokens, not a measured gradient share.",
        "firstUserMessagesAcrossPrefixes": "Includes contextual rehearsal across supervised prefixes in the same family.",
        "normalization": "Casefold plus whitespace collapse; no semantic equivalence inferred.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-staging-", dir=args.output.parent))
    try:
        (stage / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        shutil.copyfile(args.measurement, stage / "measurement-receipt-v1.json")
        shutil.copyfile(args.measurement_source, stage / "audit_sft_train.py")
        shutil.copyfile(Path(__file__), stage / "publish_sft_train_audit.py")
        if sha256(args.corpus) != report["corpusSha256"] or sha256(args.measurement) != raw_hash:
            raise ValueError("Frozen input changed during publication")
        stage.rename(args.output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps({"output": str(args.output), "receiptSha256": sha256(args.output / "receipt.json"),
                      "trainFamilyAccounting": report["trainFamilyAccounting"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
