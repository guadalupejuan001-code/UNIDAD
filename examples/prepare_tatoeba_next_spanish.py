"""Prepare a later, bounded official Spanish Tatoeba shard without ML imports.

Sources must be the already archived, pinned October 2026 exports. No network,
training, tokenizer fitting, test-suite access, queue mutation or active-run edit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import bz2
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import tempfile
import time
import unicodedata
import urllib.parse


EXPORT_URL = "https://downloads.tatoeba.org/exports/per_language/spa/"
EXPORTS = {
    "spa_sentences_detailed.tsv.bz2": {
        "bytes": 9502065,
        "sha256": "3141acd2354ecbb65199268ec222f680a232ca494129ba77b53a16b3890a11e0",
    },
    "spa_sentences_CC0.tsv.bz2": {
        "bytes": 2266,
        "sha256": "e8cecab4e084fa4d9843f66c1dab90c5747057e0bc63a758c406235297b636c7",
    },
}
PREVIOUS_SHA = "5b4c7e6f46b667fc41c0088f0265f09a0d81aa0ff5d178f7401f276bc6014e79"
LICENSES = {
    "CC-BY-2.0-FR": "https://creativecommons.org/licenses/by/2.0/fr/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}
MAX_CORPUS_BYTES = 20 * 1024 ** 2
MAX_DISK_BYTES = 128 * 1024 ** 2
SPLIT_BITS = {"train": 1, "validation": 2}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duplicate_keys(text):
    """Conservative mechanical keys; never substitute these keys for targets."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(" " if unicodedata.category(char).startswith("P") else char
                         for char in normalized)
    normalized = " ".join(normalized.split())
    unaccented = "".join(char for char in unicodedata.normalize("NFKD", normalized)
                         if not unicodedata.combining(char))
    words = unaccented.split()
    numbered = " ".join("<number>" if word.isdecimal() else word for word in words) if len(words) >= 4 else unaccented
    return tuple(hashlib.sha256((prefix + value).encode("utf-8")).digest()
                 for prefix, value in (("normalized\0", normalized),
                                       ("unaccented\0", unaccented),
                                       ("numbered\0", numbered)))


def family(username):
    return "tatoeba-spa-author:" + hashlib.sha256(
        unicodedata.normalize("NFKC", username).casefold().encode("utf-8")
    ).hexdigest()[:24]


def establish_limits():
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 ** 2, 512 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_DISK_BYTES, MAX_DISK_BYTES))
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("600-second preparation budget")))
    signal.alarm(600)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--previous-corpus", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--license-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-count", type=int, default=150000)
    args = parser.parse_args()
    if not 1 <= args.row_count <= 150000:
        raise ValueError("New detailed-row cap must be between 1 and 150000")
    establish_limits()
    start = time.monotonic()
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Refusing to replace immutable prepared output")
    native_hashes = {name: sha256(root / "src/ia_propia/conversation" / name)
                     for name in ("data.py", "language_data.py", "contract.py")}
    helper_hash = sha256(Path(__file__))
    sys.path.insert(0, str(root / "src"))
    from ia_propia.conversation.data import assign_split
    from ia_propia.conversation.language_data import validate_text_record
    from ia_propia.conversation.contract import SPECIAL_TOKENS

    previous_manifest_hash = sha256(args.previous_manifest)
    previous_manifest = json.loads(args.previous_manifest.read_text(encoding="utf-8"))
    if sha256(args.previous_corpus) != PREVIOUS_SHA:
        raise ValueError("Prior immutable Tatoeba TRAIN/VALIDATION shard changed")
    if previous_manifest["files"][args.previous_corpus.name]["sha256"] != PREVIOUS_SHA:
        raise ValueError("Previous manifest does not identify the pinned corpus")
    offset = previous_manifest["sourceRowsScanned"]["detailed"]
    if offset != 99963:
        raise ValueError("Unexpected previous raw-row boundary")
    license_review_hash = sha256(args.license_review)
    receipts = {}
    for name, expected in EXPORTS.items():
        path = args.raw_dir / name
        receipt = json.loads((args.raw_dir / (name + ".receipt.json")).read_text(encoding="utf-8"))
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Pinned official source changed")
        if receipt["sha256"] != expected["sha256"] or receipt["bytes"] != expected["bytes"] or receipt["url"] != EXPORT_URL + name:
            raise ValueError("Official source receipt does not match its bytes")
        receipts[name] = receipt

    previous_keys, previous_families, previous_ids = set(), {}, set()
    previous_sentences = 0
    with args.previous_corpus.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["split"] not in SPLIT_BITS:
                raise ValueError("Prior corpus unexpectedly contains TEST")
            username = record["source"]["attributionNameAsExported"]
            group = family(username)
            if record["groupId"] != group or assign_split(group) != record["split"]:
                raise ValueError("Native stable author partition differs from previous shard")
            old = previous_families.setdefault(group, record["split"])
            if old != record["split"]:
                raise ValueError("Previous author family appears in multiple folds")
            sentences = record["text"].split("\n")
            if len(sentences) != len(record["source"]["sentenceIds"]):
                raise ValueError("Prior sentence-level provenance cannot be recovered")
            previous_ids.update(record["source"]["sentenceIds"])
            for text in sentences:
                previous_keys.update(duplicate_keys(text))
                previous_sentences += 1

    cc0 = {}
    cc0_rows = 0
    with bz2.open(args.raw_dir / "spa_sentences_CC0.tsv.bz2", "rt", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            cc0_rows += 1
            if cc0_rows > 1000:
                raise ValueError("CC0 export exceeds its finite row guard")
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4 or fields[1] != "spa":
                raise ValueError("Unexpected official CC0 four-column format")
            cc0[fields[0]] = fields[2]
    counts, dates, candidate_masks = Counter(), Counter(), {}
    rows, authors = [], {}
    scanned, exhausted = 0, False
    with bz2.open(args.raw_dir / "spa_sentences_detailed.tsv.bz2", "rt", encoding="utf-8", errors="strict") as stream:
        skipped = sum(1 for _ in itertools.islice(stream, offset))
        if skipped != offset:
            raise ValueError("Official source ended before the recorded previous boundary")
        for line in itertools.islice(stream, args.row_count):
            scanned += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 6:
                counts["excludedUnexpectedColumnCount"] += 1
                continue
            sentence_id, language, text, username, added, modified = fields
            if not sentence_id.isdigit() or language != "spa":
                counts["excludedIdOrLanguage"] += 1
                continue
            has_cc0 = cc0.get(sentence_id) == text
            if username in ("", "\\N"):
                if not has_cc0:
                    counts["excludedMissingAttribution"] += 1
                    continue
                username = "anonymous-cc0-sentence-" + sentence_id
            if username not in authors:
                group = family(username)
                split = assign_split(group)
                authors[username] = (username, group, split)
            username, group, split = authors[username]
            # TEST-assigned text is never normalized, bucketed, tokenized or published.
            if split == "test":
                counts["withheldTestAssignedRowsBeforeTextProcessing"] += 1
                continue
            if group in previous_families and previous_families[group] != split:
                raise ValueError("Existing author family would change fold")
            if (not text.strip() or len(text.encode("utf-8")) > 2048
                    or "\n" in text or "\r" in text
                    or any(token in text for token in SPECIAL_TOKENS)
                    or any(ord(char) < 32 for char in text)):
                counts["excludedTextBoundsOrControl"] += 1
                continue
            keys = duplicate_keys(text)
            if sentence_id in previous_ids or any(key in previous_keys for key in keys):
                counts["excludedPriorSentenceIdOrMechanicalVariant"] += 1
                continue
            bit = SPLIT_BITS[split]
            for key in keys:
                candidate_masks[key] = candidate_masks.get(key, 0) | bit
            rows.append((sentence_id, text, username, group, split,
                         "CC0-1.0" if has_cc0 else "CC-BY-2.0-FR", added, modified))
            counts["eligibleNovelSentencesBeforeInternalDedup"] += 1
        # Use TextIO's own buffer; reading its underlying binary buffer would
        # miss already decoded lookahead and could incorrectly report EOF.
        exhausted = stream.readline() == ""

    rows.sort(key=lambda row: int(row[0]))
    buckets, accepted_keys = defaultdict(list), set()
    for row in rows:
        keys = duplicate_keys(row[1])
        if any(candidate_masks[key] == 3 for key in keys):
            counts["excludedNewCrossfoldVariantSentences"] += 1
            continue
        if any(key in accepted_keys for key in keys):
            counts["excludedNewSamefoldMechanicalVariantSentences"] += 1
            continue
        accepted_keys.update(keys)
        buckets[(row[3], row[2], row[5], row[4])].append(row)
    del rows, candidate_masks, accepted_keys, previous_keys
    staging = Path(tempfile.mkdtemp(prefix=".tatoeba-next-staging-", dir=output.parent))
    try:
        corpus = staging / "tatoeba-spa-next.train-validation.jsonl"
        split_records, split_sentences, split_words, split_text_bytes = Counter(), Counter(), Counter(), Counter()
        family_splits, author_credits, text_hashes = {}, {}, set()
        total_bytes = 0
        with corpus.open("xb") as stream:
            for (group, username, license_id, split), sentences in sorted(buckets.items()):
                packs, pack, size = [], [], 0
                for row in sentences:
                    extra = len(row[1].encode("utf-8")) + 1
                    if pack and size + extra > 4096:
                        packs.append(pack)
                        pack, size = [], 0
                    pack.append(row)
                    size += extra
                if pack:
                    packs.append(pack)
                profile = None if username.startswith("anonymous-cc0-sentence-") else "https://tatoeba.org/en/user/profile/" + urllib.parse.quote(username, safe="")
                credit = author_credits.setdefault(username, {
                    "exportUsername": username, "profileUrl": profile,
                    "exportUsernameIsVerifiedOriginalAuthor": False,
                    "sentences": 0, "licenses": [], "family": group,
                    "split": split, "familyPresentInPreviousShard": group in previous_families,
                })
                credit["sentences"] += len(sentences)
                credit["licenses"].append(license_id)
                family_splits[group] = split
                for index, pack_rows in enumerate(packs):
                    text = "\n".join(row[1] for row in pack_rows)
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in text_hashes:
                        raise ValueError("Exact duplicated output document")
                    text_hashes.add(digest)
                    dates.update(row[6][:4] for row in pack_rows if row[6] not in ("", "\\N"))
                    record = {
                        "id": f"{group}:{license_id}:next-rows-{offset}:shard-{index:05d}",
                        "groupId": group, "split": split, "text": text,
                        "source": {
                            "dataset": "Tatoeba/spa-detailed-2026-10-10-next-v1",
                            "kind": "public-community-sentences", "language": "es",
                            "license": license_id, "licenseUrl": LICENSES[license_id],
                            "attributionNameAsExported": username, "authorProfileUrl": profile,
                            "sentenceIds": [row[0] for row in pack_rows],
                            "sentenceUrls": ["https://tatoeba.org/en/sentences/show/" + row[0] for row in pack_rows],
                            "exportUrl": EXPORT_URL + "spa_sentences_detailed.tsv.bz2",
                            "exportSha256": EXPORTS["spa_sentences_detailed.tsv.bz2"]["sha256"],
                            "contentChanged": "Only concatenated intact sentences with newline separators",
                            "independentHumanOrFactualVerification": False,
                        },
                    }
                    validate_text_record(record)
                    raw_record = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                    if total_bytes + len(raw_record) > MAX_CORPUS_BYTES:
                        raise ValueError("Fixed 20-MiB corpus budget exceeded; no partial publication")
                    stream.write(raw_record)
                    total_bytes += len(raw_record)
                    split_records[split] += 1
                    split_sentences[split] += len(pack_rows)
                    split_words[split] += len(text.split())
                    split_text_bytes[split] += len(text.encode("utf-8"))
        if not split_records["train"] or not split_records["validation"]:
            raise ValueError("Author split lacks usable TRAIN or VALIDATION")
        dump(staging / "attribution.json", list(author_credits.values()))
        manifest = {
            "schemaVersion": 1, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
            "purpose": "New Spanish causal text for a future stage; not active training or acceptance evidence",
            "files": {corpus.name: {"sha256": sha256(corpus), "bytes": total_bytes,
                                    "records": sum(split_records.values())}},
            "splitCounts": {"train": split_records["train"], "validation": split_records["validation"], "test": 0},
            "sentenceCounts": dict(split_sentences), "wordCountsWhitespace": dict(split_words),
            "textBytes": dict(split_text_bytes), "familyCounts": dict(Counter(family_splits.values())),
            "familiesSharedWithPriorShard": len(set(family_splits) & set(previous_families)),
            "novelAuthorFamilies": len(set(family_splits) - set(previous_families)),
            "sourceReceipts": receipts,
            "sourceRows": {"previousDetailedPrefixRowsSkipped": offset,
                           "newDetailedRowsScanned": scanned,
                           "newDetailedRowStartInclusiveOneBased": offset + 1,
                           "newDetailedRowEndInclusiveOneBased": offset + scanned,
                           "nextDetailedRowOffset": offset + scanned,
                           "cc0RowsReadForLicensing": cc0_rows,
                           "oneDecodedLineLookaheadUsedOnlyToCheckEof": True,
                           "exportExhausted": exhausted,
                           "remainingRawRowsNotCounted": not exhausted},
            "priorShard": {"corpusSha256": PREVIOUS_SHA, "manifestSha256": previous_manifest_hash,
                           "publishedTrainValidationSentencesUsedOnlyAsDedupSeed": previous_sentences,
                           "retainedPriorSentencesInThisCorpus": 0},
            "dedupPolicy": "Omit any old ID or normalized, unaccented or number-masked variant of previous TRAIN/VALIDATION sentences. New crossfold mechanical classes omitted before samefold lowest-ID selection. Unicode punctuation/case/space and diacritics normalized only in keys; decimal tokens masked only for phrases of >=4 words. This is conservative mechanical dedup, not semantic paraphrase detection.",
            "partitionPolicy": "Native assign_split on same canonical public-export username family as v1; no previous family moves. New TEST-assigned rows skipped before text normalization and publication.",
            "testAssignedTextComparedForDuplicateKeys": False,
            "testUsedForTraining": False, "heldoutTestCorpusPublished": False,
            "ownerConversationOrEvaluationSuitesRead": False,
            "attributionPolicy": "Public username as exported with profile and sentence URLs; not independently certified original authorship. Missing attribution excluded unless official CC0 ID/text matches.",
            "licensingPolicy": "Official CC BY 2.0 FR default; exact CC0 ID/text matches retain CC0 1.0.",
            "attributionSha256": sha256(staging / "attribution.json"),
            "licenseReviewSha256": license_review_hash, "filters": dict(counts),
            "retainedSentenceDateAddedYearCounts": dict(sorted(dates.items())),
            "helperSha256": helper_hash, "nativeSourceHashes": native_hashes,
            "modelsLoaded": False, "torchImported": "torch" in sys.modules,
            "tokenizersLoaded": "tokenizers" in sys.modules, "tokenizerFit": False,
            "trainingExecuted": False, "softwareTestsExecuted": False,
            "resources": {"wallSeconds": time.monotonic() - start,
                          "cpuSeconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                          "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          "networkBytesDownloadedThisInvocation": 0},
            "hardLimits": {"cpu": 1, "seconds": 600, "addressSpaceBytes": 512 * 1024 ** 2,
                           "diskBytes": MAX_DISK_BYTES, "networkBytes": 32 * 1024 ** 2,
                           "corpusBytes": MAX_CORPUS_BYTES, "newDetailedRows": args.row_count},
            "limitations": [
                "This finite later prefix is not the full export or all Spanish; chronological/contributor bias remains.",
                "Snapshot date is not factual freshness. Community text is not certified human-only or fact-checked.",
                "Independent sentences do not form verified continuous dialogues.",
                "Mechanical keys can remove nonidentical phrases and cannot rule out semantic paraphrases or names substituted in templates.",
                "TEST-assigned sentences were not normalized; no exhaustive cross-TEST variant-leak claim is made.",
                "Deduplication with other source corpora remains the future mixture curator's responsibility.",
                "A new corpus does not show that the active model already learned its contents.",
            ],
        }
        if manifest["torchImported"] or manifest["tokenizersLoaded"]:
            raise ValueError("Forbidden ML dependency imported")
        if sha256(Path(__file__)) != helper_hash or any(
                sha256(root / "src/ia_propia/conversation" / name) != digest
                for name, digest in native_hashes.items()):
            raise ValueError("Frozen helper/native schema changed during preparation")
        if sha256(args.previous_corpus) != PREVIOUS_SHA or sha256(args.previous_manifest) != previous_manifest_hash:
            raise ValueError("Frozen prior shard changed during preparation")
        if sha256(args.license_review) != license_review_hash or any(
                sha256(args.raw_dir / name) != expected["sha256"] for name, expected in EXPORTS.items()):
            raise ValueError("Frozen licensing/source inputs changed during preparation")
        if sum(path.stat().st_size for path in staging.rglob("*") if path.is_file()) > MAX_DISK_BYTES:
            raise ValueError("Finite output disk budget exceeded")
        dump(staging / "manifest.json", manifest)
        staging.rename(output)
        signal.alarm(0)
        print(json.dumps({"output": str(output), "files": manifest["files"],
                          "splitCounts": manifest["splitCounts"], "sentenceCounts": manifest["sentenceCounts"],
                          "sourceRows": manifest["sourceRows"], "filters": manifest["filters"],
                          "resources": manifest["resources"]}, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
