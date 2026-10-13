"""Bounded official Tatoeba Spanish ingestion, plain causal text, no models."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import bz2
import hashlib
import io
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
import urllib.request


URL = "https://downloads.tatoeba.org/exports/per_language/spa/"
SOURCES = {
    "spa_sentences_detailed.tsv.bz2": {"bytes": 9502065, "etag": '"6ac9db4c-90fd71"'},
    "spa_sentences_CC0.tsv.bz2": {"bytes": 2266, "etag": '"6ac9db9b-8da"'},
}
LICENSE_URLS = {"CC-BY-2.0-FR": "https://creativecommons.org/licenses/by/2.0/fr/",
                "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/"}
MAX_ROWS = 100000
MAX_CORPUS_BYTES = 24 * 1024 ** 2
MAX_NETWORK_BYTES = 64 * 1024 ** 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def variant_key(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return hashlib.sha256(" ".join(text.split()).encode()).digest()


def download(directory, name, limits):
    target = directory / name
    receipt_path = directory / f"{name}.receipt.json"
    expected = SOURCES[name]
    if target.exists():
        receipt = json.loads(receipt_path.read_text())
        if target.stat().st_size != expected["bytes"] or sha256(target) != receipt["sha256"]:
            raise ValueError("Immutable downloaded source changed")
        return receipt
    request = urllib.request.Request(URL + name, headers={"User-Agent": "Unidad-data-research/1.0",
                                                        "If-Match": expected["etag"]})
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=40) as response, temporary.open("xb") as stream:
            if response.status != 200 or response.headers.get("ETag") != expected["etag"]:
                raise ValueError("Official export changed or download is partial")
            count = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                limits["networkBytes"] += len(block)
                count += len(block)
                if limits["networkBytes"] > MAX_NETWORK_BYTES or count > expected["bytes"]:
                    raise ValueError("Finite network/source budget exceeded")
                stream.write(block)
            if count != expected["bytes"]:
                raise ValueError("Official export has unexpected byte length")
            stream.flush()
            os.fsync(stream.fileno())
            receipt = {"url": URL + name, "status": response.status,
                       "etag": response.headers.get("ETag"),
                       "lastModified": response.headers.get("Last-Modified"),
                       "contentType": response.headers.get("Content-Type"),
                       "bytes": count, "sha256": sha256(temporary)}
        temporary.rename(target)
        dump(receipt_path, receipt)
        return receipt
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main():
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 ** 2, 512 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 ** 2, 256 * 1024 ** 2))
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("600-second ingestion budget")))
    signal.alarm(600)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True
    start = time.monotonic()
    root = Path(__file__).resolve().parents[4]
    native_hashes = {name: sha256(root / "src/ia_propia/conversation" / name)
                     for name in ("data.py", "language_data.py", "contract.py")}
    helper_hash = sha256(Path(__file__))
    sys.path.insert(0, str(root / "src"))
    from ia_propia.conversation.data import assign_split
    from ia_propia.conversation.language_data import validate_text_record
    from ia_propia.conversation.contract import SPECIAL_TOKENS
    base = Path(__file__).resolve().parent
    output = base / "prepared-v1"
    if output.exists():
        raise FileExistsError("Refusing to replace immutable prepared output")
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    limits = {"networkBytes": 0}
    receipts = {name: download(raw, name, limits) for name in SOURCES}
    cc0 = {}
    cc0_rows = 0
    with bz2.open(raw / "spa_sentences_CC0.tsv.bz2", "rt", encoding="utf-8", errors="strict") as stream:
        for line in itertools.islice(stream, 1000):
            cc0_rows += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3 or fields[1] != "spa":
                raise ValueError("Unexpected official CC0 export format")
            cc0[fields[0]] = fields[2]
    if cc0_rows == 1000:
        raise ValueError("CC0 source exceeds declared bounded format probe")
    counts = Counter()
    candidates = {}
    variant_splits = defaultdict(set)
    dates = Counter()
    orphan_cc0 = 0
    detailed_rows = 0
    with bz2.open(raw / "spa_sentences_detailed.tsv.bz2", "rt", encoding="utf-8", errors="strict") as stream:
        for line in itertools.islice(stream, MAX_ROWS - cc0_rows):
            detailed_rows += 1
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
                username = f"anonymous-cc0-sentence-{sentence_id}"
                orphan_cc0 += 1
            if (not text.strip() or len(text.encode()) > 2048
                    or any(token in text for token in SPECIAL_TOKENS)
                    or any(ord(char) < 32 and char not in "\t\n\r" for char in text)):
                counts["excludedTextBoundsOrControl"] += 1
                continue
            group = "tatoeba-spa-author:" + hashlib.sha256(unicodedata.normalize("NFKC", username).casefold().encode()).hexdigest()[:24]
            split = assign_split(group)
            key = variant_key(text)
            variant_splits[key].add(split)
            counts["eligibleSentencesBeforeVariantDedup"] += 1
            if key in candidates:
                counts["normalizedDuplicateSentences"] += 1
                # Lowest numeric ID is the deterministic representative in this shard.
                if int(sentence_id) >= int(candidates[key]["id"]):
                    continue
            candidates[key] = {"id": sentence_id, "text": text, "username": username,
                               "groupId": group, "split": split,
                               "license": "CC0-1.0" if has_cc0 else "CC-BY-2.0-FR",
                               "dateAdded": added, "dateModified": modified}
    buckets = defaultdict(list)
    for key, row in candidates.items():
        if len(variant_splits[key]) > 1:
            counts["excludedCrossfoldVariantClasses"] += 1
            continue
        if row["split"] == "test":
            counts["withheldTestAssignedSentences"] += 1
            continue
        buckets[(row["groupId"], row["username"], row["license"], row["split"])].append(row)
    stage = Path(tempfile.mkdtemp(prefix=".prepared-v1-staging-", dir=base))
    try:
        corpus = stage / "tatoeba-spa.train-validation.jsonl"
        split_records, split_sentences, split_text_bytes = Counter(), Counter(), Counter()
        family_splits, author_credits, text_hashes = {}, {}, set()
        total_bytes = 0
        with corpus.open("xb") as stream:
            for (group, username, license_id, split), sentences in sorted(buckets.items()):
                sentences.sort(key=lambda row: int(row["id"]))
                packs, pack, size = [], [], 0
                for row in sentences:
                    extra = len(row["text"].encode()) + 1
                    if pack and size + extra > 4096:
                        packs.append(pack); pack, size = [], 0
                    pack.append(row); size += extra
                if pack:
                    packs.append(pack)
                anonymous = username.startswith("anonymous-cc0-sentence-")
                profile = None if anonymous else "https://tatoeba.org/en/user/profile/" + urllib.parse.quote(username, safe="")
                if username not in author_credits:
                    author_credits[username] = {"exportUsername": username, "profileUrl": profile,
                                               "exportUsernameIsVerifiedOriginalAuthor": False,
                                               "sentences": 0, "licenses": [],
                                               "family": group, "split": split}
                author_credits[username]["sentences"] += len(sentences)
                author_credits[username]["licenses"].append(license_id)
                for index, rows in enumerate(packs):
                    text = "\n".join(row["text"] for row in rows)
                    digest = hashlib.sha256(text.encode()).digest()
                    if digest in text_hashes:
                        raise ValueError("Exact cross-document text duplicate survived")
                    text_hashes.add(digest)
                    dates.update(row["dateAdded"][:4] for row in rows if row["dateAdded"] not in ("", "\\N"))
                    record = {"id": f"{group}:{license_id}:shard-{index:05d}", "groupId": group,
                              "split": split, "text": text,
                              "source": {"dataset": "Tatoeba/spa-detailed-2026-10-10",
                                         "kind": "public-community-sentences", "language": "es",
                                         "license": license_id, "licenseUrl": LICENSE_URLS[license_id],
                                         "attributionNameAsExported": username, "authorProfileUrl": profile,
                                         "sentenceIds": [row["id"] for row in rows],
                                         "sentenceUrls": ["https://tatoeba.org/en/sentences/show/" + row["id"] for row in rows],
                                         "exportUrl": URL + "spa_sentences_detailed.tsv.bz2",
                                         "exportSha256": receipts["spa_sentences_detailed.tsv.bz2"]["sha256"],
                                         "contentChanged": "Only concatenated intact sentences with newline separators",
                                         "independentHumanOrFactualVerification": False}}
                    validate_text_record(record)
                    raw_record = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    if total_bytes + len(raw_record) > MAX_CORPUS_BYTES:
                        raise ValueError("Fixed 24-MiB corpus budget exceeded; no implicit partial publication")
                    stream.write(raw_record); total_bytes += len(raw_record)
                    split_records[split] += 1
                    split_sentences[split] += len(rows)
                    split_text_bytes[split] += len(text.encode())
                    family_splits[group] = split
        if not split_records["train"] or not split_records["validation"]:
            raise ValueError("Stable author split lacks a usable TRAIN or VALIDATION fold")
        dump(stage / "attribution.json", list(author_credits.values()))
        manifest = {"schemaVersion": 1, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                    "purpose": "New Spanish plain causal source, not active training or acceptance evidence",
                    "files": {corpus.name: {"sha256": sha256(corpus), "bytes": total_bytes,
                                            "records": sum(split_records.values())}},
                    "splitCounts": {"train": split_records["train"], "validation": split_records["validation"], "test": 0},
                    "sentenceCounts": dict(split_sentences), "textBytes": dict(split_text_bytes),
                    "familyCounts": dict(Counter(family_splits.values())),
                    "sourceReceipts": receipts, "sourceRowsScanned": {"detailed": detailed_rows, "cc0": cc0_rows},
                    "boundedShardPolicy": "At most 100000 TSV rows including CC0; detailed export prefix only, not representative full-source or current-fact coverage",
                    "dedupPolicy": "NFKC, casefold, whitespace collapse and Unicode punctuation removal for duplicate keys only. Original sentence bytes preserved. Crossfold variant classes omitted entirely.",
                    "partitionPolicy": "Native assign_split on stable public-export username family; never move an existing family to rebalance",
                    "testUsedForTraining": False, "heldoutTestCorpusPublished": False,
                    "attributionPolicy": "Public username as exported plus original sentence/profile URLs; not independently verified original authorship. Missing attribution excluded unless exact ID+text matched official CC0 export.",
                    "licensingPolicy": "CC BY 2.0 FR official export default; exact CC0 ID+text matches retain CC0 1.0 label",
                    "attributionSha256": sha256(stage / "attribution.json"),
                    "orphanCc0AcceptedBeforeSplit": orphan_cc0, "filters": dict(counts),
                    "retainedSentenceDateAddedYearCounts": dict(sorted(dates.items())),
                    "modelsLoaded": False, "torchImported": "torch" in sys.modules,
                    "tokenizersLoaded": "tokenizers" in sys.modules, "tokenizerFit": False,
                    "ownerConversationOrEvaluationSuitesRead": False,
                    "helperSha256": helper_hash,
                    "nativeSourceHashes": native_hashes,
                    "licenseReview": {"path": str(base / "license-review.json"),
                                      "sha256": sha256(base / "license-review.json")},
                    "resources": {"wallSeconds": time.monotonic() - start,
                                  "cpuSeconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                                  "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                  "networkBytesDownloadedThisInvocation": limits["networkBytes"]},
                    "networkBytesArchivedSourceExports": sum(receipt["bytes"] for receipt in receipts.values()),
                    "hardLimits": {"cpu": 1, "seconds": 600, "addressSpaceBytes": 512 * 1024 ** 2,
                                   "diskBytes": 256 * 1024 ** 2, "networkBytes": MAX_NETWORK_BYTES,
                                   "corpusBytes": MAX_CORPUS_BYTES, "rows": MAX_ROWS},
                    "limitations": ["Community sentences are not certified human-only text or fact-checked world knowledge.",
                                    "No claim of discourse continuity between concatenated independent sentences.",
                                    "A fixed prefix shard has chronological/contributor selection bias.",
                                    "Separate curator must deduplicate against other new/old causal corpora before merging.",
                                    "New data alone does not demonstrate learned language or conversation."]}
        if manifest["torchImported"] or manifest["tokenizersLoaded"]:
            raise ValueError("Forbidden model/tokenizer dependency imported")
        if sha256(Path(__file__)) != helper_hash or any(
                sha256(root / "src/ia_propia/conversation" / name) != digest
                for name, digest in native_hashes.items()):
            raise ValueError("Frozen helper/native schema source changed during ingestion")
        if any(sha256(raw / name) != receipt["sha256"] for name, receipt in receipts.items()):
            raise ValueError("Frozen official export changed during ingestion")
        if sum(path.stat().st_size for path in base.rglob("*") if path.is_file()) > 256 * 1024 ** 2:
            raise ValueError("Finite total disk budget exceeded")
        dump(stage / "manifest.json", manifest)
        stage.rename(output)
        signal.alarm(0)
        print(json.dumps({"output": str(output), "files": manifest["files"],
                          "splitCounts": manifest["splitCounts"], "sentenceCounts": manifest["sentenceCounts"],
                          "resources": manifest["resources"]}, ensure_ascii=False))
    except BaseException:
        shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    main()
