#!/usr/bin/env python3
"""Prepare a bounded, hash-pinned Spanish historical TEXT corpus.

Only original ebook text is a language-model target. Catalog pages, license,
retrieval receipts and editorial metadata remain provenance. All books by an
author share the project's stable family fold, including other public sources.
This helper does not import a tokenizer, generate answers or train a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ia_propia.conversation.data import assign_split
from ia_propia.conversation.language_data import validate_text_record
from ia_propia.conversation.contract import SPECIAL_TOKENS

MIB = 1 << 20
START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}\s*$", re.I | re.M)
END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}\s*$", re.I | re.M)
WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl(path, records):
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    data = path.read_bytes()
    return {"file": path.name, "bytes": len(data), "sha256": sha(data)}


def download(url, budget, max_source_bytes):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.gutenberg.org":
        raise ValueError("Only explicit HTTPS Project Gutenberg source URLs are accepted")
    request = urllib.request.Request(url, headers={"User-Agent": "UNIDAD bounded Spanish corpus preparation"})
    with urllib.request.urlopen(request, timeout=25) as response:
        if urllib.parse.urlparse(response.url).hostname != "www.gutenberg.org":
            raise ValueError("Source redirect left the pinned host")
        chunks, size = [], 0
        while True:
            budget.check()
            block = response.read(65536)
            if not block:
                break
            budget.network_bytes += len(block)
            size += len(block)
            if size > max_source_bytes:
                raise ValueError("Source exceeds per-file byte budget")
            chunks.append(block)
        return b"".join(chunks), {
            "sourceUrl": url, "resolvedUrl": response.url,
            "retrievedAtUtc": datetime.now(timezone.utc).isoformat(),
            "status": response.status,
            "headers": {k: response.headers.get(k) for k in ("Content-Type", "ETag", "Last-Modified")},
        }


class Budget:
    def __init__(self, args, directory):
        self.start = time.monotonic()
        self.args, self.directory = args, directory
        self.network_bytes = 0

    def check(self):
        if time.monotonic() - self.start >= self.args.max_seconds:
            raise RuntimeError("Preparation wall-time budget reached")
        if self.network_bytes > self.args.max_network_mib * MIB:
            raise RuntimeError("Preparation network budget reached")
        if self.directory.exists():
            size = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
            if size > self.args.max_disk_mib * MIB:
                raise RuntimeError("Preparation disk budget reached")


def body_paragraphs(raw, book=None):
    text = raw.decode("utf-8-sig", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    start, end = START.search(text), END.search(text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Missing or ambiguous Gutenberg body boundaries")
    body = text[start.end():end.start()].strip()
    paragraphs, reasons = [], Counter()
    # Some PG Spanish files put a modern transcription preface inside the
    # body markers. Remove that whole initial section before deduplication.
    transcription = re.match(r"\A\s*NOTA(?:S)? DE TRANSCRIPCI[ÓO]N\s*\n.*?(?=\n{4,})", body, re.S | re.I)
    if transcription:
        reasons["initial_transcription_preface"] += 1
        body = body[transcription.end():].strip()
    if book and book.get("bodyStartRegex"):
        selected_start = re.search(book["bodyStartRegex"], body)
        if not selected_start:
            raise ValueError("Pinned original-work start marker was not found")
        reasons["publisher_catalog_and_commentator_preface"] += 1
        body = body[selected_start.start():].strip()
    for raw_paragraph in re.split(r"\n[ \t]*\n+", body):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue
        if len(WORDS.findall(paragraph)) < 12:
            reasons["short_heading_or_fragment"] += 1
        elif any(token in paragraph for token in SPECIAL_TOKENS):
            reasons["reserved_control_token"] += 1
        elif re.search(r"Project Gutenberg|www\.gutenberg\.org|Transcriber.?s Note", paragraph, re.I):
            reasons["distribution_or_transcriber_boilerplate"] += 1
        elif sum(c.isalpha() for c in paragraph) / max(1, sum(not c.isspace() for c in paragraph)) < .5:
            reasons["low_letter_fraction"] += 1
        else:
            paragraphs.append(paragraph)
    return paragraphs, dict(reasons), {"bodySha256": sha(body.encode()), "bodyBytes": len(body.encode())}


def chunks(paragraphs, byte_limit):
    current = []
    size = 0
    for paragraph in paragraphs:
        # A very long paragraph is split at whitespace, without changing words.
        pieces = [paragraph]
        if len(paragraph.encode()) > byte_limit:
            pieces, words, used = [], [], 0
            for word in paragraph.split():
                amount = len(word.encode()) + bool(words)
                if words and used + amount > byte_limit:
                    pieces.append(" ".join(words)); words, used = [], 0
                if len(word.encode()) > byte_limit:
                    raise ValueError("Unsplit source word exceeds the record budget")
                words.append(word); used += len(word.encode()) + (len(words) > 1)
            if words:
                pieces.append(" ".join(words))
        for piece in pieces:
            amount = len(piece.encode()) + (2 if current else 0)
            if current and size + amount > byte_limit:
                yield "\n\n".join(current)
                current, size = [], 0
            current.append(piece); size += len(piece.encode()) + (2 if len(current) > 1 else 0)
    if current:
        yield "\n\n".join(current)


def prepare(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir()
    budget = Budget(args, args.output_dir)
    catalog_bytes = args.catalog.read_bytes()
    catalog = json.loads(catalog_bytes)
    if catalog.get("schemaVersion") != 1 or not catalog.get("books"):
        raise ValueError("Expected a pinned bibliography schemaVersion 1")
    books, paragraph_owners = [], defaultdict(set)
    retrievals = []
    for book in catalog["books"]:
        budget.check()
        death = book.get("authorDeathYear")
        old_original = isinstance(death, int) and 0 < death <= 1800
        old_anonymous = death is None and book.get("anonymousOriginal") is True and book["publicationYear"] <= 1600
        if book["language"] != "es" or not (old_original or old_anonymous):
            raise ValueError("Catalog must contain explicitly Spanish, old original authors")
        group = "public-spanish-author:" + book["authorSlug"]
        if not re.fullmatch(r"[a-z0-9-]+", book["authorSlug"]):
            raise ValueError("Author family must use the shared ASCII canonical slug")
        path = args.sources_dir / f"pg{book['ebookId']}.txt" if args.sources_dir else None
        if path is not None:
            raw = path.read_bytes()
            if len(raw) > args.max_source_mib * MIB:
                raise ValueError("Cached source exceeds per-source budget")
            receipt_path = path.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"mode": "cached-source", "sourceUrl": book["textUrl"]}
        else:
            raw, receipt = download(book["textUrl"], budget, args.max_source_mib * MIB)
        if sha(raw) != book["textSha256"]:
            raise ValueError(f"Pinned source digest differs for ebook {book['ebookId']}")
        (raw_dir / f"pg{book['ebookId']}.txt").write_bytes(raw)
        receipt.update({"bytes": len(raw), "sha256": sha(raw), "encoding": "UTF-8"})
        write_json(raw_dir / f"pg{book['ebookId']}.receipt.json", receipt)
        paragraphs, filters, body_receipt = body_paragraphs(raw, book)
        for paragraph in paragraphs:
            paragraph_owners[sha(" ".join(paragraph.casefold().split()).encode())].add(group)
        books.append((book, group, assign_split(group), paragraphs, filters, body_receipt))
        retrievals.append({**book, **body_receipt, "familyId": group, "split": assign_split(group), "paragraphFilters": filters,
                           "retainedParagraphs": len(paragraphs), "rawReceiptSha256": sha((raw_dir / f"pg{book['ebookId']}.receipt.json").read_bytes())})
    # Repeated paragraphs from differently assigned author families could leak.
    excluded = set()
    for owners in paragraph_owners.values():
        if len({assign_split(group) for group in owners}) > 1:
            excluded.update(owners)
    rows, seen_text, repeated_chunks = [], set(), 0
    for book, group, split, paragraphs, filters, body_receipt in books:
        if group in excluded:
            continue
        for number, text in enumerate(chunks(paragraphs, args.chunk_bytes)):
            digest = sha(" ".join(text.casefold().split()).encode())
            if digest in seen_text:
                repeated_chunks += 1
                continue
            seen_text.add(digest)
            record = {"id": f"pg-es:{book['ebookId']}:chunk:{number:05d}", "groupId": group, "split": split,
                      "source": {"dataset": "Project-Gutenberg", "kind": "public-original-historical",
                                 "language": "es", "license": book["license"], "ebookId": book["ebookId"],
                                 "author": book["author"], "authorSlug": book["authorSlug"], "title": book["title"],
                                 "canonicalWork": book["canonicalWork"], "publicationYear": book["publicationYear"],
                                 "textUrl": book["textUrl"], "bibliographyUrl": book["bibliographyUrl"],
                                 "sourceSha256": book["textSha256"], "bodySha256": body_receipt["bodySha256"],
                                 "synthetic": False, "historicalTextNotCurrentAdvice": True}, "text": text}
            validate_text_record(record)
            rows.append(record)
    counts = Counter(row["split"] for row in rows)
    if not all(counts[s] for s in ("train", "validation", "test")):
        raise ValueError("Bibliography did not provide all stable held-out folds")
    if sum(len(row["text"].encode()) for row in rows) > args.max_corpus_mib * MIB:
        raise RuntimeError("Whole selected author corpus exceeds the configured payload budget")
    files = {}
    for split in ("train", "validation", "test"):
        name = f"gutenberg-spanish.{split}.jsonl"
        if split == "test":
            name = "gutenberg-spanish.test-reserved.jsonl"
        files[split] = jsonl(args.output_dir / name, [row for row in rows if row["split"] == split])
    files["trainValidation"] = jsonl(args.output_dir / "gutenberg-spanish.train-validation.jsonl", [row for row in rows if row["split"] != "test"])
    corpus_file = files["trainValidation"]
    manifest = {"schemaVersion": 1, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                "catalogSha256": sha(catalog_bytes), "helperSha256": sha(Path(__file__).read_bytes()),
                "corpusSha256": corpus_file["sha256"], "files": files,
                "splitCounts": {"train": counts["train"], "validation": counts["validation"], "test": 0},
                "reservedTestRecords": counts["test"], "testUsedForTraining": False,
                "tokenizerFitted": False, "evaluationCasesRead": False, "externalModelOutputsIncluded": False,
                "sources": retrievals, "familyPolicy": "public-spanish-author:<canonical-slug>; whole author including all works/variants remains one stable assign_split fold",
                "guards": {"excludedCrossFoldAuthorFamilies": sorted(excluded), "normalizedChunkDuplicatesRemoved": repeated_chunks},
                "distribution": {split: {"records": counts[split], "authors": len({r["groupId"] for r in rows if r["split"] == split}),
                                         "words": sum(len(WORDS.findall(r["text"])) for r in rows if r["split"] == split),
                                         "payloadBytes": sum(len(r["text"].encode()) for r in rows if r["split"] == split)} for split in ("train", "validation", "test")},
                "processing": "UTF-8 strict; PG start/end markers removed; short headings/control-token/boilerplate/low-letter paragraphs removed; record chunks preserve paragraph words; oversized paragraphs split only at whitespace",
                "limits": {"maxSeconds": args.max_seconds, "maxRSSMiB": args.max_rss_mib, "maxDiskMiB": args.max_disk_mib,
                           "maxNetworkMiB": args.max_network_mib, "maxCorpusMiB": args.max_corpus_mib, "cpuAffinityCount": 1},
                "preparation": {"elapsedSeconds": round(time.monotonic() - budget.start, 3), "networkBytesInThisInvocation": budget.network_bytes,
                                "peakRSSMiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)},
                "limitations": ["A bounded historical selection, not all Spanish or contemporary conversation.",
                                 "Stored text is not evidence that a model learned it.", "Archaic style and historical prejudices require modern-language/SFT anchors and real response evaluation.",
                                 "Copyright declarations are PG USA status plus old original authors; keep raw PG license and provenance when distributing."]}
    (args.output_dir / "catalog.json").write_bytes(catalog_bytes)
    write_json(args.output_dir / "manifest.json", manifest)
    budget.check()
    return {"corpusSha256": corpus_file["sha256"], "corpusBytes": corpus_file["bytes"], "distribution": manifest["distribution"], "manifestSha256": sha((args.output_dir / "manifest.json").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sources-dir", type=Path)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--max-disk-mib", type=int, default=256)
    parser.add_argument("--max-network-mib", type=int, default=64)
    parser.add_argument("--max-source-mib", type=int, default=8)
    parser.add_argument("--max-corpus-mib", type=int, default=10)
    parser.add_argument("--chunk-bytes", type=int, default=16000)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 600 or not 1 <= args.max_rss_mib <= 512 or not 1 <= args.max_disk_mib <= 256 or not 1 <= args.max_network_mib <= 64 or not 1 <= args.max_corpus_mib <= 24 or not 256 <= args.chunk_bytes <= 64000:
        parser.error("Preparation budgets exceed the bounded ingestion envelope")
    resource.setrlimit(resource.RLIMIT_AS, (args.max_rss_mib * MIB, args.max_rss_mib * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (args.max_seconds, args.max_seconds))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
