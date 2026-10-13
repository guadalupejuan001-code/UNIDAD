#!/usr/bin/env python3
"""Bounded original Spanish public-domain essays/grammar from Wikisource.

Discover actual public revisions, freeze their wikitext, strip structural wiki
markup, preserve author-family folds and publish TRAIN/VALIDATION only. No
models, tokenizer, owner chat, or evaluation targets are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import tempfile
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ia_propia.conversation.data import assign_split
from ia_propia.conversation.language_data import validate_text_record

MIB = 1 << 20
API = "https://es.wikisource.org/w/api.php"
WORKS = [
    {"title": "Teatro crítico universal", "author": "Benito Jerónimo Feijoo",
     "authorSlug": "benito-jeronimo-feijoo", "deathYear": 1764},
    {"title": "Gramática de la lengua castellana destinada al uso de los americanos",
     "author": "Andrés Bello", "authorSlug": "andres-bello", "deathYear": 1865},
    {"title": "La mujer del porvenir", "author": "Concepción Arenal",
     "authorSlug": "concepcion-arenal", "deathYear": 1893},
    {"title": "La educación de la mujer", "author": "Concepción Arenal",
     "authorSlug": "concepcion-arenal", "deathYear": 1893},
    {"title": "La instrucción del obrero", "author": "Concepción Arenal",
     "authorSlug": "concepcion-arenal", "deathYear": 1893},
]
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
FUNCTIONS = set("de la el los las y en del que es un una por con para su se al como pero este esta sus sobre entre también fue son lo o a le si cuando sin desde donde no cual esto estas estos esa ese esos otras otro otra hasta hay han hacer ser tiene están está eran era uno cada todo todos toda todas puede pueden después antes durante aunque porque sino mismo misma".split())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def balanced_drop(text, opening, closing):
    """Remove wiki templates/tables without expanding external dependencies."""
    output, depth, cursor = [], 0, 0
    while cursor < len(text):
        if text.startswith(opening, cursor):
            depth += 1
            cursor += len(opening)
        elif depth and text.startswith(closing, cursor):
            depth -= 1
            cursor += len(closing)
        else:
            if not depth:
                output.append(text[cursor])
            cursor += 1
    if depth:
        raise ValueError("unbalanced source markup")
    return "".join(output)


def clean_wikitext(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", "", text, flags=re.S | re.I)
    text = balanced_drop(balanced_drop(text, "{{", "}}"), "{|", "|}")
    def link(match):
        target, *labels = match.group(1).split("|")
        if target.casefold().lstrip(":").startswith(("categoría:", "category:", "archivo:", "file:", "imagen:", "image:")):
            return ""
        return labels[-1] if labels else target.lstrip(":")
    text = re.sub(r"\[\[([^\[\]]+)\]\]", link, text)
    text = re.sub(r"\[https?://\S+(?:\s+([^\]]+))?\]", lambda m: m.group(1) or "", text)
    text = re.sub(r"<(?:br|hr)\b[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = html.unescape(text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:={1,6}|[*#:;]+)\s*", "", line)
        line = re.sub(r"\s*={1,6}\s*$", "", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def chunks(text, cap=65536):
    result, current = [], ""
    for paragraph in text.split("\n\n"):
        if len(paragraph.encode()) > cap:
            raise ValueError("paragraph exceeds canonical chunk bound")
        candidate = current + ("\n\n" if current else "") + paragraph
        if len(candidate.encode()) > cap:
            result.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        result.append(current)
    return result


class Collector:
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()
        self.deadline = self.started + 590
        self.network_bytes = 0
        self.receipts = []
        self.helper_sha256 = sha(Path(__file__).read_bytes())
        (output / "raw").mkdir()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("600-second ingestion limit approached")
        if sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) > 512 * MIB:
            raise ValueError("512-MiB ingestion store limit")

    def request(self, parameters):
        self.check()
        query = {**parameters, "format": "json", "formatversion": 2}
        url = API + "?" + urlencode(query)
        if self.network_bytes + 8 * MIB > 128 * MIB:
            raise ValueError("128-MiB response budget")
        time.sleep(1.5)
        for attempt in range(4):
            self.check()
            try:
                with urlopen(Request(url, headers={"User-Agent": "Unidad-public-Spanish-corpus/1.0"}), timeout=min(30, max(1, self.deadline - time.monotonic()))) as response:
                    raw = response.read(8 * MIB + 1)
                    status = response.status
                break
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                delay = min(30, int(exc.headers.get("Retry-After", "5")) if exc.headers.get("Retry-After", "5").isdigit() else 5)
                print(json.dumps({"stage": "source_rate_limit", "status": exc.code, "retryAfterSeconds": delay}), flush=True)
                time.sleep(delay)
        self.network_bytes += len(raw)
        if len(raw) > 8 * MIB:
            raise ValueError("source response exceeds8MiB")
        value = json.loads(raw)
        if "error" in value:
            raise ValueError("public API rejected source request: " + value["error"].get("code", "unknown"))
        name = "source-" + str(len(self.receipts)).zfill(3) + ".json"
        (self.output / "raw" / name).write_bytes(raw)
        self.receipts.append({"file": "raw/" + name, "url": url, "sha256": sha(raw), "bytes": len(raw), "status": status})
        return value

    def revisions(self, titles=None, revids=None):
        query = {"action": "query", "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main"}
        query["titles" if titles is not None else "revids"] = "|".join(titles if titles is not None else map(str, revids))
        return self.request(query)["query"]["pages"]

    def prepare(self):
        site = self.request({"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"})["query"]["rightsinfo"]
        if site["url"] != "https://creativecommons.org/licenses/by-sa/4.0/deed.es":
            raise ValueError("site license declaration changed")
        author_pages = self.revisions(titles=["Autor:" + name for name in dict.fromkeys(w["author"] for w in WORKS)])
        authors = {}
        for work in WORKS:
            author = next(p for p in author_pages if p["title"] == "Autor:" + work["author"])
            revision = author["revisions"][0]
            content = revision["slots"]["main"]["content"]
            if str(work["deathYear"]) not in content or 2026 - work["deathYear"] < 100:
                raise ValueError("author centenary provenance unavailable")
            authors[work["authorSlug"]] = {"name": work["author"], "deathYear": work["deathYear"],
                                          "sourcePageId": author["pageid"], "sourceRevisionId": revision["revid"],
                                          "sourceWikitextSha256": sha(content.encode()),
                                          "sourceUrl": "https://es.wikisource.org/w/index.php?oldid=" + str(revision["revid"])}
        targets, works = {}, []
        for work in WORKS:
            root = self.revisions(titles=[work["title"]])[0]
            if root.get("missing"):
                continue
            work_info = {**work, "rootPageId": root["pageid"], "rootRevisionId": root["revisions"][0]["revid"]}
            works.append(work_info)
            targets[root["title"]] = work_info
            pages = self.request({"action": "query", "list": "allpages", "apprefix": work["title"] + "/", "apnamespace": 0, "aplimit": 200})
            if pages.get("continue"):
                raise ValueError("work exceeds fixed chapter discovery limit")
            for page in pages["query"]["allpages"]:
                targets[page["title"]] = work_info
        if len(targets) > 150:
            raise ValueError("fixed ingestion exceeds150chapter bound")
        discovery = []
        names = sorted(targets)
        for first in range(0, len(names), 50):
            discovery.extend(self.revisions(titles=names[first:first + 50]))
        # Re-fetch fixed revision IDs, reject changes; mutable discovery is not
        # the reproducibility anchor used for the final corpus.
        fixed = []
        ids = [p["revisions"][0]["revid"] for p in discovery if not p.get("missing")]
        for first in range(0, len(ids), 50):
            fixed.extend(self.revisions(revids=ids[first:first + 50]))
        rows, pages, reasons = [], [], Counter()
        seen_texts, seen_paragraphs = {}, {}
        for page in sorted(fixed, key=lambda p: p["pageid"]):
            work = targets[page["title"]]
            family = "public-spanish-author:" + work["authorSlug"]
            split = assign_split(family)
            revision = page["revisions"][0]
            wiki = revision["slots"]["main"]["content"]
            info = {"pageId": page["pageid"], "revisionId": revision["revid"], "revisionTimestamp": revision["timestamp"],
                    "title": page["title"], "workTitle": work["title"], "authorSlug": work["authorSlug"],
                    "groupId": family, "split": split, "rawWikitextSha256": sha(wiki.encode()),
                    "url": "https://es.wikisource.org/w/index.php?oldid=" + str(revision["revid"]),
                    "historyUrl": "https://es.wikisource.org/w/index.php?curid=" + str(page["pageid"]) + "&action=history"}
            pages.append(info)
            if split == "test":
                reasons["reserved_author_text_not_cleaned"] += 1
                continue
            if any(t in wiki for t in ("<|", "|>")):
                reasons["reserved_control_token"] += 1
                continue
            try:
                text = clean_wikitext(wiki)
                parts = chunks(text)
            except ValueError:
                reasons["unsupported_or_unbounded_markup"] += 1
                continue
            words = WORD.findall(text)
            if len(words) < 150 or len({w.casefold() for w in words} & FUNCTIONS) < 8:
                reasons["short_index_or_weak_spanish_prose"] += 1
                continue
            digest = sha(text.encode())
            paragraph_hashes = {sha(" ".join(p.split()).encode()) for p in text.split("\n\n") if len(WORD.findall(p)) >= 12}
            if digest in seen_texts and seen_texts[digest] == split:
                reasons["samefold_duplicate_document"] += 1
                continue
            seen_texts[digest] = split
            for h in paragraph_hashes:
                seen_paragraphs[h] = split
            info["cleanTextSha256"] = digest
            for index, part in enumerate(parts):
                record = {"id": f"wikisource:es:{page['pageid']}:{revision['revid']}:chunk:{index}",
                          "groupId": family, "split": split, "source": {"dataset": "Wikisource/es-original-essays-v1",
                          "kind": "public-original-essay-grammar", "language": "es", "license": "CC-BY-SA-4.0 transcription; public-domain original",
                          "author": work["author"], "authorSlug": work["authorSlug"], "authorDeathYear": work["deathYear"],
                          "workTitle": work["title"], "pageId": page["pageid"], "revisionId": revision["revid"],
                          "revisionTimestamp": revision["timestamp"], "title": page["title"], "url": info["url"], "attributionHistoryUrl": info["historyUrl"],
                          "rawWikitextSha256": info["rawWikitextSha256"], "cleanTextSha256": digest, "chunkIndex": index,
                          "extraction": "No template expansion; drop templates/tables/references/navigation, keep link display text, remove markup and normalize structural whitespace"}, "text": part}
                validate_text_record(record)
                rows.append(record)
        # Whole-author guards: even a single crossfold duplicate excludes both
        # entire author families rather than relabeling or slicing chapters.
        by_text, by_paragraph = {}, {}
        collisions = set()
        for row in rows:
            signatures = [sha(row["text"].encode())] + [sha(" ".join(p.split()).encode()) for p in row["text"].split("\n\n") if len(WORD.findall(p)) >= 12]
            for signature in signatures:
                previous = by_text.get(signature)
                if previous and previous["split"] != row["split"]:
                    collisions.update((previous["groupId"], row["groupId"]))
                by_text[signature] = row
        rows = [r for r in rows if r["groupId"] not in collisions]
        unique, seen = [], set()
        for row in rows:
            digest = sha(row["text"].encode())
            if digest not in seen:
                seen.add(digest)
                unique.append(row)
        counts = Counter(r["split"] for r in unique)
        if not counts["train"] or not counts["validation"]:
            raise ValueError("new corpus lacks TRAIN/VALIDATION")
        corpus = self.output / "wikisource-essays.train-validation.jsonl"
        with corpus.open("xb") as stream:
            for row in unique:
                stream.write(encode(row))
        if corpus.stat().st_size > 24 * MIB:
            raise ValueError("24-MiB final corpus cap")
        content = hashlib.sha256()
        for row in unique:
            content.update(encode(row["text"]))
        manifest = {"schemaVersion": 1, "purpose": "new public original Spanish essay/grammar language corpus, not model evaluation",
                    "corpusSha256": sha(corpus.read_bytes()), "contentSha256": content.hexdigest(),
                    "file": corpus.name, "bytes": corpus.stat().st_size,
                    "splitCounts": {"train": counts["train"], "validation": counts["validation"], "test": 0},
                    "testUsedForTraining": False, "sources": [{"dataset": "Wikisource/es-original-essays-v1",
                    "license": "CC-BY-SA-4.0 transcription; original authors died more than100years ago",
                    "receipt": "source-receipts.json", "licenseDeclaration": site}],
                    "addedRecords": len(unique), "addedSourceIds": sorted({str(r["source"]["pageId"]) for r in unique}),
                    "authors": authors, "works": works, "pages": pages,
                    "authorFamilyPolicy": "public-spanish-author:<lowercase ASCII slug>; all works and chapters share assign_split(authorfamily)",
                    "guards": {"excludedWholeAuthorFamilyIds": sorted(collisions), "sourceFilterCounts": dict(reasons), "foldsMoved": 0},
                    "discoveryResponses": self.receipts, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                    "helperSha256": self.helper_sha256, "modelsLoaded": False, "tokenizerCalls": False,
                    "ownerMessagesIncluded": False, "evaluationCasesRead": False,
                    "limits": {"seconds": 600, "cpuCores": 1, "memoryMiB": 512, "diskMiB": 512, "responseMiB": 128, "corpusMiB": 24},
                    "resources": {"elapsedSeconds": time.monotonic() - self.started, "downloadedBytes": self.network_bytes},
                    "limitations": ["Historical originals contain old spelling, dated science and social views, not current factual ground truth",
                                    "Source transcription may contain errors; neither every fact nor every rights claim is independently certified",
                                    "Removing structural markup changes separators and drops template-contained material; no modern rewriting or generated labels",
                                    "Exact guards do not detect semantic paraphrases or all overlap with other public datasets",
                                    "This causal TEXT corpus is not conversational SFT and does not prove conversational ability"]}
        (self.output / "manifest.json").write_bytes(encode(manifest))
        (self.output / "source-receipts.json").write_bytes(encode(self.receipts))
        self.check()
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {max(affinity)})
    resource.setrlimit(resource.RLIMIT_AS, (512 * MIB, 512 * MIB))
    if args.output.exists():
        raise ValueError("output already exists; source receipts remain immutable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + args.output.name + ".", dir=args.output.parent))
    try:
        result = Collector(staging).prepare()
        os.rename(staging, args.output)
        print(json.dumps({key: result[key] for key in ("corpusSha256", "contentSha256", "splitCounts", "bytes", "addedRecords", "resources")}, ensure_ascii=False))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
