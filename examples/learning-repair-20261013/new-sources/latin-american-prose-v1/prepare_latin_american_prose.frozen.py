#!/usr/bin/env python3
"""Collect a finite, revision-pinned Wikipedia Spanish dialect prose subset.

This is causal TEXT, not conversational supervision or a complete regional
catalogue. Whole macro-dialect families receive folds before chunking. Existing
TRAIN/VALIDATION article IDs and exact paragraphs are protected. No TEST file,
model, tokenizer, owner conversation, or evaluation suite is accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ia_propia.conversation.data import assign_split
from ia_propia.conversation.language_data import validate_text_record

MIB = 1 << 20
API = "https://es.wikipedia.org/w/api.php"
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# Grouping is fixed before discovery, text cleaning, and chunk construction.
# Countries have many varieties: these are study families, not an assertion
# that national speech or macro-dialects are homogeneous or mutually exclusive.
TARGETS = [
    ("Español mexicano", "mexicano", "México"),
    ("Español rioplatense", "rioplatense", "Argentina; Uruguay"),
    ("Español andino", "andino", "regiones andinas de varios países"),
    ("Español chileno", "chileno", "Chile"),
    ("Español colombiano", "andino", "Colombia; incluye regiones no andinas"),
    ("Español venezolano", "caribeno", "Venezuela; incluye regiones no caribeñas"),
    ("Español ecuatoriano", "andino", "Ecuador"),
    ("Español boliviano", "andino", "Bolivia"),
    ("Español peruano", "andino", "Perú"),
    ("Español paraguayo", "paraguayo", "Paraguay"),
    ("Español centroamericano", "centroamericano", "Centroamérica"),
    ("Español cubano", "caribeno", "Cuba"),
    ("Español dominicano", "caribeno", "República Dominicana"),
    ("Español puertorriqueño", "caribeno", "Puerto Rico"),
    ("Español guatemalteco", "centroamericano", "Guatemala"),
    ("Español hondureño", "centroamericano", "Honduras"),
    ("Español salvadoreño", "centroamericano", "El Salvador"),
    ("Español nicaragüense", "centroamericano", "Nicaragua"),
    ("Español costarricense", "centroamericano", "Costa Rica"),
    ("Español panameño", "centroamericano", "Panamá"),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def signatures(text):
    result = {sha(text.encode()), sha(" ".join(text.split()).encode())}
    for paragraph in text.split("\n\n") + text.splitlines():
        if len(WORD.findall(paragraph)) >= 12:
            result.add(sha(" ".join(paragraph.split()).encode()))
    return result


def drop_balanced(text, opening, closing):
    output, cursor, depth = [], 0, 0
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
        raise ValueError("unbalanced structural source markup")
    return "".join(output)


def clean(wiki):
    text = re.sub(r"<!--.*?-->", "", wiki, flags=re.S)
    text = re.sub(r"<ref\b[^>]*/\s*>", "", text, flags=re.I)
    text = re.sub(r"<(ref|blockquote|poem|q)\b[^>]*>.*?</\1\s*>", "", text, flags=re.S | re.I)
    text = drop_balanced(drop_balanced(text, "{{", "}}"), "{|", "|}")
    # Bibliography/link sections are source provenance, not lexical targets.
    text = re.split(r"(?im)^={1,6}\s*(?:Referencias|Bibliografía|Notas|Enlaces externos|Véase también)\s*={1,6}\s*$", text)[0]
    def link(match):
        target, *labels = match.group(1).split("|")
        if target.casefold().lstrip(":").startswith(("archivo:", "file:", "imagen:", "image:", "categoría:", "category:")):
            return ""
        return labels[-1] if labels else target.lstrip(":")
    text = re.sub(r"\[\[([^\[\]]+)\]\]", link, text)
    text = re.sub(r"\[https?://\S+(?:\s+([^\]]+))?\]", lambda m: m.group(1) or "", text)
    text = re.sub(r"<(?:br|hr)\b[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(re.sub(r"'{2,5}", "", text))
    retained, dropped = [], Counter()
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:={1,6}|[*#:;]+)\s*", "", line)
        line = re.sub(r"\s*={1,6}\s*$", "", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue
        # Keep short regional examples. Drop paragraphs containing long direct
        # quotations instead of redistributing a cited modern author's passage.
        quoted = re.findall(r'"([^"]+)"|«([^»]+)»|“([^”]+)”', line)
        if any(len(WORD.findall(next(s for s in spans if s))) >= 80 for spans in quoted):
            dropped["long_quoted_paragraph"] += 1
            continue
        if "{{" in line or "}}" in line or "[[" in line or "]]" in line or "|}" in line:
            dropped["unresolved_markup_paragraph"] += 1
            continue
        if re.search(r"\[\s*\]|\(\s*\)|\b(?:del|desde|hasta|siglo)\s*[,.;:]", line, flags=re.I):
            dropped["paragraph_with_template_removal_gap"] += 1
            continue
        retained.append(line)
    return "\n\n".join(retained), dropped


def chunks(text):
    result, current = [], ""
    for paragraph in text.split("\n\n"):
        if len(paragraph.encode()) > 65536:
            raise ValueError("source paragraph exceeds chunk bound")
        candidate = current + ("\n\n" if current else "") + paragraph
        if len(candidate.encode()) > 65536:
            result.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        result.append(current)
    return result


class Collector:
    def __init__(self, output, seeds, replay_receipts=None):
        self.output, self.seeds = output, seeds
        self.started, self.network_bytes = time.monotonic(), 0
        self.receipts, self.issues = [], []
        self.replay_receipts, self.replay_read_bytes = replay_receipts, 0
        self.replay = {}
        if replay_receipts is not None:
            self.replay = {item["url"]: item for item in json.loads((replay_receipts / "source-receipts.json").read_text())}
        self.deadline = self.started + 590
        self.helper_sha256 = sha(Path(__file__).read_bytes())
        output.mkdir(parents=True, exist_ok=False)
        (output / "raw").mkdir()

    def check(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError("600-second collection limit")
        if sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) > 32 * MIB:
            raise ValueError("32-MiB collection store limit")

    def request(self, params):
        self.check()
        url = API + "?" + urlencode({**params, "format": "json", "formatversion": 2})
        if self.replay_receipts is not None:
            if url not in self.replay:
                raise ValueError("unrecorded source request in offline replay")
            original = self.replay[url]
            raw = (self.replay_receipts / original["file"]).read_bytes()
            if sha(raw) != original["sha256"] or len(raw) != original["bytes"]:
                raise ValueError("immutable source receipt differs")
            self.replay_read_bytes += len(raw)
            name = f"response-{len(self.receipts):03d}.json"
            (self.output / "raw" / name).write_bytes(raw)
            self.receipts.append({**original, "file": "raw/" + name, "retrieval": "offline replay of actual primary API response"})
            (self.output / "source-receipts.json").write_bytes(encode(self.receipts))
            return json.loads(raw)
        for attempt in range(4):
            self.check()
            if self.network_bytes >= 24 * MIB:
                raise ValueError("24-MiB network budget exhausted")
            time.sleep(1.2)
            try:
                with urlopen(Request(url, headers={"User-Agent": "UNIDAD-Spanish-Language-Research/1.0 (finite public corpus)"}), timeout=min(25, max(1, self.deadline - time.monotonic()))) as response:
                    raw = response.read(min(4 * MIB, 24 * MIB - self.network_bytes) + 1)
                    status, headers = response.status, dict(response.headers)
                break
            except HTTPError as exc:
                delay = exc.headers.get("Retry-After", "5")
                delay = min(30, int(delay) if delay.isdigit() else 5)
                issue = {"url": url, "status": exc.code, "attempt": attempt, "retryAfterSeconds": delay}
                self.issues.append(issue)
                (self.output / "http-failures.json").write_bytes(encode(self.issues))
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(delay)
        self.network_bytes += len(raw)
        if len(raw) > 4 * MIB or self.network_bytes > 24 * MIB:
            raise ValueError("bounded public response limit")
        name = f"response-{len(self.receipts):03d}.json"
        (self.output / "raw" / name).write_bytes(raw)
        item = {"file": "raw/" + name, "url": url, "status": status, "headers": headers,
                "bytes": len(raw), "sha256": sha(raw), "fetchedAtUtc": datetime.now(timezone.utc).isoformat()}
        self.receipts.append(item)
        (self.output / "source-receipts.json").write_bytes(encode(self.receipts))
        value = json.loads(raw)
        if "error" in value:
            raise ValueError("source API rejected request: " + value["error"].get("code", "unknown"))
        return value

    def collect(self):
        rights = self.request({"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"})["query"]["rightsinfo"]
        if rights["url"] != "https://creativecommons.org/licenses/by-sa/4.0/deed.es":
            raise ValueError("primary-source license declaration changed")
        old_keys, old_pages, seed_receipts = defaultdict(set), defaultdict(set), []
        for path in self.seeds:
            if "test" in path.name.lower():
                raise ValueError("TEST inputs are forbidden")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    row = json.loads(line)
                    if row["split"] not in ("train", "validation"):
                        raise ValueError("TRAIN/VALIDATION-only input required")
                    for signature in signatures(row["text"]):
                        old_keys[signature].add(row["split"])
                    source = row["source"]
                    if source["dataset"] == "wikimedia/wikipedia":
                        old_pages[str(source["pageId"])].add(row["split"])
            seed_receipts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        discovered, page_targets, resolutions = {}, defaultdict(list), []
        for requested, slug, region in TARGETS:
            family = "wikipedia-es-dialect-family:" + slug
            split = assign_split(family)
            query = self.request({"action": "query", "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main", "redirects": 1, "titles": requested})["query"]
            page = query["pages"][0]
            resolution = {"requestedTitle": requested, "groupId": family, "split": split, "region": region,
                          "normalized": query.get("normalized", []), "redirects": query.get("redirects", []),
                          "resolvedTitle": page["title"], "missing": bool(page.get("missing"))}
            resolutions.append(resolution)
            if page.get("missing"):
                continue
            page_id = str(page["pageid"])
            revision = page["revisions"][0]
            page_targets[page_id].append(resolution)
            discovered[page_id] = {"page": page, "revision": revision}
        # Freeze each discovered revision by a second request for its exact ID.
        fixed = {}
        ids = [d["revision"]["revid"] for d in discovered.values()]
        for first in range(0, len(ids), 5):
            query = self.request({"action": "query", "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main", "revids": "|".join(map(str, ids[first:first + 5]))})["query"]
            for page in query["pages"]:
                revision = page["revisions"][0]
                expected = discovered[str(page["pageid"])]
                if revision != expected["revision"]:
                    raise ValueError("fixed revision differs from discovered content")
                fixed[str(page["pageid"])] = page
        rows, pages, reasons, excluded = [], [], Counter(), set()
        for page_id, page in sorted(fixed.items(), key=lambda pair: int(pair[0])):
            targets = page_targets[page_id]
            families = {t["groupId"] for t in targets}
            if len(families) != 1:
                excluded.update(families)
                reasons["redirect_links_different_macro_families"] += 1
                continue
            family = targets[0]["groupId"]
            split = targets[0]["split"]
            revision = page["revisions"][0]
            wiki = revision["slots"]["main"]["content"]
            info = {"pageId": int(page_id), "revisionId": revision["revid"], "revisionTimestamp": revision["timestamp"],
                    "title": page["title"], "groupId": family, "split": split, "requestedTitles": [t["requestedTitle"] for t in targets],
                    "regions": [t["region"] for t in targets], "rawWikitextSha256": sha(wiki.encode()),
                    "url": "https://es.wikipedia.org/w/index.php?oldid=" + str(revision["revid"]),
                    "attributionHistoryUrl": "https://es.wikipedia.org/w/index.php?curid=" + page_id + "&action=history"}
            pages.append(info)
            if split == "test":
                reasons["reserved_family_not_cleaned"] += 1
                continue
            if page_id in old_pages:
                if old_pages[page_id] != {split}:
                    excluded.add(family)
                    reasons["existing_article_fold_conflict_whole_family"] += 1
                else:
                    reasons["existing_article_not_reintroduced"] += 1
                continue
            if "<|" in wiki or "|>" in wiki:
                reasons["reserved_control_token"] += 1
                continue
            try:
                text, dropped = clean(wiki)
            except ValueError:
                reasons["unsupported_unbalanced_source_markup"] += 1
                continue
            reasons.update(dropped)
            if len(WORD.findall(text)) < 100:
                reasons["short_or_empty_prose"] += 1
                continue
            full_matches = old_keys.get(sha(text.encode()), set()) | old_keys.get(sha(" ".join(text.split()).encode()), set())
            if full_matches:
                if full_matches != {split}:
                    excluded.add(family)
                    reasons["existing_crossfold_fulltext_whole_family"] += 1
                else:
                    reasons["existing_samefold_fulltext_not_reintroduced"] += 1
                continue
            retained = []
            for paragraph in text.split("\n\n"):
                matches = old_keys.get(sha(" ".join(paragraph.split()).encode()), set()) if len(WORD.findall(paragraph)) >= 12 else set()
                if matches and matches != {split}:
                    excluded.add(family)
                    reasons["existing_crossfold_paragraph_whole_family"] += 1
                elif matches:
                    reasons["existing_samefold_paragraph_not_reintroduced"] += 1
                else:
                    retained.append(paragraph)
            text = "\n\n".join(retained)
            if len(WORD.findall(text)) < 100:
                reasons["short_after_dedup"] += 1
                continue
            info["cleanTextSha256"] = sha(text.encode())
            for index, part in enumerate(chunks(text)):
                source = {"dataset": "Wikipedia/es-latin-american-dialect-prose-v1", "kind": "public-encyclopedic-dialect-description",
                          "language": "es", "license": "CC-BY-SA-4.0", **info, "chunkIndex": index,
                          "extraction": "drop templates/tables/reference and external-link sections/HTML block quotes; retain link display text and own prose; remove long quoted paragraphs, unresolved markup and visibly empty phonetic/template gaps; remove exact samefold seed paragraphs"}
                row = {"id": f"wikipedia:es:regional:{page_id}:{revision['revid']}:chunk:{index}",
                       "groupId": family, "split": split, "source": source, "text": part}
                validate_text_record(row)
                rows.append(row)
        seen = defaultdict(list)
        for row in rows:
            for signature in signatures(row["text"]):
                seen[signature].append((row["groupId"], row["split"]))
        for occurrences in seen.values():
            if len({split for _, split in occurrences}) > 1:
                excluded.update(group for group, _ in occurrences)
        unique, seen_text = [], set()
        for row in rows:
            key = sha(row["text"].encode())
            if row["groupId"] not in excluded and key not in seen_text:
                seen_text.add(key)
                unique.append(row)
        counts = Counter(row["split"] for row in unique)
        if not counts["train"] or not counts["validation"]:
            raise ValueError("bounded subset lacks TRAIN/VALIDATION after guards")
        corpus = self.output / "latin-american-prose.train-validation.jsonl"
        corpus.write_bytes(b"".join(encode(row) for row in unique))
        if corpus.stat().st_size > 24 * MIB:
            raise ValueError("24-MiB final corpus bound")
        content = hashlib.sha256()
        for row in unique:
            content.update(encode(row["text"]))
        manifest = {"schemaVersion": 1, "purpose": "bounded new Latin American Spanish dialect explanatory prose for causal learning",
                    "file": corpus.name, "bytes": corpus.stat().st_size, "corpusSha256": sha(corpus.read_bytes()), "contentSha256": content.hexdigest(),
                    "splitCounts": {"train": counts["train"], "validation": counts["validation"], "test": 0},
                    "wordCounts": {fold: sum(len(WORD.findall(row["text"])) for row in unique if row["split"] == fold) for fold in ("train", "validation")},
                    "familyCount": len({row["groupId"] for row in unique}), "sourcePageCount": len({row["source"]["pageId"] for row in unique}),
                    "sources": [{"dataset": "Wikipedia/es-latin-american-dialect-prose-v1", "api": API, "licenseDeclaration": rights}],
                    "resolutionCatalogue": resolutions, "pages": pages, "seedInputs": seed_receipts,
                    "macroFamilyPolicy": "fixed wikipedia-es-dialect-family:<slug>; Caribbean, Central American, Andean, Rioplatense articles share their macro-family before chunking; no fold moved",
                    "guards": {"excludedWholeMacroFamilies": sorted(excluded), "sourceFilterCounts": dict(reasons), "foldsMoved": 0,
                               "existingArticlePolicy": "skip existing IDs if same fold; exclude new whole macro-family on old fold conflict", "exactGuardScope": "full raw and normalized texts; normalized paragraphs or single lines >=12 words; no semantic paraphrase guarantee"},
                    "createdAtUtc": datetime.now(timezone.utc).isoformat(), "helperSha256": self.helper_sha256,
                    "modelsLoaded": False, "tokenizerCalls": False, "ownerMessagesIncluded": False, "evaluationCasesRead": False, "testUsedForTraining": False,
                    "limits": {"seconds": 600, "cpuCores": 1, "memoryMiB": 512, "networkMiB": 24, "storeMiB": 32, "corpusMiB": 24, "minimumRequestIntervalSeconds": 1.2},
                    "resources": {"elapsedSeconds": time.monotonic() - self.started, "downloadedBytes": self.network_bytes,
                                  "offlineSourceReadBytes": self.replay_read_bytes,
                                  "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                    "limitations": ["Selected articles describe regional language; not a complete vocabulary catalogue or naturally occurring conversations",
                                    "Many national and macro-dialects overlap, contain diverse registers and vary by locality, social group, age and speaker",
                                    "Wikipedia community prose is not an independently fact-checked truth guarantee; source errors remain possible",
                                    "Templates and tables are not expanded; some IPA notation, lexical tables and quotations are intentionally omitted",
                                    "This corpus is pending training and does not prove regional conversational competence"]}
        (self.output / "manifest.json").write_bytes(encode(manifest))
        self.check()
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=Path, action="append", required=True)
    parser.add_argument("--replay-receipts", type=Path, help="replay immutable private primary API responses without network")
    args = parser.parse_args()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (512 * MIB, 512 * MIB))
    result = Collector(args.output, args.seed, args.replay_receipts).collect()
    print(json.dumps({key: result[key] for key in ("bytes", "corpusSha256", "contentSha256", "splitCounts", "wordCounts", "familyCount", "sourcePageCount", "resources")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
