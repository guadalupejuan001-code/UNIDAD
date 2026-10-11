#!/usr/bin/env python3
"""Prepare bounded public Spanish TEXT and SFT corpora without ML packages.

Only explicitly supplied, already-reviewed canonical inputs are read. Wikipedia
is fetched from the pinned public snapshot through the environment's normal
HTTP(S) proxy. This helper never loads weights, fits a tokenizer, reads private
chat, runs model evaluation, or chooses data using model responses/losses.

Source license fields record declarations, not certification of article-level
rights. The pinned HF card declares CC-BY-SA-3.0/GFDL; current Wikimedia legal
documentation declares CC-BY-SA-4.0/GFDL. Attribution metadata is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ia_propia.conversation.contract import DEFAULT_SYSTEM_PROMPT, SPECIAL_TOKENS
from ia_propia.conversation.data import assign_split, load_corpus
from ia_propia.conversation.language_data import load_text_corpus, validate_text_record

WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_CONFIG = "20231101.es"
WIKI_ROWS = 1_841_155
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
SPANISH_FUNCTIONS = set("de la el los las y en del que es un una por con para su se al como más pero este esta sus sobre entre también fue son lo o a le si cuando sin desde donde no muy cual esto estas estos esa ese esos otras otro otra hasta hay han hacer ser tiene están está eran era uno dos cada todo todos toda todas puede pueden después antes durante sólo aunque porque sino mismo misma".split())
NAMESPACES = ("anexo:", "wikipedia:", "portal:", "categoría:", "archivo:", "plantilla:", "ayuda:", "discusión:", "usuario:")
MAX_TEXT_BYTES = 65_536
SOURCE_LIMITS = {"wikimedia/wikipedia": 64, "OpenAssistant/oasst1": 128}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def create_output_directory(path: Path) -> None:
    """An existing output, even an empty directory, is never reused."""
    path.mkdir(parents=True, exist_ok=False, mode=0o700)


def write_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o600)


def write_json(path: Path, value: object) -> None:
    write_exclusive(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def write_jsonl(path: Path, records: list[dict]) -> dict:
    with path.open("xb") as stream:
        for record in records:
            stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode())
    path.chmod(0o600)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}


def verify_frozen_inputs(expected: dict[Path, str]) -> None:
    """Reject identity changes rather than blessing a new hash at the end."""
    for path, fingerprint in expected.items():
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or file_sha(path) != fingerprint:
            raise ValueError(f"input hash mismatch: {path.name}")


def load_reviewed_inputs(oasst: Path, oasst_manifest: Path, historical: Path,
                         historical_sha: str, rich: Path, rich_sha: str):
    """Check input identity and canonical schema without importing ML code."""
    manifest_sha = file_sha(oasst_manifest)
    manifest = json.loads(oasst_manifest.read_text())
    source = manifest.get("source", {})
    if source.get("dataset") != "OpenAssistant/oasst1" or source.get("revision") != OASST_REVISION or source.get("language") != "es" or source.get("license") != "Apache-2.0" or source.get("externalSyntheticExcluded") is not True:
        raise ValueError("OASST manifest is not the pinned reviewed Spanish source")
    expected = {oasst: manifest["sha256"], oasst_manifest: manifest_sha,
                historical: historical_sha, rich: rich_sha}
    verify_frozen_inputs(expected)
    human = load_corpus(oasst)
    old = load_corpus(historical)
    authored = load_corpus(rich)
    for r in human:
        s = r["source"]
        if s.get("dataset") != "OpenAssistant/oasst1" or s.get("kind") != "public-human-reviewed" or s.get("revision") != OASST_REVISION or s.get("license") != "Apache-2.0" or s.get("language") != "es" or assign_split(r["groupId"]) != r["split"]:
            raise ValueError("OASST record provenance or historical fold differs")
    if len(human) != manifest["records"]:
        raise ValueError("OASST manifest record count differs")
    if any(r["source"].get("kind") != "authored-synthetic" or r["source"].get("dataset") != "unidad-rich-spanish-dialogues-v1" for r in authored):
        raise ValueError("rich source must be project-authored fictional dialogues")
    raw_lines = {}
    for path, records in ((historical, old), (oasst, human), (rich, authored)):
        expected_records = {r["id"]: r for r in records}
        seen_ids = set()
        for raw in path.open("rb"):
            r = json.loads(raw)
            if expected_records.get(r["id"]) != r or r["id"] in seen_ids:
                raise ValueError("input content changed between canonical and raw-line reads")
            seen_ids.add(r["id"])
            if r["id"] in raw_lines and json.loads(raw_lines[r["id"]]) != r:
                raise ValueError("same input id has different content")
            # Full canonical OASST is read after its historical subset so its
            # original line wins, even if the subset used other JSON spacing.
            raw_lines[r["id"]] = raw
        if seen_ids != set(expected_records):
            raise ValueError("input records disappeared during preparation")
    verify_frozen_inputs(expected)
    return human, old, authored, manifest, raw_lines, expected


def sft_union(old: list[dict], human: list[dict], rich: list[dict]) -> list[dict]:
    full_human = {r["id"]: r for r in human}
    authored_old = []
    for r in old:
        if r["source"].get("kind") == "public-human-reviewed":
            if full_human.get(r["id"]) != r:
                raise ValueError("historical human record differs from full reviewed canonical input")
        else:
            authored_old.append(r)
    selected, ids, families = [], {}, {}
    for r in authored_old + human + rich:
        if r["source"].get("kind") not in ("authored-synthetic", "public-human-reviewed"):
            raise ValueError("unexpected SFT source kind")
        if r["id"] in ids:
            if ids[r["id"]] != r:
                raise ValueError("same SFT id has different content")
            continue
        if r["groupId"] in families and families[r["groupId"]] != r["split"]:
            raise ValueError("SFT family crosses folds")
        families[r["groupId"]] = r["split"]
        ids[r["id"]] = r
        selected.append(r)
    return selected


def ambiguous_sft_families(records: list[dict]) -> tuple[set[str], dict]:
    by_prompt, by_dialogue = defaultdict(list), defaultdict(list)
    for r in records:
        effective = r["messages"]
        if effective[0]["role"] != "system":
            effective = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + effective
        messages = [{"role": m["role"], "content": " ".join(m["content"].split())} for m in effective]
        for mapping, content in ((by_prompt, messages[:-1]), (by_dialogue, messages)):
            mapping[sha(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode())].append(r)
    groups = set()
    counts = {}
    for label, mapping in (("fullPromptCrossSplit", by_prompt), ("fullDialogueCrossSplit", by_dialogue)):
        collisions = [values for values in mapping.values() if len({r["split"] for r in values}) > 1]
        counts[label] = len(collisions)
        for values in collisions:
            if any(r["source"].get("dataset") != "OpenAssistant/oasst1" or r["source"].get("kind") != "public-human-reviewed" for r in values):
                raise ValueError("crossfold authored collision requires an explicit source correction")
            groups.update(r["groupId"] for r in values)
    if any(r["groupId"] in groups and (r["source"].get("dataset") != "OpenAssistant/oasst1" or r["source"].get("kind") != "public-human-reviewed") for r in records):
        raise ValueError("ambiguous OASST family contains authored or other provenance")
    return groups, counts


def exclude_whole_families(records: list[dict], groups: set[str]):
    return ([r for r in records if r["groupId"] not in groups],
            [r for r in records if r["groupId"] in groups])


def extract_human_messages(human: list[dict]) -> dict[str, dict]:
    messages = {}
    for r in human:
        mids = r["source"].get("messageIds")
        if not isinstance(mids, list) or len(mids) != len(r["messages"]):
            raise ValueError("reviewed human messageIds do not align with raw messages")
        for mid, message in zip(mids, r["messages"]):
            value = {"messageId": mid, "groupId": r["groupId"], "split": r["split"], "role": message["role"], "text": message["content"]}
            if mid in messages and messages[mid] != value:
                raise ValueError("same human message id changes text/family/fold")
            messages[mid] = value
    return messages


def crossfold_text_clone_families(messages: dict[str, dict]):
    by_text = defaultdict(list)
    for m in messages.values():
        by_text[sha(m["text"].encode())].append(m)
    collisions = [values for values in by_text.values() if len({m["split"] for m in values}) > 1]
    groups = {m["groupId"] for values in collisions for m in values}
    return groups, {"classes": len(collisions), "messageIds": sum(map(len, collisions)), "families": len(groups)}


def human_text_records(messages: dict[str, dict], corpus_sha: str, seen: set[str]):
    records, duplicates = [], Counter()
    for mid in sorted(messages):
        m = messages[mid]
        fingerprint = sha(m["text"].encode())
        if fingerprint in seen:
            duplicates[m["split"]] += 1
            continue
        seen.add(fingerprint)
        r = {"id": f"oasst-text:{mid}", "groupId": m["groupId"], "split": m["split"], "source": {"dataset": "OpenAssistant/oasst1", "kind": "public-human-reviewed", "language": "es", "license": "Apache-2.0", "revision": OASST_REVISION, "messageId": mid, "role": m["role"], "canonicalCorpusSha256": corpus_sha, "extraction": "unique_message_content_only_no_role_or_dialogue_wrapper", "externalSyntheticExcluded": True}, "text": m["text"]}
        validate_text_record(r)
        records.append(r)
    return records, dict(duplicates)


def split_utf8_exact(text: str) -> list[str]:
    raw, start, chunks = text.encode(), 0, []
    while start < len(raw):
        end = min(start + MAX_TEXT_BYTES, len(raw))
        if end < len(raw):
            while (raw[end] & 0xC0) == 0x80:
                end -= 1
            paragraph = raw.rfind(b"\n\n", start, end)
            if paragraph >= start + MAX_TEXT_BYTES // 2:
                end = paragraph + 2
        chunk = raw[start:end].decode("utf-8", errors="strict")
        if not chunk.strip():
            raise ValueError("whitespace-only chunk; reject whole article")
        chunks.append(chunk)
        start = end
    if not chunks or "".join(chunks) != text:
        raise ValueError("empty article or changed chunk text")
    return chunks


def wiki_quality(row: dict, truncated: list):
    if truncated:
        return "server_truncated_cells"
    text, title, page_id, url = (row.get(k) for k in ("text", "title", "id", "url"))
    if not all(isinstance(v, str) and v.strip() for v in (text, title, page_id, url)):
        return "missing_document_fields"
    if not page_id.isdigit() or not url.startswith("https://es.wikipedia.org/"):
        return "invalid_article_identity"
    if title.casefold().startswith(NAMESPACES):
        return "non_article_namespace"
    if any(t in text for t in SPECIAL_TOKENS):
        return "reserved_control_token"
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
        return "non_text_control_character"
    words = WORD.findall(text)
    if len(words) < 150:
        return "short_article_below_150_words"
    if sum(ch.isalpha() for ch in text) / max(1, sum(not ch.isspace() for ch in text)) < .45:
        return "low_prose_letter_fraction"
    frequencies = Counter(w.casefold() for w in words)
    matches = [w for w in SPANISH_FUNCTIONS if frequencies[w]]
    if len(matches) < 8 or sum(frequencies[w] for w in matches) < 20:
        return "weak_spanish_function_word_signal"
    return None


def download(url: str, *, revision: str | None = None):
    # Honor inherited HTTP(S)_PROXY and normal certificate verification.
    request = Request(url, headers={"User-Agent": "Unidad-public-corpus-preparation/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read(25_000_001)
                headers = {"status": response.status, "contentType": response.headers.get("Content-Type"), "xRevision": response.headers.get("x-revision"), "etag": response.headers.get("ETag")}
            if len(raw) > 25_000_000 or (revision is not None and headers["xRevision"] != revision):
                raise ValueError("oversized response or unpinned source revision")
            return raw, headers
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(min(attempt + 1, 3))
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(min(attempt + 1, 3))
    raise AssertionError("unreachable")


def fetch_page(order: int):
    page_count = math.ceil(WIKI_ROWS / 100)
    if math.gcd(7919, page_count) != 1:
        raise ValueError("page stride is not coprime")
    offset = ((order * 7919 + 37) % page_count) * 100
    query = urlencode({"dataset": "wikimedia/wikipedia", "config": WIKI_CONFIG, "split": "train", "offset": offset, "length": min(100, WIKI_ROWS - offset), "revision": WIKI_REVISION})
    url = "https://datasets-server.huggingface.co/rows?" + query
    raw, headers = download(url, revision=WIKI_REVISION)
    value = json.loads(raw)
    if value.get("partial") or value.get("num_rows_total") != WIKI_ROWS or not isinstance(value.get("rows"), list):
        raise ValueError("partial rows or changed source size")
    return order, offset, raw, headers, value


def collect_wikipedia(output: Path, seen: set[str], target: int, max_pages: int, workers: int):
    cache = output / "wiki-row-pages"
    cache.mkdir(mode=0o700)
    records, receipts, reasons, document_shas, page_ids = [], [], Counter(), set(), set()
    train_words = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0, max_pages, 4):
            for order, offset, raw, headers, value in pool.map(fetch_page, range(first, min(first + 4, max_pages))):
                name = f"page-{order:04d}-offset-{offset}.json"
                write_exclusive(cache / name, raw)
                receipts.append({"order": order, "offset": offset, "rows": len(value["rows"]), "file": name, "sha256": sha(raw), "response": headers})
                for wrapped in value["rows"]:
                    row = wrapped.get("row", {})
                    reason = wiki_quality(row, wrapped.get("truncated_cells", []))
                    if reason:
                        reasons[reason] += 1
                        continue
                    if row["id"] in page_ids:
                        raise ValueError("article id repeated in page permutation")
                    page_ids.add(row["id"])
                    document_sha = sha(row["text"].encode())
                    if document_sha in document_shas:
                        reasons["duplicate_full_document_text"] += 1
                        continue
                    document_shas.add(document_sha)
                    try:
                        chunks = split_utf8_exact(row["text"])
                    except ValueError:
                        reasons["whitespace_only_document_chunk"] += 1
                        continue
                    group = "wikipedia:es:" + row["id"]
                    split = assign_split(group)
                    for i, text in enumerate(chunks):
                        fingerprint = sha(text.encode())
                        if fingerprint in seen:
                            reasons["duplicate_exact_text_chunk"] += 1
                            continue
                        seen.add(fingerprint)
                        record = {"id": f"wikipedia:{WIKI_CONFIG}:{row['id']}:chunk:{i}", "groupId": group, "split": split, "source": {"dataset": "wikimedia/wikipedia", "kind": "public-encyclopedic", "language": "es", "configuration": WIKI_CONFIG, "revision": WIKI_REVISION, "license": "CC-BY-SA-3.0", "licenseDeclaration": "pinned Hugging Face dataset card; some texts also GFDL; no claim of all-text dual licensing", "pageId": row["id"], "title": row["title"], "url": row["url"], "attributionHistoryUrl": "https://es.wikipedia.org/w/index.php?curid=" + row["id"] + "&action=history", "snapshotDate": "2023-11-01", "articleTextSha256": document_sha, "articleRevisionIdAvailable": False, "rowIndex": wrapped["row_idx"], "rowPageOffset": offset, "chunkIndex": i, "chunksBeforeGlobalDedup": len(chunks), "textNormalization": "none; original Unicode and line breaks preserved"}, "text": text}
                        validate_text_record(record)
                        records.append(record)
                        if split == "train":
                            train_words += len(WORD.findall(text))
            print(json.dumps({"stage": "download", "pages": len(receipts), "wikiTrainWords": train_words}), flush=True)
            if train_words >= target:
                break
    if train_words < target:
        raise ValueError("bounded download did not reach declared TRAIN word target")
    return records, {"pinnedRevision": WIKI_REVISION, "configuration": WIKI_CONFIG, "sampling": "page=(order*7919+37) mod ceil(1841155/100), finish four-page batch when TRAIN word target reached", "trainWordTarget": target, "trainWords": train_words, "sourceRows": WIKI_ROWS, "pages": receipts, "filterCounts": dict(reasons)}


def select_holdouts(records: list[dict], character_cap: int = 250_000,
                    family_limits: dict[str, int] | None = None):
    limits = SOURCE_LIMITS if family_limits is None else family_limits
    families = defaultdict(list)
    for r in records:
        families[(r["split"], r["source"]["dataset"], r["groupId"])].append(r)
    chosen = {k for k in families if k[0] == "train"}
    receipts = {}
    for split in ("validation", "test"):
        receipts[split] = {}
        for source, limit in limits.items():
            candidates = [k for k in families if k[:2] == (split, source)]
            ordered = sorted(candidates, key=lambda k: sha(("unidad-public-general-holdout-v1\0" + k[2]).encode()))
            characters, selected = 0, []
            for key in ordered:
                if len(selected) == limit:
                    break
                cost = sum(len(r["text"]) for r in families[key])
                if characters + cost <= character_cap:
                    characters += cost
                    selected.append(key[2])
                    chosen.add(key)
            receipts[split][source] = {"poolFamilies": len(candidates), "selectedFamilyIds": selected, "selectedCharacters": characters, "characterCap": character_cap, "familyCap": limit}
    return [r for r in records if (r["split"], r["source"]["dataset"], r["groupId"]) in chosen], receipts


def summary(records: list[dict], field: str):
    result = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in records if r["split"] == split]
        by_source = {}
        for source in sorted({r["source"]["dataset"] for r in subset}):
            current = [r for r in subset if r["source"]["dataset"] == source]
            details = {"records": len(current), "families": len({r["groupId"] for r in current})}
            if field == "text":
                details.update(characters=sum(len(r["text"]) for r in current), utf8Bytes=sum(len(r["text"].encode()) for r in current), unicodeWords=sum(len(WORD.findall(r["text"])) for r in current))
            by_source[source] = details
        result[split] = {"records": len(subset), "families": len({r["groupId"] for r in subset}), "bySource": by_source}
    return result


def licenses(output: Path):
    source_dir = output / "source-license-documents"
    source_dir.mkdir(mode=0o700)
    urls = {"pinned-HF-README.md": f"https://huggingface.co/datasets/wikimedia/wikipedia/raw/{WIKI_REVISION}/README.md", "wikimedia-legal.html": "https://dumps.wikimedia.org/legal.html", "cc-by-sa-3.0.html": "https://creativecommons.org/licenses/by-sa/3.0/legalcode", "cc-by-sa-4.0.html": "https://creativecommons.org/licenses/by-sa/4.0/legalcode"}
    receipts = []
    for name, url in urls.items():
        raw, headers = download(url)
        write_exclusive(source_dir / name, raw)
        receipts.append({"file": name, "url": url, "sha256": sha(raw), "response": headers})
    card = (source_dir / "pinned-HF-README.md").read_text()
    official = (source_dir / "wikimedia-legal.html").read_text()
    if "cc-by-sa-3.0" not in card or "gfdl" not in card or WIKI_CONFIG not in card:
        raise ValueError("pinned HF card differs from declared source/license documentation")
    versions = sorted(set(re.findall(r"creativecommons\.org/licenses/by-sa/(\d+\.\d+)/", official)))
    current = ["CC-BY-SA-" + version for version in versions]
    if "GNU Free Documentation License" in official:
        current.append("GFDL")
    if not versions:
        raise ValueError("official Wikimedia legal page no longer supplies a recognized license declaration")
    return {"documentation": receipts, "hfCardDeclared": ["CC-BY-SA-3.0", "GFDL"], "currentWikimediaLegalDeclared": current, "note": "Different documented declarations are retained; not certification of each article's rights. Some content may be CC-only. Retain article title/URL/history attribution. OASST source manifest declares Apache-2.0."}


def run(args):
    human, old, rich, human_manifest, raw_lines, frozen_inputs = load_reviewed_inputs(args.oasst_corpus, args.oasst_manifest, args.historical_sft, args.historical_sft_sha256, args.rich_sft, args.rich_sft_sha256)
    candidate = sft_union(old, human, rich)
    sft_excluded, sft_before = ambiguous_sft_families(candidate)
    sft, sft_removed = exclude_whole_families(candidate, sft_excluded)
    remaining_groups, sft_after = ambiguous_sft_families(sft)
    if remaining_groups:
        raise ValueError("crossfold SFT ambiguity remains")
    messages = extract_human_messages(human)
    text_clones, clone_receipt = crossfold_text_clone_families(messages)
    text_excluded = sft_excluded | text_clones
    seen = set()
    human_text, duplicate_receipt = human_text_records(messages, human_manifest["sha256"], seen)
    create_output_directory(args.output_dir)
    license_receipt = licenses(args.output_dir)
    wiki, wiki_receipt = collect_wikipedia(args.output_dir, seen, args.wiki_train_words, args.wiki_max_pages, args.download_workers)
    verify_frozen_inputs(frozen_inputs)
    # Apply family guard after global dedup, matching the frozen v2 policy.
    text_pool, text_removed = exclude_whole_families(human_text + wiki, text_excluded)
    text, holdouts = select_holdouts(text_pool)
    text_receipt = write_jsonl(args.output_dir / "text.jsonl", text)
    pool_receipt = write_jsonl(args.output_dir / "text-pool.jsonl", text_pool)
    sft_path = args.output_dir / "sft.jsonl"
    with sft_path.open("xb") as stream:
        for r in sft:
            stream.write(raw_lines[r["id"]])
    sft_path.chmod(0o600)
    # Pure canonical schema and exact dedup guards only; no evaluation.
    load_text_corpus(args.output_dir / "text.jsonl")
    load_text_corpus(args.output_dir / "text-pool.jsonl")
    load_corpus(sft_path)
    verify_frozen_inputs(frozen_inputs)
    manifest = {"purpose": "public-source Spanish language ingestion; not conversation acceptance", "trainingReady": True, "modelOrTokenizerCalls": False, "testScoringPerformed": False, "ownerChatsScanned": False, "inputs": [{"fileName": p.name, "sha256": fingerprint} for p, fingerprint in frozen_inputs.items()], "helperSha256": file_sha(Path(__file__)), "text": {**text_receipt, "splits": summary(text, "text"), "tokenCount": None}, "pool": {**pool_receipt, "splits": summary(text_pool, "text")}, "sft": {"file": sft_path.name, "sha256": file_sha(sft_path), "bytes": sft_path.stat().st_size, "splits": summary(sft, "messages")}, "ingestionGuards": {"sftCrosssplitBefore": sft_before, "sftCrosssplitAfter": sft_after, "excludedSftFamilyIds": sorted(sft_excluded), "excludedSftRecordIds": [r["id"] for r in sft_removed], "rawTextCrossfoldClones": clone_receipt, "globalExactDuplicateHumanMessagesRemovedByOriginalSplit": duplicate_receipt, "excludedTextFamilyIds": sorted(text_excluded), "excludedTextRecordIds": [r["id"] for r in text_removed], "foldsMoved": 0, "policy": "Whole ambiguous OASST families removed; remaining records/folds preserved. Global exact TEXT dedup before selection. No results-based selection."}, "holdoutPolicy": {"rank": "sha256(unidad-public-general-holdout-v1 + NUL + groupId)", "wholeRetainedFamiliesOnly": True, "characterCapPerSource": 250000, "trainAllRetained": True, "receipt": holdouts}, "wikipedia": wiki_receipt, "oasstImport": human_manifest, "licenseDeclarations": license_receipt, "limitations": ["Exact guards do not detect semantic paraphrases", "Heldout character caps alter family length distribution", "Positive upstream review does not certify perfect grammar or facts", "Wikipedia includes extraction artifacts, references, dated statements and mixed-script terms", "No article-level revision/rights certification", "Raw language text does not establish conversational competence"]}
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"stage": "complete", "output": str(args.output_dir), "textSha256": text_receipt["sha256"], "sftSha256": manifest["sft"]["sha256"], "tokenCount": None}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("oasst-corpus", "oasst-manifest", "historical-sft", "rich-sft", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("historical-sft-sha256", "rich-sft-sha256"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--wiki-train-words", type=int, default=2_000_000)
    parser.add_argument("--wiki-max-pages", type=int, default=240)
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 10_000 <= args.wiki_train_words <= 3_000_000 or not 4 <= args.wiki_max_pages <= 240 or args.wiki_max_pages % 4 or not 1 <= args.download_workers <= 4:
        parser.error("bounded download requires words 10000..3000000, pages 4..240 in batches of four, workers 1..4")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new path")
    os.umask(0o077)
    run(args)


if __name__ == "__main__":
    main()
