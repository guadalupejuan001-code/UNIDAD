"""Author literal-copy training data without reading models or evaluations.

Targets are original user-provided substrings, including whitespace and Unicode
composition. The helper is offline authoring only; inference never calls it.
An explicit fixed byte-level BPE measures representability and complete context.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import string
import tempfile
import unicodedata

from tokenizers import Tokenizer

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer


SPLITS = ("train", "validation", "test")
DATASET = "literal-copy-v1"
KINDS = ("fictional_name", "short_phrase", "alphanumeric_code", "short_title", "unicode_word")
OPENINGS = {
    "direct-update-recall": {
        "train": "Repite exactamente el texto entre comillas; conserva espacios:\n«{a}»",
        "validation": "Copia el interior de estas comillas sin cambiar espacios:\n«{a}»",
        "test": "Devuelve sólo lo que está entre comillas, tal cual:\n«{a}»"},
    "labels-update-recall": {
        "train": "Valor A: «{a}». Copia A exactamente, sin las comillas.",
        "validation": "Guarda A = «{a}». Devuelve sólo el valor A tal cual.",
        "test": "A contiene «{a}». Reproduce A sin cambiarlo ni añadir nada."}}
WORDS = ("luna", "río", "nube", "brisa", "ámbar", "café", "papel", "hoja", "sendero", "jardín",
         "cobre", "noche", "lluvia", "música", "árbol", "lápiz", "vela", "isla", "puente", "raíz",
         "trébol", "pluma", "tinta", "círculo", "cauce", "sueño", "nido", "viento", "bahía", "flor")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def render(values, split, template_id):
    """Derive every reply from an exact span already provided in a user turn."""
    if set(values) != {"a", "b", "c"} or split not in SPLITS or template_id not in OPENINGS:
        raise ValueError("Render requires A/B/C, a valid fold and a known template")
    a, b, c = (values[key] for key in ("a", "b", "c"))
    if any(type(span) is not str or not span.strip() or "«" in span or "»" in span for span in (a, b, c)):
        raise ValueError("Literal spans must be nonempty text without the delimiters")
    opening = OPENINGS[template_id][split].format(**values)
    if template_id == "direct-update-recall":
        pairs = [(opening, a),
                 (f"El texto nuevo es «{b}». Copia el nuevo, no el anterior.", b),
                 ("Recupera el primer texto. Repítelo exactamente, sin añadidos.", a)]
    else:
        pairs = [(opening, a),
                 (f"Valor B: «{b}». Copia sólo B exactamente.", b),
                 (f"Corrijo A: ahora es «{c}». Copia B, que sigue igual.", b),
                 ("Recupera el A actual, después de la corrección. Cópialo exactamente.", c)]
    return [turn for user, reply in pairs for turn in
            ({"role": "user", "content": user}, {"role": "assistant", "content": reply})]


def _phonetic(rng, accented=False):
    vowels = "aeiouáéíóú" if accented else "aeiou"
    return "".join(rng.choice("bcdfghjklmnprstvyz") + rng.choice(vowels)
                   for _ in range(rng.randint(2, 4))).capitalize()


def _span(rng, kind):
    if kind == "fictional_name":
        value = _phonetic(rng, accented=True)
    elif kind == "short_phrase":
        value = " ".join(rng.sample(WORDS, rng.randint(2, 4)))
    elif kind == "alphanumeric_code":
        value = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(1, 12)))
    elif kind == "short_title":
        value = "Proyecto " + rng.choice(WORDS) + " " + _phonetic(rng)
    else:
        value = rng.choice(("Ñ", "É", "ü", "ó", "ç", "á")) + _phonetic(rng, accented=True)
    whitespace = rng.randrange(7)
    if whitespace == 1:
        value = " " + value
    elif whitespace == 2:
        value += " "
    elif whitespace == 3:
        value = "  " + value + "  "
    elif whitespace == 4:
        value = value.replace(" ", "  ")
    elif whitespace == 5:
        value = unicodedata.normalize("NFD", value)
    elif whitespace == 6:
        value = "\u00a0" + value + "\u00a0"
    return value


def generate(tokenizer, *, seed=20261015, worlds=1000, context_length=256):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in 0..4294967295")
    if type(worlds) is not int or not 40 <= worlds <= 2000 or worlds % 40:
        raise ValueError("worlds must be a multiple of 40 in 40..2000")
    if type(context_length) is not int or not 32 <= context_length <= 4096:
        raise ValueError("context_length must be an integer in 32..4096")
    validate_tokenizer(tokenizer, strict=True)
    rng = random.Random(seed)
    episodes, spans_seen, reasons = [], set(), Counter()
    # Values are generated before fold assignment. No fold/index is encoded in
    # the literal strings or prompts. Whole triples, variants and prefixes move
    # together; lexical-kind quotas are identical across folds.
    for index in range(worlds):
        kinds = {key: KINDS[(index + slot) % len(KINDS)] for slot, key in enumerate(("a", "b", "c"))}
        for attempt in range(2000):
            values = {key: _span(rng, kind) for key, kind in kinds.items()}
            if len(set(values.values())) != 3 or set(values.values()) & spans_seen:
                reasons["duplicate_span_candidates_regenerated"] += 1
                continue
            lengths = {key: len(tokenizer.encode(value, add_special_tokens=False).ids) for key, value in values.items()}
            if any(not 1 <= length <= 20 for length in lengths.values()):
                reasons["span_length_candidates_regenerated"] += 1
                continue
            if any(tokenizer.decode(tokenizer.encode(value, add_special_tokens=False).ids,
                                    skip_special_tokens=False) != value for value in values.values()):
                raise ValueError("Fixed tokenizer changes an authored literal span")
            full_lengths = [len(dialogue_tokens(render(values, split, template), tokenizer)[0])
                            for split in SPLITS for template in OPENINGS]
            if max(full_lengths) > context_length + 1:
                reasons["whole_world_context_candidates_regenerated"] += 1
                continue
            episodes.append({"values": values, "kinds": kinds, "lengths": lengths,
                             "stratum": index % len(KINDS), "worldId": sha(canonical(values))[:24]})
            spans_seen.update(values.values())
            break
        else:
            raise ValueError("Cannot construct the requested full-context worlds")
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["stratum"]].append(episode)
    records = []
    for stratum in sorted(groups):
        bank = sorted(groups[stratum], key=lambda episode: sha(("literal-copy-fold-v1\0" + episode["worldId"]).encode("utf-8")))
        reserved = len(bank) // 8
        for index, episode in enumerate(bank):
            split = "train" if index < len(bank) - 2 * reserved else "validation" if index < len(bank) - reserved else "test"
            group_id = "literal-copy-v1-" + episode["worldId"]
            for template in OPENINGS:
                messages = render(episode["values"], split, template)
                slots = ("a", "b", "a") if template == "direct-update-recall" else ("a", "b", "b", "c")
                user_turns = (1, 2, 1) if template == "direct-update-recall" else (1, 2, 2, 3)
                for prefix, slot in enumerate(slots, 1):
                    record = {"id": f"{group_id}-{template}-p{prefix}", "groupId": group_id, "split": split,
                              "source": {"dataset": DATASET, "kind": "authored-synthetic", "skill": "literal_copy",
                                         "templateId": template, "worldId": episode["worldId"], "prefixPairs": prefix,
                                         "targetSlot": slot, "targetFromUserTurn": user_turns[prefix - 1],
                                         "spanTokenLengths": episode["lengths"], "spanKinds": episode["kinds"],
                                         "targetSha256": sha(episode["values"][slot].encode("utf-8")),
                                         "fictional": True, "privateConversationUsed": False, "modelResponsesUsed": False},
                              "messages": messages[:2 * prefix]}
                    validate_record(record)
                    records.append(record)
    return records, reasons


def exclusive_write(path, raw):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Existing outputs are immutable; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".literal-copy-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(tokenizer_path, output_path, manifest_path, *, seed=20261015, worlds=1000, context_length=256):
    tokenizer_path, output, manifest = map(Path, (tokenizer_path, output_path, manifest_path))
    if len({path.resolve() for path in (tokenizer_path, output, manifest)}) != 3:
        raise ValueError("Tokenizer and both outputs must have distinct paths")
    if any(path.exists() or path.is_symlink() for path in (output, manifest)):
        raise ValueError("Existing outputs are immutable; choose new paths")
    raw_tokenizer = tokenizer_path.read_bytes()
    if len(raw_tokenizer) > 16_777_216:
        raise ValueError("Tokenizer exceeds its byte limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(raw_tokenizer.decode("utf-8")), strict=True)
    records, reasons = generate(tokenizer, seed=seed, worlds=worlds, context_length=context_length)
    prompt_folds, prompt_targets, exact, groups = {}, {}, set(), {}
    lengths = []
    for record in records:
        prompt = canonical(record["messages"][:-1])
        target = record["messages"][-1]["content"]
        if prompt in prompt_folds and prompt_folds[prompt] != record["split"]:
            raise ValueError("Identical full prompt occurs across folds")
        if prompt in prompt_targets and prompt_targets[prompt] != target:
            raise ValueError("Identical full prompt has conflicting literal targets")
        prompt_folds[prompt], prompt_targets[prompt] = record["split"], target
        encoded = canonical(record["messages"])
        if encoded in exact:
            raise ValueError("Duplicate dialogue in authored corpus")
        exact.add(encoded)
        if record["groupId"] in groups and groups[record["groupId"]] != record["split"]:
            raise ValueError("A whole world crosses folds")
        groups[record["groupId"]] = record["split"]
        length = len(dialogue_tokens(record["messages"], tokenizer)[0])
        if length > context_length + 1:
            raise ValueError("An authored full dialogue exceeds its context")
        lengths.append(length)
    raw = b"".join(canonical(record) + b"\n" for record in records)
    ordered = sorted(lengths)
    values_by_fold = {split: {record["messages"][-1]["content"] for record in records if record["split"] == split} for split in SPLITS}
    kinds = {split: Counter() for split in SPLITS}
    for record in records:
        kinds[record["split"]][record["source"]["spanKinds"][record["source"]["targetSlot"]]] += 1
    report = {"schemaVersion": 1, "dataset": DATASET, "purpose": "auxiliary-training-curriculum",
              "sha256": sha(raw), "seed": seed, "worlds": worlds, "records": len(records),
              "splits": dict(Counter(record["split"] for record in records)),
              "groupSplits": dict(Counter(groups.values())),
              "prefixPairs": dict(Counter(str(record["source"]["prefixPairs"]) for record in records)),
              "templates": {template: dict(Counter(record["split"] for record in records if record["source"]["templateId"] == template)) for template in OPENINGS},
              "targetKindsBySplit": {split: dict(counts) for split, counts in kinds.items()},
              "uniqueLiteralTargets": len(set.union(*values_by_fold.values())),
              "uniqueLiteralTargetsBySplit": {split: len(values) for split, values in values_by_fold.items()},
              "exactLiteralTargetsSharedAcrossFolds": len({value for value in set.union(*values_by_fold.values()) if sum(value in values for values in values_by_fold.values()) > 1}),
              "spanTokenLengths": dict(Counter(str(length) for record in records for length in record["source"]["spanTokenLengths"].values())),
              "fullDialogueTokens": {"min": min(ordered), "median": ordered[len(ordered)//2], "p95": ordered[math.ceil(.95*len(ordered))-1], "max": max(ordered)},
              "tokenizer": {"sha256": sha(raw_tokenizer), "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
              "authoring": {"contextLength": context_length, "fullDialogueTokenLimit": context_length + 1,
                            "dialogsTruncated": False, "prefixSupervision": "last_reply_only; every 1/2/3 or 1/2/3/4-pair prefix exported",
                            "targetRule": "Exact original substring from a preceding user message; no strip, normalization or token reconstruction",
                            "splitPolicy": "75/12.5/12.5 quotas within each lexical-kind stratum; opaque hash order after values are authored",
                            "literalSpanPolicy": "Three distinct globally unique spans per world; shared alphabet/word components, unseen full spans across folds",
                            "spanTokenLimit": [1, 20], "allSpansRoundTrip": True,
                            "existingCorpusRead": False, "evaluationRead": False, "modelRead": False,
                            "privateConversationUsed": False, "modelResponsesUsed": False,
                            "mergeCollisionAudit": "Must be run separately against existing corpora without changing old folds"},
              "reasons": dict(reasons), "generatorSha256": sha(Path(__file__).read_bytes()),
              "limitations": ["Procedural literal-copy practice is not evidence of useful conversation.",
                              "Prior assistant messages are authored correct replies, not actual model generations.",
                              "Whole spans are held out; character and word components are shared.",
                              "No model, development response, existing held-out corpus or final acceptance case is read."]}
    exclusive_write(output, raw)
    load_corpus(output)
    exclusive_write(manifest, canonical(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--worlds", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        report = build(args.tokenizer, args.output, args.manifest, seed=args.seed, worlds=args.worlds, context_length=args.context_length)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
