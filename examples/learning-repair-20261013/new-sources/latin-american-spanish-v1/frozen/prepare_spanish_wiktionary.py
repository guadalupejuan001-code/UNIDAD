#!/usr/bin/env python3
"""Acquire a bounded, attributed Spanish Wikcionario vocabulary corpus.

This stdlib-only data helper never reads model weights, private conversations,
evaluation cases, or a tokenizer. Every accepted page is tied to its original
revision. It extracts only Spanish definitions and uncredited source examples;
translations, quotations, references, etymologies, and unresolved templates are
not training text. A snapshot of the fetched source is retained for replays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import http.client
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ia_propia.conversation.data import assign_split
from ia_propia.conversation.language_data import validate_text_record

API = "https://es.wiktionary.org/w/api.php"
LICENSE = "CC-BY-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/deed.es"
USER_AGENT = "UNIDAD-spanish-vocabulary/1.0 (bounded attributed public corpus)"
NETWORK_ERRORS = (HTTPError, URLError, TimeoutError, OSError, http.client.HTTPException)
ACQUISITION_ERRORS = NETWORK_ERRORS + (RuntimeError,)
# Several sort-key anchors avoid taking only the beginning of the alphabet.
ANCHORS = ("A", "C", "E", "H", "L", "M", "P", "R", "S", "T", "V")
MAJOR_CATEGORIES = ("ES:Sustantivos", "ES:Adjetivos", "ES:Verbos", "ES:Adverbios")
SMALL_CATEGORIES = ("ES:Artículos", "ES:Pronombres", "ES:Preposiciones", "ES:Conjunciones", "ES:Interjecciones", "ES:Locuciones")
SEEDS = "ser estar haber tener hacer poder decir ir ver dar saber querer llegar pasar deber poner parecer quedar creer llevar dejar seguir encontrar llamar venir pensar salir volver tomar conocer vivir sentir tratar mirar contar empezar esperar buscar existir entrar trabajar escribir perder producir ocurrir entender pedir recibir recordar terminar permitir aparecer conseguir comenzar servir sacar necesitar mantener resultar leer caer cambiar presentar crear abrir considerar oír hablar español palabra oración nombre verbo pronombre sustantivo adjetivo adverbio artículo preposición conjunción interjección singular plural femenino masculino pasado presente futuro yo tú él ella nosotros vosotros ustedes usted ellos ellas mí ti sí me te se nos os le les lo la los las mi tu su nuestro vuestro este ese aquel esto eso aquello aquí ahí allí acá allá ahora antes después siempre nunca todavía ya también tampoco sí no quizá quizás por para con sin sobre bajo entre hacia hasta desde durante mediante salvo según ante tras a en de y e o u pero aunque porque pues que si cuando donde como cuánto quién cual cada todo nada algo alguien nadie uno dos tres mucho poco más menos muy tanto tan bastante demasiado mismo otro nuevo bueno malo grande pequeño primero último hola adiós gracias perdón favor acuerdo corrección aclaración pregunta respuesta conversación memoria historia persona gente cosa casa agua comida tiempo día noche año mundo mano ojo voz escuchar comprender explicar preguntar responder corregir ayudar aprender enseñar repetir olvidar negar afirmar ayer hoy mañana pronto tarde temprano lejos cerca dentro fuera encima debajo delante detrás mejor peor bien mal solamente incluso mientras entonces luego además embargo obstante".split()
SEEDS += "hablo hablas habla hablamos habláis hablan hablé hablaste habló hablaron hablaba hablábamos hablaré hablarás hablaría hablado hablando soy eres es somos sois son fui fuiste fue fuimos fueron era éramos seré serás sería sido siendo estoy estás está estamos estáis están estuve estuvo estuvieron estaba estarán estado estando tengo tienes tiene tenemos tenéis tienen tuve tuvo tuvieron tenía tendrán tenido teniendo hago haces hace hacemos hacéis hacen hice hizo hicieron hacía hará hecho haciendo voy vas va vamos vais van iba irán ido yendo buenas buena buenos casas palabras preguntas respuestas conversaciones personas días noches años ojos manos voces nombres verbos pronombres sustantivos adjetivos amigos amigas niños niñas nuevos nuevas pequeños pequeñas primeros primeras últimos últimas nuestras nuestros vuestras vuestros estas estos esas esos aquellas aquellos correctos correctas mejores peores grandes felices fáciles difíciles".split()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> dict:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}


class BoundedSource:
    def __init__(self, output: Path, seconds: float, network_bytes: int):
        self.output, self.started = output, time.monotonic()
        self.seconds, self.network_limit = seconds, network_bytes
        self.network_bytes, self.requests = 0, []
        self.last_request = 0.0
        (output / "responses").mkdir()

    def fetch(self, url: str, max_bytes: int = 2 << 20) -> bytes:
        if time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("acquisition wall-clock budget reached")
        available = min(max_bytes, self.network_limit - self.network_bytes)
        if available <= 0:
            raise RuntimeError("acquisition network budget reached")
        timeout = min(25, max(1, self.seconds - (time.monotonic() - self.started)))
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/html"})
        for attempt in range(3):
            try:
                delay = max(0, self.last_request + 3 - time.monotonic())
                if time.monotonic() - self.started + delay >= self.seconds:
                    raise RuntimeError("acquisition wall-clock budget reached before rate-limited request")
                if delay:
                    time.sleep(delay)
                self.last_request = time.monotonic()
                with urlopen(request, timeout=timeout) as response:
                    raw = response.read(available + 1)
                    self.network_bytes += len(raw)
                    if len(raw) > available:
                        raise RuntimeError("source response exceeds remaining byte budget")
                    headers = {k: response.headers.get(k) for k in ("Content-Type", "Last-Modified", "ETag") if response.headers.get(k)}
                    final_url = response.url
                break
            except NETWORK_ERRORS as exc:
                failed = {"url": url, "attempt": attempt + 1, "error": str(exc)[:250], "elapsedSeconds": time.monotonic() - self.started}
                with (self.output / "request-errors.jsonl").open("ab") as stream:
                    stream.write((json.dumps(failed, ensure_ascii=False) + "\n").encode())
                pause = 2 + attempt
                if isinstance(exc, HTTPError) and exc.code == 429:
                    advertised = exc.headers.get("Retry-After", "")
                    pause = max(30, int(advertised)) if advertised.isdigit() else 30
                if attempt == 2 or time.monotonic() - self.started + pause + 5 >= self.seconds or isinstance(exc, HTTPError) and exc.code not in {429, 502, 503, 504}:
                    raise
                time.sleep(pause)
                timeout = min(25, max(1, self.seconds - (time.monotonic() - self.started)))
        path = self.output / "responses" / f"{len(self.requests):04d}.response"
        with path.open("xb") as stream:
            stream.write(raw)
        receipt = {"url": url, "finalUrl": final_url, "responsePath": str(path), "bytes": len(raw), "sha256": sha(raw), "headers": headers}
        self.requests.append(receipt)
        with (self.output / "requests.jsonl").open("ab") as stream:
            stream.write((json.dumps(receipt, ensure_ascii=False) + "\n").encode())
        return raw

    def query(self, **kwargs):
        params = {"action": "query", "format": "json", "formatversion": 2, "maxlag": 5, **kwargs}
        value = json.loads(self.fetch(API + "?" + urlencode(params)))
        if "error" in value:
            raise RuntimeError("MediaWiki API error: " + str(value["error"].get("code")))
        return value


class ReplayedSource(BoundedSource):
    """Replay verified fixed responses; there is no network fallback."""
    def __init__(self, output: Path, seconds: float, network_bytes: int,
                 original: Path, expected_manifest_sha256: str):
        super().__init__(output, seconds, network_bytes)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) or file_sha(original / "manifest.json") != expected_manifest_sha256:
            raise ValueError("replay manifest identity differs")
        self.original = original
        manifest = json.loads((original / "manifest.json").read_text())
        self.retained = {}
        for receipt in manifest["acquisition"]["requests"]:
            if receipt["url"] in self.retained and self.retained[receipt["url"]]["sha256"] != receipt["sha256"]:
                raise ValueError("same source URL has different cached responses")
            self.retained[receipt["url"]] = receipt
        self.replay_info = {"directory": str(original), "manifestSha256": expected_manifest_sha256,
                            "originalNetworkBytes": manifest["acquisition"]["networkBytes"],
                            "originalElapsedSeconds": manifest["acquisition"]["elapsedSeconds"],
                            "originalStoppedReason": manifest["acquisition"]["stoppedReason"],
                            "originalScriptSha256": manifest["preparationScriptSha256"], "verifiedResponseReads": 0, "readBytes": 0}

    def fetch(self, url: str, max_bytes: int = 2 << 20) -> bytes:
        if time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("offline replay wall-clock budget reached")
        receipt = self.retained.get(url)
        if receipt is None:
            raise RuntimeError("frozen response absent; original bounded acquisition ended here")
        name = Path(receipt["responsePath"]).name
        if not re.fullmatch(r"\d{4}\.response", name):
            raise ValueError("invalid frozen source response basename")
        original_path = self.original / "responses" / name
        raw = original_path.read_bytes()
        if len(raw) != receipt["bytes"] or len(raw) > max_bytes or sha(raw) != receipt["sha256"]:
            raise ValueError("frozen source response bytes/hash differ")
        target = self.output / "responses" / f"{len(self.requests):04d}.response"
        with target.open("xb") as stream:
            stream.write(raw)
        replay_receipt = {**receipt, "responsePath": str(target), "replayedFrom": str(original_path)}
        self.requests.append(replay_receipt)
        with (self.output / "requests.jsonl").open("ab") as stream:
            stream.write((json.dumps(replay_receipt, ensure_ascii=False) + "\n").encode())
        self.replay_info["verifiedResponseReads"] += 1
        self.replay_info["readBytes"] += len(raw)
        return raw


def spanish_section(raw: str) -> str | None:
    headings = list(re.finditer(r"(?m)^==(?!=)\s*(.*?)\s*(?<!=)==\s*$", raw))
    for i, match in enumerate(headings):
        if re.fullmatch(r"\{\{\s*lengua\s*\|\s*es\s*\}\}", match.group(1), re.I):
            return raw[match.end():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]
    return None


def template_parts(raw: str) -> tuple[str, list[str], dict[str, str]]:
    parts = raw.split("|")
    args, named = [], {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            named[key.strip().lower()] = value.strip()
        else:
            args.append(part.strip())
    return parts[0].strip().lower(), args, named


def inline(raw: str) -> str | None:
    """Conservative rendering: unknown content templates reject that segment."""
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    if re.search(r"<ref\b|\{\{(?:cita|DRAE|DLE|DLC|Labernia|Autoridades)", raw, re.I):
        return None
    raw = re.sub(r"\[\[(?:Archivo|File|Imagen|Categoría):[^\]]*\]\]", "", raw, flags=re.I)
    raw = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", lambda m: m.group(2) or m.group(1).split("#", 1)[0], raw)
    for _ in range(8):
        if "{{" not in raw:
            break
        invalid = [False]

        def replace(match):
            name, args, named = template_parts(match.group(1))
            if name in {"plm", "m", "enlace", "link", "impropia", "coloquial", "definición", "definicion"} and args:
                return args[0]
            if name in {"l", "l+", "l-", "l-l", "l1", "l2"} and len(args) >= 2 and args[0] == "es":
                return named.get("alt", args[1])
            if name in {"csem", "ámbito", "ambito", "uso", "escopo", "contexto", "taxón", "taxon"}:
                return ""
            if name in {"onomatopeya", "onom", "pronombre", "plural", "singular", "femenino", "masculino"} and args:
                return args[-1]
            if name == "comparativo" and args:
                return f"Comparativo de «{args[0]}»"
            if name == "forma sustantivo plural" and len(args) == 1 and named.get("leng", "es") == "es":
                return f"Plural del sustantivo «{args[0]}»"
            if name in {"forma adjetivo", "forma sustantivo"} and args and named.get("leng", "es") == "es":
                grammar = args[1:]
                if all(value in {"femenino", "masculino", "singular", "plural"} for value in grammar):
                    return f"Forma {' '.join(grammar)} del {'adjetivo' if name == 'forma adjetivo' else 'sustantivo'} «{args[0]}»"
            if name == "forma verbo" and args and named.get("leng", "es") == "es":
                person = named.get("p", args[1] if len(args) > 1 else "")
                tense = named.get("t", args[2] if len(args) > 2 else "")
                mode = named.get("m", args[3] if len(args) > 3 else "")
                persons = {"1s": "Primera persona del singular", "2s": "Segunda persona del singular", "3s": "Tercera persona del singular", "1p": "Primera persona del plural", "2p": "Segunda persona del plural", "3p": "Tercera persona del plural"}
                tenses = {"pres": "presente", "presente": "presente", "pret ind": "pretérito perfecto simple", "pret": "pretérito perfecto simple", "pretérito": "pretérito perfecto simple", "pret imp": "pretérito imperfecto", "imperfecto": "pretérito imperfecto", "fut": "futuro", "futuro": "futuro", "cond": "condicional", "condicional": "condicional"}
                modes = {"ind": "indicativo", "indicativo": "indicativo", "sub": "subjuntivo", "subjuntivo": "subjuntivo", "imp": "imperativo", "imperativo": "imperativo"}
                if person in persons and tense in tenses and mode in modes:
                    return f"{persons[person]} del {tenses[tense]} de {modes[mode]} del verbo «{args[0]}»"
            invalid[0] = True
            return ""

        raw, count = re.subn(r"\{\{([^{}]*)\}\}", replace, raw)
        if invalid[0] or not count:
            return None
    if "{{" in raw or "}}" in raw or "[[" in raw or "http" in raw:
        return None
    raw = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    raw = re.sub(r"'{2,5}", "", raw)
    raw = " ".join(raw.split()).strip(" :;-")
    if not raw or not any(c.isalpha() for c in raw) or len(raw) > 1800:
        return None
    if any(ord(c) < 32 for c in raw) or re.search(r"[{}|]", raw):
        return None
    return raw


def family_key(title: str) -> str:
    # Accent/case variants and reflexive variants remain together conservatively.
    value = "".join(c for c in unicodedata.normalize("NFKD", title.casefold()) if not unicodedata.combining(c))
    value = re.sub(r"\s+", " ", value).strip()
    if value.endswith(("arse", "erse", "irse")):
        value = value[:-2]
    return value


def extract(title: str, section: str) -> tuple[str | None, set[str], dict]:
    lines, links, stats = [f"Palabra: {title}."], {family_key(title)}, Counter()
    part = None
    senses = 0
    for raw in section.splitlines():
        heading = re.match(r"^={3,6}\s*\{\{([^{}]+)\}\}\s*={3,6}\s*$", raw)
        if heading:
            name, args, _ = template_parts(heading.group(1))
            if args and args[0] == "es" and any(name.startswith(s) for s in ("sustantivo", "adjetivo", "verbo", "adverbio", "pronombre", "preposición", "conjunción", "interjección", "artículo", "locución", "forma")):
                part = name
                lines.append(f"Categoría gramatical: {name}.")
                stats["grammaticalHeadings"] += 1
            else:
                part = None
            continue
        plain_heading = re.match(r"^={3,6}\s*(.*?)\s*={3,6}\s*$", raw)
        if plain_heading:
            label = plain_heading.group(1).strip().casefold()
            part = label if label.startswith(("forma verbal", "forma adjetiva", "forma sustantiva")) else None
            if part:
                lines.append(f"Categoría gramatical: {part}.")
                stats["grammaticalHeadings"] += 1
            continue
        # Family links from explicitly declared orthographic/pronoun variants.
        if raw.startswith("{{pron-graf"):
            _, _, named = template_parts(raw.strip().removeprefix("{{").removesuffix("}}"))
            links.update(family_key(value) for key, value in named.items() if re.fullmatch(r"v\d*", key) and value and not re.search(r"[{}\[\]|]", value))
        if part is None:
            continue
        definition = re.match(r"^;\s*(\d+)(.*?)\s*:\s*(.*)$", raw)
        if definition:
            value = inline(definition.group(3))
            if value and len(value.split()) >= 2:
                lines.append(f"Significado {definition.group(1)}: {value}")
                stats["definitions"] += 1
                senses += 1
            else:
                stats["discardedDefinitions"] += 1
            # Inflection templates are not expanded guessed grammar: they only
            # inform grouping when they contain an explicit Spanish lemma.
            for m in re.finditer(r"\{\{([^{}]+)\}\}", definition.group(3)):
                name, args, named = template_parts(m.group(1))
                if name.startswith(("forma", "f.", "plural", "femenino")) or name == "comparativo":
                    # Only the declared lemma is a lexical relationship. Mood,
                    # tense and gender labels are not roots that join unrelated
                    # words into one giant family.
                    declared = [named.get("lema", ""), named.get("lemma", "")]
                    if args:
                        declared.append(args[1] if name.startswith("f.") and args[0] in {"f", "m", "p", "s", "es"} and len(args) > 1 else args[0])
                    for candidate in declared:
                        if candidate and re.fullmatch(r"[a-záéíóúñü -]{2,60}", candidate, re.I) and candidate not in {"es", "f", "m", "s", "p", "plural", "singular", "femenino", "masculino"}:
                            links.add(family_key(candidate))
                            stats["declaredFormLinks"] += 1
        elif senses and raw.strip().startswith("{{ejemplo|") and raw.strip().endswith("}}"):
            name, args, named = template_parts(raw.strip()[2:-2])
            # Cited songs/books/web quotations have different reuse conditions;
            # exclude them instead of treating the site's overall license as a
            # separate permission for the quoted underlying work.
            if name == "ejemplo" and len(args) == 1 and not named:
                value = inline(args[0])
                if value and 3 <= len(value.split()) <= 70:
                    lines.append(f"Ejemplo de uso: {value}")
                    stats["sourceExamples"] += 1
    if not senses:
        return None, links, dict(stats)
    # Empty headings are not useful targets.
    cleaned = [line for i, line in enumerate(lines) if not line.startswith("Categoría gramatical:") or (i + 1 < len(lines) and not lines[i + 1].startswith("Categoría gramatical:"))]
    if cleaned[-1].startswith("Categoría gramatical:"):
        cleaned.pop()
    return "\n".join(cleaned), links, dict(stats)


def write_jsonl(path: Path, records: list[dict]) -> dict:
    with path.open("xb") as stream:
        for record in records:
            stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode())
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path), "records": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--network-mib", type=int, default=32)
    parser.add_argument("--corpus-mib", type=int, default=4)
    parser.add_argument("--per-anchor", type=int, default=35)
    parser.add_argument("--max-pages", type=int, default=2500)
    parser.add_argument("--replay-from", type=Path)
    parser.add_argument("--replay-manifest-sha256")
    args = parser.parse_args()
    if not (1 <= args.seconds <= 600 and 1 <= args.network_mib <= 32 and 1 <= args.corpus_mib <= 4 and 1 <= args.per_anchor <= 60 and 1 <= args.max_pages <= 4000):
        parser.error("limits exceed the explicitly bounded acquisition")
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    args.output.mkdir(parents=True, exist_ok=False)
    if bool(args.replay_from) != bool(args.replay_manifest_sha256):
        parser.error("replay directory and expected manifest hash must be supplied together")
    source = ReplayedSource(args.output, args.seconds, args.network_mib << 20, args.replay_from, args.replay_manifest_sha256) if args.replay_from else BoundedSource(args.output, args.seconds, args.network_mib << 20)
    counts, errors, pages = Counter(), [], []
    site = source.query(meta="siteinfo", siprop="general|rightsinfo")
    rights = site["query"]["rightsinfo"]
    if site["query"]["general"].get("lang") != "es" or rights.get("url") != LICENSE_URL:
        raise RuntimeError("original Spanish site does not declare the expected license")
    dump(args.output / "siteinfo.json", site)
    for url in (LICENSE_URL, "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use", "https://www.mediawiki.org/wiki/API:Etiquette"):
        try:
            source.fetch(url, 1 << 20)
        except ACQUISITION_ERRORS as exc:
            errors.append({"stage": "license-reference", "url": url, "error": str(exc)[:250]})
    titles = set(SEEDS)
    selection = []
    stopped = None
    try:
        stop_categories = False
        for category in SMALL_CATEGORIES + MAJOR_CATEGORIES:
            anchors = ANCHORS if category in MAJOR_CATEGORIES else ("",)
            for anchor in anchors:
                try:
                    value = source.query(list="categorymembers", cmtitle="Categoría:" + category, cmnamespace=0, cmtype="page", cmlimit=args.per_anchor if anchor else 500, cmstartsortkeyprefix=anchor)
                except ACQUISITION_ERRORS as exc:
                    errors.append({"stage": "category-list", "category": category, "anchor": anchor, "error": str(exc)[:250]})
                    # Preserve collected titles and allocate the remaining time
                    # to revision acquisition instead of losing the whole run.
                    stop_categories = True
                    break
                members = value.get("query", {}).get("categorymembers", [])
                selected = [p["title"] for p in members if p.get("ns") == 0 and not re.search(r"[:|{}\[\]]", p["title"])]
                titles.update(selected)
                selection.append({"category": category, "sortKeyAnchor": anchor, "selected": len(selected)})
            if stop_categories:
                break
        # A stable content-independent permutation keeps a page cap from
        # throwing away the later alphabet anchors.
        selected_titles = sorted(titles, key=lambda x: (sha(x.encode()), x))[:args.max_pages]
        for cursor in range(0, len(selected_titles), 30):
            value = source.query(titles="|".join(selected_titles[cursor:cursor + 30]), prop="revisions", rvprop="ids|timestamp|content", rvslots="main")
            for page in value.get("query", {}).get("pages", []):
                counts["fetchedPages"] += 1
                if page.get("missing") or page.get("ns") != 0 or not page.get("revisions"):
                    counts["missingOrInvalidPages"] += 1
                    continue
                revision = page["revisions"][0]
                raw = revision.get("slots", {}).get("main", {}).get("content", "")
                section = spanish_section(raw)
                if section is None:
                    counts["nonSpanishPages"] += 1
                    continue
                text, links, stats = extract(page["title"], section)
                counts.update(stats)
                if text is None:
                    counts["pagesWithoutUsableDefinitions"] += 1
                    continue
                pages.append({"title": page["title"], "pageId": page["pageid"], "revisionId": revision["revid"], "revisionTimestamp": revision["timestamp"], "wikitextSha256": sha(raw.encode()), "sectionSha256": sha(section.encode()), "text": text, "links": sorted(links), "statistics": stats})
            print(json.dumps({"progressPages": counts["fetchedPages"], "usablePages": len(pages), "networkBytes": source.network_bytes, "elapsedSeconds": round(time.monotonic() - source.started, 2)}, ensure_ascii=False), flush=True)
    except ACQUISITION_ERRORS as exc:
        stopped = str(exc)[:250]
        errors.append({"stage": "acquisition", "error": stopped})
    # Connected components join declared variants/inflections before assigning
    # folds, so the same family cannot leak through separate lemma spellings.
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    for page in pages:
        key = family_key(page["title"])
        for linked in page["links"]:
            union(key, linked)
    records, duplicates, bytes_used, text_hashes = [], [], 0, set()
    for page in pages:
        group = "wiktionary:es:lexical-family:" + find(family_key(page["title"]))
        record = {"id": f"wiktionary:es:{page['pageId']}:{page['revisionId']}", "groupId": group, "split": assign_split(group),
                  "source": {"dataset": "Wikcionario/es", "kind": "public-wiki-collaborative", "language": "es", "license": LICENSE, "licenseUrl": LICENSE_URL,
                             "title": page["title"], "pageId": page["pageId"], "revisionId": page["revisionId"], "revisionTimestamp": page["revisionTimestamp"],
                             "url": "https://es.wiktionary.org/wiki/" + quote(page["title"].replace(" ", "_"), safe=""), "revisionUrl": "https://es.wiktionary.org/w/index.php?oldid=" + str(page["revisionId"]),
                             "historyUrl": "https://es.wiktionary.org/w/index.php?title=" + quote(page["title"].replace(" ", "_"), safe="") + "&action=history",
                             "attribution": "Wikcionario contributors; see the retained revision and page history", "wikitextSha256": page["wikitextSha256"], "spanishSectionSha256": page["sectionSha256"],
                             "extraction": "Spanish grammatical definition lines and uncredited source examples only; conservative inline cleanup; quoted references, translations and unknown templates excluded", "statistics": page["statistics"]}, "text": page["text"]}
        validate_text_record(record)
        digest = sha(record["text"].encode())
        if digest in text_hashes:
            duplicates.append(record["id"])
            continue
        raw_bytes = len((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        if bytes_used + raw_bytes > (args.corpus_mib << 20):
            counts["excludedByCorpusByteLimit"] += 1
            continue
        bytes_used += raw_bytes
        text_hashes.add(digest)
        records.append(record)
    if not records:
        dump(args.output / "failure-receipt.json", {"status": "failed-before-corpus", "errors": errors, "networkBytes": source.network_bytes, "requests": source.requests, "statistics": dict(counts), "scriptSha256": file_sha(Path(__file__))})
        raise RuntimeError("no usable source records acquired; retained raw receipts identify the blockage")
    train_validation = [r for r in records if r["split"] != "test"]
    test = [r for r in records if r["split"] == "test"]
    files = {"trainValidation": write_jsonl(args.output / "wiktionary-es.train-validation.jsonl", train_validation), "test": write_jsonl(args.output / "wiktionary-es.test.jsonl", test), "rawPages": write_jsonl(args.output / "acquired-pages.jsonl", pages)}
    groups = defaultdict(set)
    for record in records:
        groups[record["groupId"]].add(record["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise RuntimeError("lexical family crosses folds")
    split_counts = Counter(r["split"] for r in records)
    manifest = {"schemaVersion": 1, "dataset": "unidad-spanish-wiktionary-v1", "objective": "causal_lm", "source": {"dataset": "Wikcionario/es", "originalApi": API, "language": "es", "rightsinfo": rights, "siteinfo": {"path": str(args.output / "siteinfo.json"), "sha256": file_sha(args.output / "siteinfo.json")}},
                "acquisition": {"elapsedSeconds": time.monotonic() - source.started, "networkBytes": source.network_bytes, "requests": source.requests, "stoppedReason": stopped, "errors": errors, "seedTitles": SEEDS, "categorySelections": selection, "requestedUniqueTitles": len(titles)},
                "limits": {"wallSeconds": args.seconds, "networkBytes": args.network_mib << 20, "corpusBytes": args.corpus_mib << 20, "diskBytes": 128 << 20, "addressSpaceBytes": 512 << 20, "cpuAffinityCount": 1, "maxPages": args.max_pages},
                "files": files, "corpusSha256": files["trainValidation"]["sha256"], "splitCounts": {"train": split_counts["train"], "validation": split_counts["validation"], "test": split_counts["test"]}, "trainingFileSplitCounts": {"train": split_counts["train"], "validation": split_counts["validation"], "test": 0}, "records": len(records), "families": len(groups), "textBytes": sum(len(r["text"].encode()) for r in records), "words": sum(len(r["text"].split()) for r in records), "statistics": dict(counts), "exactDuplicateRecordsExcluded": duplicates,
                "partitionMethod": "Connected components of declared lemma/form/orthographic variants plus accent/case/reflexive normalization; assign_split whole lexical family; 80/10/10 deterministic family folds", "testUsedForTraining": False, "testTokenized": False, "tokenizerFitted": False, "modelLoaded": False, "sources": [{"dataset": "Wikcionario/es", "license": LICENSE, "language": "es", "originalApi": API}],
                "limitations": ["Bounded alphabetically stratified sample, not the complete Spanish language or all Wikcionario", "Extraction omits unknown templates, sourced quotations, translations, etymologies and generated conjugation tables", "Definitions are source statements, not independently reviewed factual truths", "License metadata records the original site's declaration; quoted underlying works are conservatively excluded", "Newly prepared data, not evidence of an already executed model training stage or conversational competence"], "preparationScriptSha256": file_sha(Path(__file__))}
    if isinstance(source, ReplayedSource):
        manifest["replay"] = source.replay_info
    manifest["peakRssKiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    disk_bytes = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    manifest["diskBytesBeforeManifest"] = disk_bytes
    if disk_bytes > 128 << 20:
        raise RuntimeError("acquisition disk budget reached")
    receipt = dump(args.output / "manifest.json", manifest)
    print(json.dumps({"completed": True, "manifest": receipt, "files": files, "splitCounts": dict(split_counts), "words": manifest["words"], "sourceStatistics": dict(counts), "errors": errors}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
