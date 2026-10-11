"""Author new fictional open-text facts, without reading models or evaluations.

Every world has four question paraphrases, two counterfactual assignments and
three supervised prefixes. All variants share a fold. A frozen local tokenizer
checks the complete dialogue; it is never retrained by this helper. Templates
are used to author data, not to provide replies during model inference.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from tokenizers import Tokenizer

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer


SPLITS = ("train", "validation", "test")
VERSION = "open-facts-v2"


def _template(template_id, *turns):
    return {"templateId": template_id, "turns": turns}


def _templates():
    """Manually authored Spanish; all user questions have a definite scope."""
    return {
        "name_alias": [
            _template("alias-preference",
                ("Me llamo {name}, vivo en {place}, pero prefiero que me digas {alias}. ¿Cómo quiero que me llames?", "Quieres que te llame {alias}."),
                ("¿Cuál es el nombre con el que me presenté?", "Te presentaste como {name}."),
                ("Si me saludas ahora, ¿qué nombre prefiero?", "Prefieres {alias}.")),
            _template("alias-nickname",
                ("Soy {name}, de {place}. Mi apodo es {alias}. ¿Cuál es mi apodo?", "Tu apodo es {alias}."),
                ("¿Y cuál es mi nombre?", "Tu nombre es {name}."),
                ("Quiero que uses el apodo. ¿Qué palabra usarás?", "Usaré {alias}.")),
            _template("alias-two-names",
                ("En {place} me conocen como {name}; en esta charla usa {alias}. ¿Cómo me llamarás aquí?", "Aquí te llamaré {alias}."),
                ("¿Cómo dije que me conocen en {place}?", "Dijiste que te conocen como {name}."),
                ("¿Qué nombre seguimos usando en esta charla?", "Seguimos usando {alias}.")),
            _template("alias-chat-label",
                ("Mi nombre es {name} y mi ciudad es {place}. Para conversar elijo el nombre {alias}. ¿Qué nombre elijo?", "Eliges {alias}."),
                ("Repite mi nombre original, sin cambiarlo.", "Tu nombre original es {name}."),
                ("Vuelve al nombre elegido para esta conversación. ¿Cuál es?", "El nombre elegido es {alias}.")),
        ],
        "name_change": [
            _template("name-correction",
                ("Soy {name}, de {place}. ¿Cómo me llamo?", "Te llamas {name}."),
                ("Corrijo mi nombre: es {new_name}. ¿Cuál es el nombre correcto?", "El nombre correcto es {new_name}."),
                ("Después de la corrección, ¿cómo debes llamarme?", "Debo llamarte {new_name}.")),
            _template("name-new-preference",
                ("En mi ficha de {place} puse {name}. ¿Qué nombre puse?", "Pusiste {name}."),
                ("Quiero reemplazarlo por {new_name}. ¿Qué nombre queda en la ficha?", "Queda {new_name}."),
                ("¿Qué nombre debo conservar ahora?", "Conserva {new_name}.")),
            _template("name-retraction",
                ("Me presenté como {name} y dije que vivo en {place}. ¿Qué nombre di?", "Diste el nombre {name}."),
                ("Ese nombre estaba mal escrito: usa {new_name}. ¿Cómo queda?", "Queda {new_name}."),
                ("¿Se mantiene {name}?", "No. Lo corregiste a {new_name}.")),
            _template("name-current-choice",
                ("Para la charla desde {place}, primero elijo el nombre {name}. ¿Cuál elegí?", "Elegiste {name}."),
                ("Cambio mi elección a {new_name}. Dime el nombre actual.", "El nombre actual es {new_name}."),
                ("No vuelvas al anterior. ¿Qué nombre uso?", "Usas {new_name}.")),
        ],
        "project_title": [
            _template("title-renaming",
                ("Soy {name}. Mi proyecto se llama {title}. ¿Cómo se llama?", "Se llama {title}."),
                ("Le cambio el título a {other_title}. ¿Cuál es el título nuevo?", "El título nuevo es {other_title}."),
                ("¿Qué título tiene el proyecto ahora?", "Ahora se llama {other_title}.")),
            _template("title-draft",
                ("El borrador de {name} tiene este título: {title}. ¿Qué título lleva?", "Lleva el título {title}."),
                ("El título definitivo será {other_title}. ¿Cuál será?", "Será {other_title}."),
                ("Repite el definitivo, no el del borrador.", "El título definitivo es {other_title}.")),
            _template("title-active-project",
                ("Tengo dos proyectos: {title} y {other_title}. Soy {name}. ¿Cómo se llama el primero?", "El primero se llama {title}."),
                ("Ahora trabajaré en el segundo. ¿En cuál trabajaré?", "Trabajarás en {other_title}."),
                ("Pregunto por el proyecto actual, ¿cuál es?", "El proyecto actual es {other_title}.")),
            _template("title-exact-text",
                ("Soy {name}. Escribí el título {title}. Repítelo sin cambiar las palabras.", "El título es {title}."),
                ("Lo sustituí por {other_title}. Repite el título que queda.", "El título que queda es {other_title}."),
                ("¿Sigue siendo {title}?", "No. Ahora es {other_title}.")),
        ],
        "attribute": [
            _template("attribute-label-code",
                ("Soy {name}. La etiqueta del paquete dice {value}. ¿Qué dice la etiqueta?", "La etiqueta dice {value}."),
                ("Su código es {code}. ¿Cuál es el código?", "El código es {code}."),
                ("Ahora quiero la etiqueta, no el código. ¿Qué dice?", "La etiqueta dice {value}.")),
            _template("attribute-code-edit",
                ("En el paquete de {name} anoté el código {code}. ¿Qué código anoté?", "Anotaste {code}."),
                ("Lo corrijo a {other_code}. ¿Cuál es el código correcto?", "El código correcto es {other_code}."),
                ("Copia el código actualizado.", "El código actualizado es {other_code}.")),
            _template("attribute-open-value",
                ("Soy {name}. En el campo nota puse {value}. ¿Qué texto puse?", "Pusiste {value}."),
                ("En el campo referencia puse {code}. ¿Cuál es la referencia?", "La referencia es {code}."),
                ("¿Qué texto conserva el campo nota?", "Conserva {value}.")),
            _template("attribute-change-text",
                ("Soy {name}. La descripción de mi proyecto {title} es {value}. ¿Cuál es la descripción?", "La descripción es {value}."),
                ("La descripción cambió a {other_value}. ¿Qué descripción queda?", "Queda {other_value}."),
                ("Dime sólo la descripción actual, en una frase corta.", "La descripción actual es {other_value}.")),
        ],
        "time_place": [
            _template("where-before-when",
                ("{name} tiene una cita en {place} a las {time}. ¿En qué lugar es?", "La cita es en {place}."),
                ("Ahora pregunto cuándo: ¿a qué hora es?", "Es a las {time}."),
                ("La hora cambia a las {other_time}; el lugar sigue igual. ¿Dónde es?", "Sigue siendo en {place}.")),
            _template("when-before-where",
                ("{name} dará una charla a las {time} en {place}. ¿A qué hora será?", "Será a las {time}."),
                ("Deja la hora aparte. ¿En qué lugar será?", "Será en {place}."),
                ("Se mueve a {other_place}, sin cambiar la hora. ¿A qué hora será?", "Será a las {time}.")),
            _template("time-correction",
                ("La cita de {name} está prevista en {place} a las {time}. ¿Qué hora está prevista?", "Está prevista a las {time}."),
                ("Corrijo la hora: será a las {other_time}. ¿Qué hora es ahora?", "Ahora es a las {other_time}."),
                ("¿Cambió también el lugar? No he anunciado otro.", "No, el lugar sigue siendo {place}.")),
            _template("place-correction",
                ("El encuentro de {name} será en {place} a las {time}. ¿Dónde será?", "Será en {place}."),
                ("El lugar cambia a {other_place}. ¿Dónde será ahora?", "Ahora será en {other_place}."),
                ("Conservamos la misma hora. ¿A qué hora quedamos?", "Quedaron a las {time}.")),
        ],
        "list_count": [
            _template("list-entries",
                ("Soy {name}. Mi lista es: {items_text}. ¿Cuántas entradas tiene?", "Tiene {count} entradas."),
                ("La reemplazo por esta lista: {other_items_text}. ¿Cuántas entradas hay ahora?", "Ahora hay {other_count} entradas."),
                ("¿Cuál es la primera entrada de la lista actual?", "La primera es {other_first}.")),
            _template("list-options",
                ("Estas son mis opciones, separadas por punto y coma: {items_text}. Soy {name}. ¿Cuántas opciones son?", "Son {count} opciones."),
                ("Cambio todas las opciones por: {other_items_text}. ¿Cuántas son ahora?", "Ahora son {other_count} opciones."),
                ("¿Cuál aparece primero en las opciones nuevas?", "Primero aparece {other_first}.")),
            _template("list-final-entry",
                ("En la lista de {name} figuran: {items_text}. ¿Cuántos elementos figuran?", "Figuran {count} elementos."),
                ("La lista actual pasa a ser: {other_items_text}. ¿Cuántos elementos figuran ahora?", "Ahora figuran {other_count} elementos."),
                ("Dime el último elemento de la lista actual.", "El último es {other_last}.")),
            _template("list-first-and-count",
                ("Soy {name} y anoté esta lista: {items_text}. ¿Cuál es la primera entrada?", "La primera es {first}."),
                ("¿Cuántas entradas tiene la lista que anoté?", "Tiene {count} entradas."),
                ("La sustituí por: {other_items_text}. ¿Cuántas entradas tiene la nueva?", "La nueva tiene {other_count} entradas.")),
        ],
        "missing_data": [
            _template("missing-time",
                ("Soy {name}. Tengo un proyecto llamado {title}, pero no he dicho cuándo empieza. ¿A qué hora empieza?", "Todavía no has dicho la hora de inicio."),
                ("Empieza a las {time}. ¿A qué hora empieza entonces?", "Empieza a las {time}."),
                ("Del proyecto {other_title} sólo dije el título. ¿Sabes dónde se hará?", "No has dicho dónde se hará {other_title}.")),
            _template("missing-code",
                ("La etiqueta del paquete de {name} dice {value}. No indiqué su código. ¿Cuál es su código?", "No has indicado el código del paquete."),
                ("El código es {code}. Repite el código.", "El código es {code}."),
                ("La otra caja tiene la etiqueta {other_value}, pero no di su código. ¿Qué código tiene?", "No has dicho el código de la otra caja.")),
            _template("missing-name",
                ("En {place} hay alguien con el apodo {alias}. No dije su nombre. ¿Cuál es su nombre?", "No has dicho su nombre."),
                ("Su nombre es {name}. ¿Cómo se llama?", "Se llama {name}."),
                ("Otra persona usa el apodo {new_name}, pero no di su nombre. ¿Cómo se llama?", "Todavía no has dicho el nombre de esa otra persona.")),
            _template("missing-place",
                ("La presentación de {name} será a las {time}. No he dicho el lugar. ¿Dónde será?", "Aún no has dicho el lugar de la presentación."),
                ("Será en {place}. ¿Dónde será entonces?", "Será en {place}."),
                ("Otra presentación será a las {other_time}, sin lugar anunciado. ¿Dónde será esa otra?", "No has anunciado el lugar de la otra presentación.")),
        ],
        "declarations": [
            _template("declaration-name",
                ("Me llamo {name}.", "Hola, {name}."),
                ("¿Cómo me llamo?", "Te llamas {name}."),
                ("Corrijo mi nombre: es {new_name}.", "Entendido, te llamas {new_name}.")),
            _template("declaration-alias",
                ("Llámame {alias}.", "De acuerdo, {alias}."),
                ("¿Cómo quiero que me llames?", "Quieres que te llame {alias}."),
                ("Cambia, ahora llámame {new_name}.", "De acuerdo, {new_name}.")),
            _template("declaration-project",
                ("Quiero trabajar en el proyecto {title}.", "Quieres trabajar en {title}."),
                ("¿En qué proyecto quiero trabajar?", "Quieres trabajar en {title}."),
                ("Cambié de idea: ahora quiero trabajar en {other_title}.", "Ahora quieres trabajar en {other_title}.")),
            _template("declaration-preference",
                ("Para la ficha de {name}, prefiero el texto {value}.", "Prefieres el texto {value}."),
                ("¿Qué texto prefiero para esa ficha?", "Prefieres {value}."),
                ("Lo cambio por {other_value}.", "Ahora prefieres {other_value}.")),
        ],
    }


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _name(index):
    starts = ("Al", "Ber", "Cal", "Del", "El", "Far", "Gar", "Hel", "Il", "Jar", "Kel", "Lor", "Mer", "Nel", "Or", "Par", "Quel", "Ril", "Sar", "Tal", "Ur", "Val", "Wen", "Zel")
    vowels = ("a", "e", "i", "o", "u", "ai", "ei", "ia", "io", "ua", "ue", "au")
    ends = ("na", "ro", "lia", "ven", "mir", "sel", "dan", "ris", "ton", "bel", "sor", "nel")
    return starts[index % len(starts)] + vowels[(index // len(starts)) % len(vowels)] + ends[(index // (len(starts) * len(vowels))) % len(ends)]


def _title(index):
    nouns = ("Ruta", "Mapa", "Cuaderno", "Taller", "Puente", "Archivo", "Sendero", "Jardín", "Ventana", "Círculo", "Faro", "Patio")
    # Titles are fictional names: the modifier isn't an adjective describing
    # the noun, so no gendered agreement is assumed for this proper-name slot.
    return nouns[index % len(nouns)] + " " + _name(index + 701)


def _value(index):
    first = ("tinta", "viento", "papel", "bruma", "cobre", "semilla", "piedra", "nube", "arena", "río", "hilo", "luz")
    second = ("norte", "sur", "este", "oeste", "centro", "borde", "orilla", "cumbre", "sendero", "puente", "jardín", "patio")
    return first[index % len(first)] + " " + second[(index // len(first)) % len(second)]


def _code(index):
    letters = "bcdfghjkmnpqrstvwxyz"
    return letters[index % len(letters)] + str(2 + index % 8) + "-" + letters[(index // len(letters)) % len(letters)] + str(11 + index % 83)


def _world(seed, split, episode_index):
    fold = SPLITS.index(split)
    shift = 17 * fold
    # Individual names/words recur across folds. Their associations and code/
    # title assignments change by fold. The pool is procedurally composed,
    # rather than a fixed list of the creator's names or facts.
    k = (episode_index + seed % 97) % 384
    n = 2 + episode_index % 4
    other_n = 2 + (episode_index + 1) % 4
    place = "Sala " + _name((k + shift + 63) % 384)
    other_place = "Sala " + _name((k + shift + 137) % 384)
    world = {
        "name": _name(k), "alias": _name((k + 31 + shift) % 384), "new_name": _name((k + 83 + shift) % 384),
        "place": place, "other_place": other_place,
        "time": f"{8 + (k + fold) % 12:02d}:{(k % 4) * 15:02d}",
        "other_time": f"{8 + (k + fold + 3) % 12:02d}:{((k + 1) % 4) * 15:02d}",
        "title": _title((k + shift) % 384), "other_title": _title((k + shift + 97) % 384),
        "value": _value((k + shift) % 144), "other_value": _value((k + shift + 67) % 144),
        "code": _code(k + shift), "other_code": _code(k + shift + 101),
        "items": [_title((k + shift + i * 29) % 384) for i in range(n)],
        "other_items": [_title((k + shift + 137 + i * 29) % 384) for i in range(other_n)],
    }
    return _derived(world)


def _derived(world):
    value = dict(world)
    value.update(count=len(value["items"]), other_count=len(value["other_items"]),
                 items_text="; ".join(value["items"]), other_items_text="; ".join(value["other_items"]),
                 first=value["items"][0], last=value["items"][-1],
                 other_first=value["other_items"][0], other_last=value["other_items"][-1])
    return value


def _swapped(world):
    changed = dict(world)
    for left, right in (("alias", "new_name"), ("place", "other_place"), ("time", "other_time"),
                        ("title", "other_title"), ("value", "other_value"), ("code", "other_code"), ("items", "other_items")):
        changed[left], changed[right] = world[right], world[left]
    return _derived(changed)


def _render(template, world):
    messages = []
    for user, reply in template["turns"]:
        messages.extend(({"role": "user", "content": user.format(**world)},
                         {"role": "assistant", "content": reply.format(**world)}))
    return messages


def generate(*, seed=20261012, episodes_per_skill=32):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in 0..4294967295")
    if isinstance(episodes_per_skill, bool) or not isinstance(episodes_per_skill, int) or not 10 <= episodes_per_skill <= 1000:
        raise ValueError("episodes_per_skill must be an integer in 10..1000")
    reserved = max(2, episodes_per_skill // 8)
    counts = {"train": episodes_per_skill - 2 * reserved, "validation": reserved, "test": reserved}
    counters, records, families = Counter(), [], {}
    banks = _templates()
    # Append the new declaration skill after the previous seven; V1 files are
    # never read or changed, and the original skills keep their world inputs.
    order = sorted(skill for skill in banks if skill != "declarations") + ["declarations"]
    for skill in order:
        templates = banks[skill]
        for split, count in counts.items():
            for _ in range(count):
                index = counters[split]
                counters[split] += 1
                world = _world(seed, split, index)
                if skill == "declarations":
                    # A standalone "Me llamo X" contains only one lexical
                    # fact. Reserve that literal prompt by fold; its names
                    # may still occur in longer contexts of other skills.
                    fold = SPLITS.index(split)
                    world["name"] = _name(((index + seed % 97) + fold * 131) % 384)
                branches = (world, _swapped(world))
                # Sorted branch hashes identify the same semantic episode if
                # its initial/counterfactual order is reversed in future code.
                identity = {"skill": skill, "assignments": sorted(_sha(_canonical(branch)) for branch in branches)}
                world_id = _sha(_canonical(identity))[:24]
                if world_id in families and families[world_id] != split:
                    raise ValueError("Semantic world occurs in multiple folds")
                families[world_id] = split
                group = f"{VERSION}-{world_id}"
                for template in templates:
                    for branch, values in enumerate(branches):
                        messages = _render(template, values)
                        for prefix in (1, 2, 3):
                            record = {"id": f"{group}-{template['templateId']}-b{branch}-p{prefix}",
                                      "groupId": group, "split": split,
                                      "source": {"dataset": "unidad-open-facts-v2", "kind": "authored-synthetic",
                                                 "skill": skill, "templateId": template["templateId"], "worldId": world_id,
                                                 "branch": branch, "prefixPairs": prefix, "fictional": True,
                                                 "privateConversationUsed": False, "modelResponsesUsed": False},
                                      "messages": messages[:2 * prefix]}
                            validate_record(record)
                            records.append(record)
    return records


def _exclusive_write(path, raw):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Existing outputs are immutable; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".open-facts-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("Concurrent writer created output; it was preserved") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(tokenizer_path, output_path, manifest_path, *, seed=20261012, episodes_per_skill=32, context_length=256):
    tokenizer_path, output, manifest = Path(tokenizer_path), Path(output_path), Path(manifest_path)
    if isinstance(context_length, bool) or not isinstance(context_length, int) or not 32 <= context_length <= 4096:
        raise ValueError("context_length must be an integer in 32..4096")
    if len({path.resolve() for path in (tokenizer_path, output, manifest)}) != 3:
        raise ValueError("Tokenizer and outputs must be distinct")
    if any(path.exists() or path.is_symlink() for path in (output, manifest)):
        raise ValueError("Existing outputs are immutable; choose new paths")
    raw_tokenizer = tokenizer_path.read_bytes()
    if len(raw_tokenizer) > 16_777_216:
        raise ValueError("Tokenizer exceeds local size limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(raw_tokenizer.decode("utf-8")), strict=True)
    candidates = generate(seed=seed, episodes_per_skill=episodes_per_skill)
    prompts = defaultdict(list)
    for record in candidates:
        prompts[_canonical(record["messages"][:-1])].append(record)
    excluded, reasons = set(), Counter()
    for same_prompt in prompts.values():
        if len({record["split"] for record in same_prompt}) > 1:
            excluded.update(record["groupId"] for record in same_prompt)
            reasons["cross_fold_exact_prompt_occurrences"] += len(same_prompt)
        if len({record["messages"][-1]["content"] for record in same_prompt}) > 1:
            raise ValueError("An identical full prompt has conflicting targets")
    selected, seen, lengths = [], set(), []
    for record in candidates:
        if record["groupId"] in excluded:
            reasons["records_in_cross_fold_prompt_families_omitted"] += 1
            continue
        key = _canonical(record["messages"])
        if key in seen:
            reasons["exact_duplicates_removed"] += 1
            continue
        tokens, _ = dialogue_tokens(record["messages"], tokenizer)
        if len(tokens) > context_length + 1:
            reasons["full_dialogue_exceeds_context"] += 1
            continue
        seen.add(key)
        selected.append(record)
        lengths.append(len(tokens))
    template_splits, skill_splits = defaultdict(Counter), defaultdict(Counter)
    for record in selected:
        template_splits[record["source"]["templateId"]][record["split"]] += 1
        skill_splits[record["source"]["skill"]][record["split"]] += 1
    template_ids = {template["templateId"] for bank in _templates().values() for template in bank}
    if set(template_splits) != template_ids or any(set(folds) != set(SPLITS) for folds in template_splits.values()):
        raise ValueError("Every authored template must remain represented in every fold")
    # Check prompt separation after context filtering and deduplication too.
    final_prompt_folds = {}
    for record in selected:
        key = _canonical(record["messages"][:-1])
        if key in final_prompt_folds and final_prompt_folds[key] != record["split"]:
            raise ValueError("Cross-fold prompt leakage survived authoring")
        final_prompt_folds[key] = record["split"]
    raw = b"".join(_canonical(record) + b"\n" for record in selected)
    sorted_lengths = sorted(lengths)
    lexical = {split: set() for split in SPLITS}
    possible_names = {_name(index) for index in range(1200)}
    for record in selected:
        for message in record["messages"]:
            words = set(re.findall(r"\b[A-Z][a-z]+\b", message["content"]))
            lexical[record["split"]].update(words & possible_names)
    shared_names = set.intersection(*(lexical[split] for split in SPLITS))
    report = {
        "schemaVersion": 1, "dataset": "unidad-open-facts-v2", "purpose": "auxiliary-training-curriculum",
        "sha256": _sha(raw), "records": len(selected), "seed": seed,
        "splits": dict(Counter(record["split"] for record in selected)),
        "groupSplits": {split: len({record["groupId"] for record in selected if record["split"] == split}) for split in SPLITS},
        "skills": {skill: dict(folds) for skill, folds in sorted(skill_splits.items())},
        "templates": {template: dict(folds) for template, folds in sorted(template_splits.items())},
        "prefixPairs": dict(Counter(str(record["source"]["prefixPairs"]) for record in selected)),
        "phoneticNameStringsBySplit": {split: len(values) for split, values in lexical.items()},
        "phoneticNameStringsSharedAcrossAllFolds": len(shared_names),
        "sharedNameSamples": sorted(shared_names)[:12],
        "fullDialogueTokens": {"min": min(lengths), "median": sorted_lengths[len(lengths)//2],
                              "p95": sorted_lengths[math.ceil(.95*len(lengths))-1], "max": max(lengths)},
        "tokenizer": {"sha256": _sha(raw_tokenizer), "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
        "authoring": {"episodesPerSkill": episodes_per_skill, "contextLength": context_length,
                      "fullDialogueTokenLimit": context_length+1, "dialoguesTruncated": False,
                      "templates": len(template_ids), "skills": len(_templates()), "counterfactualBranches": 2,
                      "prefixSupervision": "last assistant reply of each one/two/three exchange prefix",
                      "splitPolicy": "75/12.5/12.5 world quotas at defaults; all paraphrases/counterfactuals/prefixes share the episode fold",
                      "inputSelection": "Frozen tokenizer only; no corpora, weights, model responses or evaluation suite read",
                      "oldCorpusCollisionAudit": "Required separately before merge; existing folds must be preserved",
                      "fullPromptCrossFoldGuard": "quarantine every involved world before deduplication; reject conflicting prompt targets",
                      "modelScoresUsed": False, "privateConversationUsed": False, "evaluationRead": False},
        "reasons": dict(reasons), "excludedWorlds": len(excluded),
        "generatorSha256": _sha(Path(__file__).read_bytes()),
        "limitations": ["Procedural fiction and repeated templates cannot establish open-ended conversational competence.",
                        "New phonetic names, compound titles and codes broaden copying practice; they are not the owner's personal facts.",
                        "Individual lexical values can repeat across folds; complete episode assignments and exact prompts are separated.",
                        "Authored history differs from evaluation with the model's own previous replies.",
                        "No final or development evaluation result was used to author or choose these records."],
    }
    _exclusive_write(output, raw)
    load_corpus(output)
    _exclusive_write(manifest, _canonical(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--episodes-per-skill", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        report = build(args.tokenizer, args.output, args.manifest, seed=args.seed,
                       episodes_per_skill=args.episodes_per_skill, context_length=args.context_length)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
