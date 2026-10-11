"""Author fictional short dialogues; never route model inference through templates.

The frozen local tokenizer is used only to measure full dialogue length. No model,
scores, private conversation, old corpus or evaluation suite is read. Worlds,
counterfactual assignments, paraphrases and supervised prefixes share a fold.
Entity/value associations are reserved per fold while the lexical pools are shared.
Run with the conversation extras installed and PYTHONPATH=src.
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
import re
import tempfile

from tokenizers import Tokenizer

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer
if __package__:
    from .grounded_conversation_templates import TEMPLATES as CONVERSATION_TEMPLATES
else:
    from grounded_conversation_templates import TEMPLATES as CONVERSATION_TEMPLATES


VERSION = "grounded-v1"
SPLITS = ("train", "validation", "test")
NAMES = (
    "Hugo", "Nuria", "Diego", "Elena", "Ramón", "Clara", "Pablo", "Inés",
    "Daniel", "Alba", "Sergio", "Lucía", "Tomás", "Paula", "Raúl", "Irene",
    "Marcos", "Celia", "Andrés", "Julia", "Mario", "Elisa", "Bruno", "Noelia",
    "Rubén", "Alicia", "Jaime", "Vera", "Darío", "Olga", "Fabio", "Adela",
)
POOLS = {
    "item": ("cuaderno", "mapa", "libro", "vaso", "bolso", "cable", "lapicero", "reloj",
             "archivo", "sobre", "paquete", "plato", "boleto", "recibo", "estuche", "tablero",
             "dibujo", "teclado", "cartel", "lápiz", "borrador", "mantel", "abrigo", "documento"),
    "place": ("Quito", "Soria", "Lima", "Puebla", "Málaga", "Cusco", "Mérida", "Zamora",
              "Tacna", "Burgos", "Loja", "Zaragoza", "Osuna", "Medellín", "Cali", "Bogotá",
              "Granada", "Segovia", "León", "Durango", "Tepic", "Cartagena", "Toledo", "Huelva"),
    "day": ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"),
    "count": tuple(range(2, 26)),
    "color": ("rojo", "verde", "azul", "amarillo", "negro", "blanco", "gris", "morado", "naranja", "rosa"),
    "activity": ("leer", "pasear", "cocinar", "dibujar", "escribir", "correr", "descansar", "estudiar",
                 "nadar", "caminar", "bailar", "cantar", "pintar", "ordenar", "practicar", "conversar"),
}
FEELINGS = ("cansado", "tranquilo", "preocupado", "animado", "confundido", "contento", "inquieto", "desanimado")
REASONS = ("se atrasó el pedido", "no entiendo el ejercicio", "cambió el horario", "terminé una tarea",
           "me cuesta empezar", "llegó una noticia", "perdí una nota", "tengo poco tiempo")
REASON_FEELINGS = {
    "se atrasó el pedido": ("preocupado", "inquieto", "desanimado"),
    "no entiendo el ejercicio": ("confundido", "preocupado"),
    "cambió el horario": ("tranquilo", "preocupado", "inquieto"),
    "terminé una tarea": ("contento", "animado", "cansado"),
    "me cuesta empezar": ("desanimado", "confundido"),
    "llegó una noticia": ("preocupado", "contento", "animado"),
    "perdí una nota": ("preocupado", "desanimado", "inquieto"),
    "tengo poco tiempo": ("preocupado", "cansado", "inquieto"),
}


def _template(template_id, *turns):
    return {"templateId": template_id, "turns": list(turns)}


CONVERSATION_TEMPLATES = {
    **CONVERSATION_TEMPLATES,
    "model_identity": [
        _template("identity-names",
            ("Hola, soy {name} y vivo en {place}. ¿Cómo te llamas tú?", "Me llamo Unidad."),
            ("¿Cuál es mi nombre?", "Te llamas {name}."),
            ("¿Quién vive en {place}, tú o yo?", "Tú me dijiste que vives en {place}.")),
        _template("identity-introduction",
            ("Soy {name}; voy a {activity} el {day}. ¿Tú quién eres?", "Soy Unidad, un modelo de lenguaje."),
            ("¿Yo soy Unidad?", "No. Te presentaste como {name}; Unidad soy yo."),
            ("¿Qué dije que voy a hacer?", "Dijiste que vas a {activity}.")),
        _template("identity-recipient",
            ("Me llamo {name} y llevo un {item}. Hola, ¿cómo te llamas?", "Hola. Soy Unidad."),
            ("Quiero saludar a {friend}. ¿A quién quiero saludar?", "Quieres saludar a {friend}."),
            ("¿{friend} es tu nombre?", "No, mi nombre es Unidad.")),
        _template("identity-own-name",
            ("Soy {name}, de {place}. ¿Qué nombre uso para hablar contigo?", "Puedes llamarme Unidad."),
            ("¿Y cómo me llamo yo?", "Te llamas {name}."),
            ("Saluda a {friend}. ¿Qué saludo le escribirías?", "Hola, {friend}. ¿Cómo estás?")),
    ],
}


# These targets were written as training examples, without consulting the
# development benchmark. Facts are stated in each user's fictional dialogue.
FACT_TEMPLATES = {
    "explicit_fact": [
        _template("fact-journey",
            ("Soy {name}. Iré a {place} el {day}. ¿Qué día salgo?", "Sales el {day}."),
            ("¿A qué ciudad voy?", "Vas a {place}."),
            ("Llevo {count} libros. ¿Cuántos libros llevo?", "Llevas {count} libros.")),
        _template("fact-package",
            ("El {item} de {name} es {color}. ¿De qué color es?", "El {item} es {color}."),
            ("Lo enviará a {place}. ¿Cuál es el destino?", "El destino es {place}."),
            ("Sale el {day}. ¿Cuándo sale?", "Sale el {day}.")),
        _template("fact-meeting",
            ("Me llamo {name} y quedé con {friend} en {place}. ¿Con quién quedé?", "Quedaste con {friend}."),
            ("¿En qué ciudad quedamos?", "Quedaron en {place}."),
            ("Será el {day}. ¿Qué día será?", "Será el {day}.")),
        _template("fact-inventory",
            ("Soy {name}; tengo {count} notas y un {item}. ¿Cuántas notas tengo?", "Tienes {count} notas."),
            ("El {item} es {color}. ¿Cuál es su color?", "Su color es {color}."),
            ("Además de las notas, ¿qué objeto tengo?", "Tienes un {item}.")),
    ],
    "correction": [
        _template("edit-date",
            ("Soy {name}. La entrega en {place} es el {day}. ¿Cuándo es?", "La entrega es el {day}."),
            ("Me corrijo: será el {other_day}, no el {day}. ¿Qué día queda?", "Queda el {other_day}."),
            ("¿Sigue siendo el {day}?", "No, ahora es el {other_day}.")),
        _template("edit-color",
            ("El {item} de {name} es {color}. ¿Cuál es su color?", "El {item} es {color}."),
            ("Dije mal el color: es {other_color}. ¿Qué color es ahora?", "Es {other_color}."),
            ("¿Qué color debo anotar?", "Anota {other_color}.")),
        _template("edit-destination",
            ("Soy {name} y mi destino es {place}. ¿A dónde voy?", "Vas a {place}."),
            ("Cambio de destino: voy a {other_place}. ¿Cuál es el destino actual?", "El destino actual es {other_place}."),
            ("¿Voy a {place} o a {other_place}?", "Vas a {other_place}.")),
        _template("edit-count",
            ("{name} contó {count} sobres. ¿Cuántos contó?", "Contó {count} sobres."),
            ("La cifra correcta es {other_count}. ¿Cuántos hay según la corrección?", "Hay {other_count} sobres."),
            ("¿Cuál es la cifra que debemos conservar?", "La cifra correcta es {other_count}.")),
    ],
    "selector": [
        _template("select-city",
            ("{name} vive en {place}; {friend} vive en {other_place}. ¿Dónde vive {name}?", "{name} vive en {place}."),
            ("Ahora pregunto por {friend}. ¿Dónde vive?", "{friend} vive en {other_place}."),
            ("¿Quién vive en {place}?", "En {place} vive {name}.")),
        _template("select-item",
            ("El {item} es de {name} y el {other_item} es de {friend}. ¿De quién es el {item}?", "El {item} es de {name}."),
            ("¿Y de quién es el {other_item}?", "El {other_item} es de {friend}."),
            ("¿Qué objeto tiene {name}?", "{name} tiene el {item}.")),
        _template("select-count",
            ("{name} tiene {count} libros y {friend} tiene {other_count}. ¿Cuántos tiene {friend}?", "{friend} tiene {other_count} libros."),
            ("¿Cuántos tiene {name}?", "{name} tiene {count} libros."),
            ("¿Quién tiene {other_count} libros?", "{friend} tiene {other_count} libros.")),
        _template("select-day",
            ("{name} llega el {day}; {friend} llega el {other_day}. ¿Cuándo llega {friend}?", "{friend} llega el {other_day}."),
            ("¿Cuándo llega {name}?", "{name} llega el {day}."),
            ("Ahora dime quién llega el {other_day}.", "El {other_day} llega {friend}.")),
    ],
    "clarification": [
        _template("missing-destination",
            ("Soy {name} y enviaré un {item}. ¿Puedes decirme la ciudad de destino?", "Todavía no dijiste la ciudad de destino. ¿Cuál es?"),
            ("El destino es {place}. ¿Qué destino te acabo de dar?", "El destino es {place}."),
            ("También enviaré un {other_item}, pero no dije adónde. ¿Cuál es su destino?", "No has dicho el destino del {other_item}.")),
        _template("missing-day",
            ("Soy {name} y tengo una reunión. ¿Qué día es?", "Aún no me has dicho el día de la reunión."),
            ("La reunión será el {day}. ¿Cuándo es entonces?", "La reunión es el {day}."),
            ("La reunión de {friend} es otra; no dije su fecha. ¿Qué día es?", "No has indicado el día de la reunión de {friend}.")),
        _template("ambiguous-owner",
            ("{name} tiene un {item}; {friend} tiene un {other_item}. ¿Qué objeto tiene esa persona?", "¿Te refieres a {name} o a {friend}?"),
            ("Me refiero a {friend}. ¿Qué objeto tiene?", "{friend} tiene un {other_item}."),
            ("¿Y qué objeto tiene {name}?", "{name} tiene un {item}.")),
        _template("missing-quantity",
            ("{name} compró libros en {place}. ¿Cuántos compró?", "No has dicho cuántos libros compró."),
            ("Compró {count}. ¿Cuántos libros son?", "Son {count} libros."),
            ("{friend} también compró libros, pero no sabemos cuántos. ¿Qué cantidad compró?", "Todavía no sabemos cuántos compró {friend}.")),
    ],
    "grounded_connection": [
        _template("connect-journey",
            ("Soy {name}. Debo llevar el {item} a {place} el {day}. ¿Qué datos van juntos?", "El {item} debe ir a {place} el {day}."),
            ("¿Sabes por qué elegí esa ciudad?", "No has explicado por qué elegiste {place}."),
            ("La elegí porque {reason}. ¿Qué razón di?", "Dijiste que {reason}.")),
        _template("connect-feeling",
            ("Soy {name}. Estoy {feeling} porque {reason}. ¿Qué relacionas?", "Relaciono cómo te sientes con que {reason}."),
            ("¿Sabes algo más de mi día?", "Sólo sé lo que has contado en esta charla."),
            ("Quiero {activity} un rato. ¿Qué quiero hacer?", "Quieres {activity} un rato.")),
        _template("connect-delivery",
            ("{name} enviará un {item} {color} a {place}. ¿Puedes unir esos datos?", "{name} enviará el {item} {color} a {place}."),
            ("Sale el {day}. ¿Cuándo sale el envío?", "El envío sale el {day}."),
            ("¿Ya sabemos si llegó?", "No, no has dicho que haya llegado.")),
        _template("connect-plan",
            ("Soy {name}; voy a {activity} en {place} el {day}. ¿Cuál es mi plan?", "Tu plan es {activity} en {place} el {day}."),
            ("¿Puedes asegurar que lo cumpliré?", "No puedo asegurarlo; sólo sé que ése es tu plan."),
            ("Cambio la actividad: voy a {other_activity}. ¿Qué haré según mi plan?", "Según tu plan, vas a {other_activity}.")),
    ],
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _pool(cohort, field, split):
    """Reserve associations while rotating lexical values across name cohorts.

    Each name belongs to one of eight cohorts. Within a cohort, every name/value
    association has one fold; different cohorts rotate the pool, so a value is
    not globally reserved for validation or test. At least two values per fold
    let us swap assignments within a world without moving its family.
    """
    values = list(POOLS[field])
    shift = (cohort * math.ceil(len(values) / 8)) % len(values)
    values = values[shift:] + values[:shift]
    reserved = max(2, len(values) // 6)
    if split == "train":
        return values[:-2 * reserved]
    if split == "validation":
        return values[-2 * reserved:-reserved]
    return values[-reserved:]


def _world(rng, split, episode_index):
    # Cycle cohorts and speaker names so small held-out folds contain the same
    # complete lexical pools. Reserve associations, rather than unseen words.
    cohort = episode_index % 8
    names = NAMES[4 * cohort:4 * cohort + 4]
    position = (episode_index // 8) % 4
    friend_offset = 1 + (episode_index // 32) % 3
    name, friend = names[position], names[(position + friend_offset) % 4]
    reason = rng.choice(REASONS)
    feeling = rng.choice(REASON_FEELINGS[reason])
    if NAMES.index(name) % 2 and feeling.endswith("o"):
        feeling = feeling[:-1] + "a"
    fields = {"name": name, "friend": friend, "feeling": feeling, "reason": reason}
    for field_index, field in enumerate(POOLS):
        values = _pool(cohort, field, split)
        position = (episode_index // 8 + field_index * 3) % len(values)
        offset = 1 + (episode_index // (8 * len(values))) % (len(values) - 1)
        first, second = values[position], values[(position + offset) % len(values)]
        fields[field], fields["other_" + field] = first, second
    return fields, cohort


def _world_key(skill, world):
    # Sorting both entities and value pairs collapses counterfactual swaps.
    semantic = {"skill": skill, "entities": sorted((world["name"], world["friend"])),
                "values": {field: sorted((world[field], world["other_" + field])) for field in POOLS},
                "feeling": world["feeling"], "reason": world["reason"]}
    return _digest(_canonical(semantic))[:24]


def _swapped(world):
    changed = dict(world)
    for field in POOLS:
        changed[field], changed["other_" + field] = world["other_" + field], world[field]
    return changed


def _render(template, world):
    messages = []
    for index, (user, assistant) in enumerate(template["turns"]):
        if index == 0:
            # Ground even a greeting/listening episode in a fictional speaker
            # and city; otherwise every fold could contain the same full
            # three-turn generic dialogue. Keep the actual question last.
            if "{name}" not in user and "{place}" not in user:
                user = "Soy {name}, de {place}. " + user
            elif "{name}" not in user:
                user = "Soy {name}. " + user
            elif "{place}" not in user:
                user = "Estoy en {place}. " + user
        messages.extend(({"role": "user", "content": user.format(**world)},
                         {"role": "assistant", "content": assistant.format(**world)}))
    return messages


def _quotas(episodes):
    reserved = max(2, episodes // 10)
    return {"train": episodes - 2 * reserved, "validation": reserved, "test": reserved}


def generate(*, seed=20261011, conversation_episodes=40, fact_episodes=60):
    """Return canonical records before tokenizer-only length checks.

    Three prefixes expose every assistant reply to the trainer's final-reply
    objective. Every template and both swapped worlds stay within one family.
    """
    for name, value in (("seed", seed), ("conversation_episodes", conversation_episodes), ("fact_episodes", fact_episodes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 10):
            raise ValueError(f"{name} must be an integer >= {0 if name == 'seed' else 10}")
    if conversation_episodes + fact_episodes > 2000:
        raise ValueError("episode budget exceeds the authoring limit")
    all_templates = {**CONVERSATION_TEMPLATES, **FACT_TEMPLATES}
    rng = random.Random(seed)
    records, seen_worlds = [], set()
    world_counters = Counter()
    for skill, templates in sorted(all_templates.items()):
        episodes = conversation_episodes if skill in CONVERSATION_TEMPLATES else fact_episodes
        for split, count in _quotas(episodes).items():
            for _ in range(count):
                while True:
                    world, cohort = _world(rng, split, world_counters[split])
                    world_counters[split] += 1
                    world_id = _world_key(skill, world)
                    if world_id not in seen_worlds:
                        seen_worlds.add(world_id)
                        break
                group_id = f"{VERSION}-{world_id}"
                for template in templates:
                    for branch, values in enumerate((world, _swapped(world))):
                        messages = _render(template, values)
                        for pairs in range(1, 4):
                            record = {
                                "id": f"{group_id}-{template['templateId']}-b{branch}-p{pairs}",
                                "groupId": group_id, "split": split,
                                "source": {"dataset": "unidad-grounded-conversations-v1", "kind": "authored-synthetic",
                                           "skill": skill, "templateId": template["templateId"], "worldId": world_id,
                                           "counterfactualBranch": branch, "supervisedPrefixPairs": pairs,
                                           "nameCohort": cohort, "fictional": True,
                                           "privateConversationUsed": False, "modelResponsesUsed": False},
                                "messages": messages[:2 * pairs],
                            }
                            validate_record(record)
                            records.append(record)
    return records


def _distribution(values):
    ordered = sorted(values)
    return {"min": ordered[0], "median": ordered[len(ordered) // 2],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "max": ordered[-1]}


def _exclusive_write(path, raw):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Existing outputs are immutable; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".grounded-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("Another writer created the output; it was preserved") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(tokenizer_path, output_path, manifest_path, *, seed=20261011,
          conversation_episodes=40, fact_episodes=60, context_length=256):
    if isinstance(context_length, bool) or not isinstance(context_length, int) or not 32 <= context_length <= 4096:
        raise ValueError("context_length must be an integer between 32 and 4096")
    output, manifest, tokenizer_path = Path(output_path), Path(manifest_path), Path(tokenizer_path)
    if output.resolve() == manifest.resolve() or any(path.resolve() == tokenizer_path.resolve() for path in (output, manifest)):
        raise ValueError("Tokenizer, output and manifest must have distinct paths")
    if any(path.exists() or path.is_symlink() for path in (output, manifest)):
        raise ValueError("Existing outputs are immutable; choose new paths")
    tokenizer_raw = tokenizer_path.read_bytes()
    if len(tokenizer_raw) > 16_777_216:
        raise ValueError("Tokenizer exceeds the local byte limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(tokenizer_raw.decode("utf-8")), strict=True)
    candidates = generate(seed=seed, conversation_episodes=conversation_episodes, fact_episodes=fact_episodes)
    by_messages = defaultdict(list)
    for record in candidates:
        by_messages[_canonical(record["messages"])].append(record)
    selected, reasons, lengths = [], Counter(), []
    lengths_by_split = {split: [] for split in SPLITS}
    for same_text in by_messages.values():
        # A generic first prefix can recur in unrelated worlds. Never copy it
        # into several folds; omit all such cross-fold prefixes rather than
        # moving any world or choosing a favourable validation assignment.
        if len({record["split"] for record in same_text}) > 1:
            reasons["cross_fold_generic_prefix_records_omitted"] += len(same_text)
            continue
        record = same_text[0]
        reasons["exact_duplicates_within_fold_removed"] += len(same_text) - 1
        tokens, _ = dialogue_tokens(record["messages"], tokenizer)
        if len(tokens) > context_length + 1:
            reasons["full_dialogue_exceeds_context"] += 1
            continue
        selected.append(record)
        lengths.append(len(tokens))
        lengths_by_split[record["split"]].append(len(tokens))
    splits = Counter(record["split"] for record in selected)
    template_splits = defaultdict(Counter)
    skill_splits = defaultdict(Counter)
    lexical_splits = {split: {field: set() for field in ("name", *POOLS)} for split in SPLITS}
    for record in selected:
        source = record["source"]
        skill_splits[source["skill"]][record["split"]] += 1
        template_splits[source["templateId"]][record["split"]] += 1
        joined = " ".join(message["content"] for message in record["messages"])
        for field, values in {"name": NAMES, **POOLS}.items():
            # Coverage is descriptive, never used to select favourable records.
            for value in values:
                if re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", joined):
                    lexical_splits[record["split"]][field].add(str(value))
    if any(not splits[split] for split in SPLITS) or any(set(counts) != set(SPLITS) for counts in template_splits.values()):
        raise ValueError("Every template must retain examples in every fold; review the token budget")
    all_templates = {**CONVERSATION_TEMPLATES, **FACT_TEMPLATES}
    if set(template_splits) != {template["templateId"] for bank in all_templates.values() for template in bank}:
        raise ValueError("A complete template was excluded by the token budget")
    raw = b"".join(_canonical(record) + b"\n" for record in selected)
    group_splits = {split: len({record["groupId"] for record in selected if record["split"] == split}) for split in SPLITS}
    conversational_count = sum(count for skill, folds in skill_splits.items() if skill in CONVERSATION_TEMPLATES for count in folds.values())
    report = {
        "schemaVersion": 1, "dataset": "unidad-grounded-conversations-v1", "purpose": "training-curriculum",
        "sha256": _digest(raw), "seed": seed, "records": len(selected), "groups": sum(group_splits.values()),
        "splits": dict(splits), "groupSplits": group_splits,
        "skills": {skill: dict(folds) for skill, folds in sorted(skill_splits.items())},
        "templates": {template: dict(folds) for template, folds in sorted(template_splits.items())},
        "conversationRecords": conversational_count, "conversationFraction": conversational_count / len(selected),
        "prefixPairs": dict(Counter(str(record["source"]["supervisedPrefixPairs"]) for record in selected)),
        "fullDialogueTokens": _distribution(lengths),
        "fullDialogueTokensBySplit": {split: _distribution(values) for split, values in lengths_by_split.items()},
        "lexicalPoolCoverage": {split: {field: len(values) for field, values in pools.items()} for split, pools in lexical_splits.items()},
        "tokenizer": {"name": tokenizer_path.name, "sha256": _digest(tokenizer_raw), "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
        "authoring": {"conversationEpisodesPerSkill": conversation_episodes, "factEpisodesPerSkill": fact_episodes,
                      "contextLength": context_length, "fullDialogueTokenLimit": context_length + 1,
                      "dialoguesTruncated": False, "modelScoresUsed": False, "modelResponsesUsed": False,
                      "evaluationSuiteRead": False, "existingCorpusRead": False, "privateConversationUsed": False,
                      "counterfactualValueAssignments": 2, "prefixSupervision": "last assistant reply of each 1/2/3 exchange prefix",
                      "splitPolicy": "80/10/10 world quotas; paraphrases/counterfactuals/prefixes share world; name-value associations reserved by rotating lexical pools",
                      "oldCorpusCollisionAudit": "Required separately before merging; no old fold is changed"},
        "reasons": dict(reasons),
        "limitations": ["Procedural fiction and 44 reviewed template patterns do not represent unconstrained conversation.",
                        "The shared templates intentionally test held-out entity/fact associations, not unseen conversational skills.",
                        "No evaluation result or claim of model competence is produced by authoring data.",
                        "Training prefixes contain authored history; free-generation evaluation must also use the model's actual prior replies."],
        "generatorSha256": _digest(Path(__file__).read_bytes()),
        "conversationTemplatesSha256": _digest(Path(__file__).with_name("grounded_conversation_templates.py").read_bytes()),
    }
    _exclusive_write(output, raw)
    try:
        load_corpus(output)
        _exclusive_write(manifest, _canonical(report) + b"\n")
    except Exception:
        # Leave the immutable dataset visible if publication of its manifest
        # fails; no existing file is ever removed or overwritten.
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--conversation-episodes", type=int, default=40)
    parser.add_argument("--fact-episodes", type=int, default=60)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        report = build(args.tokenizer, args.output, args.manifest, seed=args.seed,
                       conversation_episodes=args.conversation_episodes, fact_episodes=args.fact_episodes,
                       context_length=args.context_length)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
