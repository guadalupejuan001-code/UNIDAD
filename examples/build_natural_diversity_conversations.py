"""Frozen fictional Spanish authoring bank; never imported by model inference.

Whole worlds retain every flow and last-reply training prefix in one fold. Shared
lexical and paraphrase banks have no fold-specific wording. Every opening states
an explicit fictional fact, so identity questions occur later in natural chats.
Only an optional existing authored corpus is read, for prompt hashes and folds;
no weights, personal history or development/final evaluation are accessed.
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
import tempfile
import unicodedata

from tokenizers import Tokenizer
from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer

DATASET = "unidad-natural-diversity-v2"
SPLITS = ("train", "validation", "test")
NAMES = ("Alba", "Ana", "Aroa", "Bruno", "Berta", "Celia", "Ciro", "Dalia", "Diego", "Elena", "Eva", "Fabio", "Fiona", "Gael", "Gina", "Hugo", "Inés", "Ivo", "Joel", "Julia", "Lara", "Leo", "Mara", "Mario", "Nora", "Olga", "Omar", "Pablo", "Rita", "Sara", "Teo", "Vera")
SURNAMES = ("Ríos", "Soto", "Luna", "Vega", "Sierra", "Costa", "Mora", "Gil", "Lago", "León", "Cruz", "Rey", "Flores", "Pardo", "Ruiz", "Sanz", "Vidal", "Roca", "Paz", "Olmo", "Prado", "Valle", "Soler", "Marín")
CITY_BASES = ("Puerto", "Monte", "Campo", "Valle", "Pueblo", "Río", "Cerro", "Lago", "Bosque", "Prado", "Camino", "Parque", "Puente", "Barrio", "Jardín", "Roble")
CITY_ENDINGS = ("Claro", "Alto", "Verde", "Nuevo", "Sereno", "Azul", "Abierto", "Dorado", "Blanco", "Grande", "Pequeño", "Largo", "Corto", "Viejo", "Rojo", "Tranquilo", "Redondo", "Central", "Seco", "Fresco", "Cercano", "Limpio", "Bonito", "Soleado")
TOPIC_BASES = ("Mapa", "Libro", "Plan", "Taller", "Cuaderno", "Camino", "Puente", "Faro", "Jardín", "Paseo", "Curso", "Dibujo", "Relato", "Viaje", "Juego", "Huerto")
TOPIC_ENDINGS = ("Azul", "Norte", "Sur", "Claro", "Nuevo", "Breve", "Abierto", "Sereno", "Verde", "Simple", "Largo", "Corto", "Amarillo", "Rojo", "Local", "Pequeño", "Lento", "Rápido", "Compartido", "Inicial", "Final", "Libre", "Tranquilo", "Diario")
ALIAS_BASES = ("Sol", "Mar", "Faro", "Cielo", "Lago", "Roble", "Río", "Viento", "Pino", "Brote", "Rayo", "Eco", "Trébol", "Sendero", "Nido", "Gato")
ALIAS_ENDINGS = ("Azul", "Claro", "Suave", "Breve", "Libre", "Verde", "Rojo", "Sereno", "Nuevo", "Alegre", "Tranquilo", "Dorado", "Lento", "Veloz", "Pequeño", "Abierto")
ITEMS = (("vaso", "el"), ("camisa", "la"), ("cuaderno", "el"), ("plato", "el"), ("bolso", "el"), ("taza", "la"), ("silla", "la"), ("mesa", "la"), ("botella", "la"), ("libro", "el"))
PUBLIC_CITIES = ("Málaga", "Córdoba", "Lima", "Quito", "Salamanca", "Granada", "Burgos", "León", "Toledo", "Teruel", "Soria", "Segovia", "Oviedo", "Cádiz", "Huelva", "Almería", "Murcia", "Logroño", "Bilbao", "Vitoria", "Pamplona", "Zamora", "Cuenca", "Ávila", "Lugo", "Ourense", "Sevilla", "Zaragoza", "Santander", "Palencia", "Huesca", "Jaén")
COLORS = ("rojo", "azul", "verde", "gris", "negro", "blanco", "amarillo", "morado")

# Eight authored formulations per semantic event; every fold uses this same bank.
USER_TEXT = {
 "introduce_name": ("Hola, me llamo {name}.", "Buenas, soy {name}.", "Soy {name}; quería charlar.", "Me llamo {name}. ¿Charlamos?", "Puedes conocerme por mi nombre, {name}.", "Hola. Mi nombre es {name}.", "{name} es mi nombre.", "Para presentarme: soy {name}."),
 "correct_name": ("Me equivoqué: me llamo {new_name}.", "Corrijo mi nombre: {new_name}.", "Mi nombre correcto es {new_name}.", "En realidad soy {new_name}.", "Antes dije mal mi nombre; es {new_name}.", "Cambia mi nombre a {new_name}.", "No me llamo así, sino {new_name}.", "Quédate con mi nombre correcto: {new_name}."),
 "introduce_alias": ("Hola, llámame {alias}.", "Prefiero que me llames {alias}.", "Para esta charla, mi apodo es {alias}.", "Puedes decirme {alias}, como apodo.", "Aquí quiero usar el apodo {alias}.", "Me gustaría que me dijeras {alias}.", "Me gusta que me llamen {alias}.", "Usa {alias} para dirigirte a mí."),
 "ask_real_name": ("¿Cómo me llamo?", "¿Cuál es mi nombre real?", "Dime mi nombre.", "¿Recuerdas mi nombre?", "¿Qué nombre mío te dije?", "¿Con qué nombre me presenté?", "Quiero saber qué nombre mío recuerdas.", "¿Puedes repetir mi nombre real?"),
 "ask_alias": ("¿Cómo quiero que me llames?", "¿Qué apodo elegí?", "Dime el apodo que uso aquí.", "¿Qué nombre prefiero para esta charla?", "¿Recuerdas cómo quiero que me digas?", "Repite mi apodo, por favor.", "¿Cómo te pedí que me llamaras?", "¿Con qué apodo me tratarás?"),
 "ask_own_name": ("¿Cómo te llamas?", "¿Quién eres?", "Dime tu nombre.", "¿Cuál es tu nombre?", "¿Cómo puedo llamarte?", "Preséntate con tu nombre.", "¿Qué nombre usas tú?", "¿Con quién estoy hablando?"),
 "introduce_city": ("Hola, estoy en {city}.", "Hoy estoy en {city}.", "Ahora me encuentro en {city}.", "Te escribo desde {city}.", "Estoy pasando el día en {city}.", "En este momento estoy en {city}.", "Por ahora estoy en {city}.", "Estoy aquí, en {city}."),
 "correct_city": ("Ahora estoy en {new_city}.", "Me moví a {new_city}.", "Corrijo el lugar: estoy en {new_city}.", "Ya no estoy allí; estoy en {new_city}.", "Cambié de lugar y estoy en {new_city}.", "Mi lugar actual es {new_city}.", "Ahora me encuentro en {new_city}.", "Quédate con esto: estoy en {new_city}."),
 "ask_city": ("¿Dónde estoy ahora?", "¿En qué lugar estoy?", "Dime dónde me encuentro.", "¿Cuál es mi lugar actual?", "¿Desde dónde te escribo ahora?", "¿Dónde dije que estaba?", "¿Recuerdas mi ubicación actual?", "Repite el lugar donde estoy."),
 "introduce_topic": ("Hola, quiero hablar de {topic}.", "Tengo en mente el proyecto {topic}.", "Me apetece hablar de {topic}.", "Hoy traigo el tema {topic}.", "Hablemos del proyecto {topic}.", "Quería contarte algo sobre {topic}.", "El tema que me interesa es {topic}.", "Propongo conversar sobre {topic}."),
 "correct_topic": ("Ahora hablemos de {new_topic}.", "Cambio de tema: {new_topic}.", "Prefiero hablar de {new_topic}.", "Dejemos eso; me interesa {new_topic}.", "El tema actual será {new_topic}.", "Mejor conversemos sobre {new_topic}.", "Quiero pasar al proyecto {new_topic}.", "Cambié de idea: hablemos de {new_topic}."),
 "ask_topic": ("¿De qué hablamos ahora?", "¿Cuál es el tema actual?", "Dime qué proyecto elegí.", "¿Qué tema tengo en mente ahora?", "¿Recuerdas de qué quería hablar?", "Repite el tema que escogí.", "¿Sobre qué estamos conversando?", "¿Qué proyecto estamos tratando?"),
 "set_color": ("Mi {item} es {color}.", "Tengo {article} {item} de color {color}.", "{Article} {item} que uso es {color}.", "Por cierto, mi {item} es {color}.", "Te cuento que {article} {item} es {color}.", "El color de mi {item} es {color}.", "Aquí tengo {article} {item} {color}.", "{Article} {item} es {color}, para que lo sepas."),
 "set_other_color": ("Mi {other_item} es {other_color}.", "Tengo {other_article} {other_item} de color {other_color}.", "{Other_article} {other_item} que uso es {other_color}.", "Por cierto, mi {other_item} es {other_color}.", "Te cuento que {other_article} {other_item} es {other_color}.", "El color de mi {other_item} es {other_color}.", "Aquí tengo {other_article} {other_item} {other_color}.", "{Other_article} {other_item} es {other_color}, para que lo sepas."),
 "correct_color": ("Corrijo: mi {item} es {new_color}.", "Me equivoqué: {article} {item} es {new_color}.", "El color correcto de mi {item} es {new_color}.", "Mi {item} no es así; es {new_color}.", "Quédate con este color: mi {item} es {new_color}.", "En realidad, {article} {item} es {new_color}.", "Cambia el color de mi {item} a {new_color}.", "Antes dije mal el color; mi {item} es {new_color}."),
 "ask_color": ("¿De qué color es mi {item}?", "¿Qué color tiene {article} {item}?", "Dime el color de mi {item}.", "¿Cómo es el color de mi {item}?", "¿Recuerdas el color del objeto, mi {item}?", "Repite el color de mi {item}.", "¿Qué color dije para {article} {item}?", "¿Cuál es el color actual de mi {item}?"),
 "ask_other_color": ("¿De qué color es mi {other_item}?", "¿Qué color tiene {other_article} {other_item}?", "Dime el color de mi {other_item}.", "¿Cómo es el color de mi {other_item}?", "¿Recuerdas el color del objeto, mi {other_item}?", "Repite el color de mi {other_item}.", "¿Qué color dije para {other_article} {other_item}?", "¿Cuál es el color actual de mi {other_item}?"),
 "listen": ("Sólo quiero que me escuches.", "Por ahora, nada de consejos; escúchame.", "Prefiero hablar sin preparar un plan.", "Necesito contarlo, sin buscar soluciones.", "Quiero conversar sin recibir propuestas.", "Me basta con que me escuches un rato.", "Dejemos los pasos; sólo quiero hablar.", "Ahora quiero que escuches, sin consejos."),
 "propose": ("Ahora sí, propón un primer paso.", "Quiero una idea sencilla para empezar.", "Prefiero que me propongas algo práctico.", "Ahora busco un paso pequeño.", "Ayúdame a empezar con una propuesta.", "Esta vez quiero un primer paso.", "Pasemos a una idea práctica para comenzar.", "Ya quiero que me sugieras cómo empezar."),
 "ask_mode": ("¿Qué quiero de esta charla ahora?", "¿Te pedí escuchar o proponer?", "¿Cómo quiero que me acompañes ahora?", "Dime qué te pedí hacer ahora.", "¿Qué prefiero en este momento?", "¿Quieres recordar mi petición actual?", "¿Cuál fue mi última petición para la charla?", "¿Qué te toca hacer según lo que pedí?"),
 "introduce_name_city": ("Hola, soy {name} y estoy en {city}.", "Me llamo {name}; hoy estoy en {city}.", "Soy {name}. Te escribo desde {city}.", "Mi nombre es {name} y ahora estoy en {city}.", "Aquí {name}, desde {city}.", "Soy {name} y me encuentro en {city}.", "{name} es mi nombre; estoy en {city}.", "Me presento: {name}, ahora en {city}."),
 "set_colors": ("Mi {item} es {color}; mi {other_item}, {other_color}.", "Tengo {article} {item} {color} y {other_article} {other_item} {other_color}.", "{Article} {item} es {color} y {other_article} {other_item} es {other_color}.", "Dos colores: mi {item} es {color} y mi {other_item} es {other_color}.", "Por cierto, mi {item} es {color}; mi {other_item} es {other_color}.", "El color de mi {item} es {color}; el de mi {other_item}, {other_color}.", "Aquí tengo mi {item} {color} y mi {other_item} {other_color}.", "Para recordar: {article} {item} es {color} y {other_article} {other_item} es {other_color}."),
 "ask_relation": ("¿Qué sabes por lo que te conté?", "Relaciona brevemente lo que te dije.", "¿Qué recuerdas de mí y del tema?", "Resume mi nombre, lugar y tema si los sabes.", "Une los datos que compartí, sin inventar.", "Dime los datos actuales de esta charla.", "¿Puedes conectar lo que ya te conté?", "Haz un resumen corto de los hechos."),
}

FLOWS = {
 "name-alias-correction": ("introduce_name", "ask_own_name", "introduce_alias", "correct_name", "ask_real_name"),
 "alias-before-real-name": ("introduce_alias", "ask_real_name", "introduce_name", "ask_alias", "ask_own_name"),
 "city-name-independent": ("introduce_city", "introduce_name", "correct_city", "ask_real_name", "ask_city"),
 "topic-city-independent": ("introduce_topic", "introduce_city", "correct_topic", "ask_city", "ask_topic"),
 "color-correction-other-kept": ("introduce_name", "set_color", "set_other_color", "correct_color", "ask_other_color"),
 "color-correction-recalled": ("introduce_city", "set_color", "set_other_color", "correct_color", "ask_color"),
 "listen-to-propose": ("introduce_topic", "listen", "ask_own_name", "propose", "ask_mode"),
 "propose-to-listen": ("introduce_name", "propose", "introduce_topic", "listen", "ask_mode"),
 "connect-shared-facts": ("introduce_name_city", "introduce_topic", "set_color", "ask_relation"),
 "unknown-city-then-shared": ("introduce_alias", "ask_city", "introduce_city", "ask_own_name", "ask_city"),
 "independent-name-topic-corrections": ("introduce_topic", "introduce_name", "correct_name", "correct_topic", "ask_relation"),
 "two-objects-topic-independent": ("introduce_name", "set_colors", "introduce_topic", "ask_color", "ask_topic"),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prompt_hash(messages):
    normalized = [(m["role"], " ".join(unicodedata.normalize("NFKC", m["content"]).casefold().split())) for m in messages]
    return sha(canonical(normalized))


def bank_identity():
    return {"userText": USER_TEXT, "flows": FLOWS, "names": NAMES, "surnames": SURNAMES,
            "cityBases": CITY_BASES, "cityEndings": CITY_ENDINGS, "topicBases": TOPIC_BASES,
            "topicEndings": TOPIC_ENDINGS, "aliasBases": ALIAS_BASES, "aliasEndings": ALIAS_ENDINGS,
            "items": ITEMS, "colors": COLORS, "publicCities": PUBLIC_CITIES}


def _pool(first, second, rng):
    values = [a + " " + b for a in first for b in second]
    rng.shuffle(values)
    return values


def _varied_values(one, two, three, rng, count, occupied):
    banks = [[v for v in dict.fromkeys(values) if v not in occupied] for values in (one, two, three)]
    for bank in banks: rng.shuffle(bank)
    used = Counter()
    result = []
    for index in range(count):
        shape = 0 if index % 24 == 0 else (1 if index % 2 else 2)
        result.append(banks[shape][used[shape]])
        used[shape] += 1
    if len(set(result)) != len(result): raise ValueError("Compound value bank repeats a value")
    rng.shuffle(result)
    occupied.update(result)
    return result


def _color(color, article):
    return color[:-1] + "a" if article == "la" and color.endswith("o") else color


def make_worlds(seed=20261015, worlds=120):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in 0..4294967295")
    if isinstance(worlds, bool) or not isinstance(worlds, int) or not 80 <= worlds <= 180:
        raise ValueError("worlds must be an integer in 80..180")
    rng = random.Random(seed)
    occupied = set()
    names = _varied_values(NAMES, [a+" "+b for a in NAMES for b in SURNAMES],
                           [a+" de "+b for a in NAMES for b in SURNAMES], rng, 2*worlds, occupied)
    cities = _varied_values(PUBLIC_CITIES, [a+" "+b for a in CITY_BASES for b in CITY_ENDINGS],
                            [a+" del "+b for a in CITY_BASES for b in ("Pino", "Mar", "Bosque", "Lago", "Olmo", "Prado", "Río", "Roble", "Valle", "Sol", "Norte", "Sur", "Jardín", "Cerro", "Faro", "Puente", "Campo", "Monte", "Camino", "Trigo", "Parque", "Pueblo", "Cielo", "Paseo")], rng, 2*worlds, occupied)
    topics = _varied_values(TOPIC_BASES, [a+" "+b for a in TOPIC_BASES for b in TOPIC_ENDINGS],
                            [a+" de "+b for a in TOPIC_BASES for b in ("rutas", "cuentos", "colores", "viajes", "lectura", "dibujos", "cocina", "música", "jardines", "papel", "ideas", "juegos", "notas", "paseos", "plantas", "mapas", "fotos", "clases", "pintura", "cartas", "palabras", "horarios", "amigos", "estudio")], rng, 2*worlds, occupied)
    aliases = _varied_values(ALIAS_BASES, [a+" "+b for a in ALIAS_BASES for b in ALIAS_ENDINGS],
                            [a+" del "+b for a in ALIAS_BASES for b in ("Mar", "Bosque", "Lago", "Norte", "Sur", "Valle", "Río", "Monte", "Pino", "Prado", "Jardín", "Campo", "Cerro", "Parque", "Puente", "Camino")], rng, worlds, occupied)
    result = []
    for index in range(worlds):
        item, other = rng.sample(ITEMS, 2)
        color, other_color, new_color = rng.sample(COLORS, 3)
        result.append({"name": names[2*index], "new_name": names[2*index+1], "alias": aliases[index],
                       "city": cities[2*index], "new_city": cities[2*index+1],
                       "topic": topics[2*index], "new_topic": topics[2*index+1],
                       "item": item[0], "article": item[1], "Article": item[1].capitalize(),
                       "other_item": other[0], "other_article": other[1], "Other_article": other[1].capitalize(),
                       "color": _color(color, item[1]), "other_color": _color(other_color, other[1]),
                       "new_color": _color(new_color, item[1])})
    return result


def author(events, world, styles=None):
    """Author targets using only facts stated so far; not an inference function."""
    styles = [0] * len(events) if styles is None else styles
    if len(styles) != len(events) or not 1 <= len(events) <= 5:
        raise ValueError("Authoring requires one to five exchanges and matching styles")
    state = {"name": None, "alias": None, "city": None, "topic": None, "mode": None, "colors": {}}
    messages = []
    for event, style in zip(events, styles):
        if event not in USER_TEXT or isinstance(style, bool) or not isinstance(style, int) or not 0 <= style < 8:
            raise ValueError("Unknown event or paraphrase")
        user = USER_TEXT[event][style].format(**world)
        if event in ("introduce_name", "correct_name", "introduce_name_city"):
            state["name"] = world["new_name" if event == "correct_name" else "name"]
            reply = f"Te llamas {state['name']}."
            if event == "introduce_name_city":
                state["city"] = world["city"]
                reply = f"Hola, {state['name']}. Estás en {state['city']}."
        elif event == "introduce_alias":
            state["alias"] = world["alias"]
            reply = f"Te llamaré {state['alias']}."
        elif event == "ask_real_name":
            reply = f"Te llamas {state['name']}." if state["name"] else "Todavía no me has dicho tu nombre real."
        elif event == "ask_alias":
            reply = f"Quieres que te llame {state['alias']}." if state["alias"] else "No has elegido un apodo."
        elif event == "ask_own_name":
            reply = "Soy Unidad."
        elif event in ("introduce_city", "correct_city"):
            state["city"] = world["new_city" if event == "correct_city" else "city"]
            reply = f"Estás en {state['city']}."
        elif event == "ask_city":
            reply = f"Estás en {state['city']}." if state["city"] else "No me has dicho dónde estás."
        elif event in ("introduce_topic", "correct_topic"):
            state["topic"] = world["new_topic" if event == "correct_topic" else "topic"]
            reply = f"Hablemos de {state['topic']}."
        elif event == "ask_topic":
            reply = f"Hablamos de {state['topic']}." if state["topic"] else "Todavía no has elegido un tema."
        elif event in ("set_color", "set_other_color", "correct_color", "set_colors"):
            if event != "set_other_color":
                state["colors"][world["item"]] = world["new_color" if event == "correct_color" else "color"]
                reply = f"{world['Article']} {world['item']} es {state['colors'][world['item']]}."
            if event in ("set_other_color", "set_colors"):
                state["colors"][world["other_item"]] = world["other_color"]
                other_reply = f"{world['Other_article']} {world['other_item']} es {world['other_color']}."
                reply = other_reply if event == "set_other_color" else reply + " " + other_reply
        elif event in ("ask_color", "ask_other_color"):
            other = event == "ask_other_color"
            item, article = (world["other_item"], world["other_article"]) if other else (world["item"], world["article"])
            value = state["colors"].get(item)
            reply = f"{article.capitalize()} {item} es {value}." if value else f"No me has dicho el color de tu {item}."
        elif event == "listen":
            state["mode"] = "listen"
            reply = "Te escucho. Puedes seguir contándome."
        elif event == "propose":
            state["mode"] = "propose"
            reply = f"Podemos anotar una idea para {state['topic']}." if state["topic"] else "¿Qué quieres empezar?"
        elif event == "ask_mode":
            reply = {"listen": "Quieres que te escuche, sin consejos.", "propose": "Quieres que proponga un primer paso.", None: "No has dicho si quieres que te escuche o que proponga algo."}[state["mode"]]
        elif event == "ask_relation":
            facts = []
            if state["name"]: facts.append(f"Te llamas {state['name']}.")
            if state["city"]: facts.append(f"Estás en {state['city']}.")
            if state["topic"]: facts.append(f"Hablamos de {state['topic']}.")
            reply = " ".join(facts) or "Todavía no has compartido esos datos."
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": reply}))
    return messages


def generate(seed=20261015, worlds=120):
    values = make_worlds(seed, worlds)
    reserved = max(8, worlds//10)
    bounds = (worlds-2*reserved, worlds-reserved)
    records = []
    for index, world in enumerate(values):
        split = "train" if index < bounds[0] else "validation" if index < bounds[1] else "test"
        world_id = sha(canonical(world))[:24]
        group = "natural-diversity-v2-" + world_id
        for flow_index, (flow_id, events) in enumerate(FLOWS.items()):
            styles = [(index + flow_index + 3*position) % 8 for position in range(len(events))]
            messages = author(events, world, styles)
            for prefix in range(1, len(events)+1):
                source = {"dataset": DATASET, "kind": "authored-synthetic", "worldId": world_id,
                          "flowId": flow_id, "prefixPairs": prefix, "fullFlowPairs": len(events),
                          "events": list(events[:prefix]), "paraphrases": styles[:prefix],
                          "questionEvent": events[prefix-1], "questionPosition": prefix,
                          "fictional": True, "privateConversationUsed": False, "modelResponsesUsed": False}
                record = {"id": f"{group}-{flow_id}-p{prefix}", "groupId": group, "split": split,
                          "source": source, "messages": messages[:2*prefix]}
                validate_record(record)
                records.append(record)
    return records


def audit_corpus(path):
    """Stream only occupied prompt hashes and folds from an authorized corpus."""
    path = Path(path)
    prompt_folds, opening_folds = defaultdict(set), defaultdict(set)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip(): continue
            row = json.loads(line)
            split = row["split"]
            if split not in SPLITS: raise ValueError("Historical corpus has an unknown fold")
            prompt_folds[prompt_hash(row["messages"][:-1])].add(split)
            opening_folds[prompt_hash(row["messages"][:1])].add(split)
            count += 1
    return prompt_folds, opening_folds, {"sha256": digest.hexdigest(), "records": count,
            "scope": "Occupied normalized first/full prompt hashes and folds only; no targets or metrics retained",
            "historicalFoldsChanged": False}


def exclusive_write(path, raw):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Existing outputs are immutable; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".natural-diversity-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try: os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("Another writer created output; it was preserved") from exc
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def build(tokenizer_path, output_path, manifest_path, seed=20261015, worlds=120, context_length=256, historical_corpus_path=None):
    tokenizer_path, output, manifest = Path(tokenizer_path), Path(output_path), Path(manifest_path)
    if isinstance(context_length, bool) or not isinstance(context_length, int) or not 32 <= context_length <= 4096:
        raise ValueError("context_length must be an integer in 32..4096")
    inputs = [tokenizer_path] + ([] if historical_corpus_path is None else [Path(historical_corpus_path)])
    if len({p.resolve() for p in inputs + [output, manifest]}) != len(inputs)+2:
        raise ValueError("Inputs and outputs must have distinct paths")
    if any(p.exists() or p.is_symlink() for p in (output, manifest)):
        raise ValueError("Existing outputs are immutable; choose new paths")
    tokenizer_raw = tokenizer_path.read_bytes()
    if len(tokenizer_raw)>16_777_216: raise ValueError("Tokenizer exceeds its local byte limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(tokenizer_raw.decode("utf-8")), strict=True)
    candidates = generate(seed, worlds)
    old_prompts, old_openings, old_report = ({}, {}, None) if historical_corpus_path is None else audit_corpus(historical_corpus_path)
    prompts, openings, seen = {}, {}, set()
    selected, lengths, reasons = [], [], Counter()
    collisions = {"crossFoldFullPrompts": 0, "crossFoldFirstPrompts": 0, "historicalCrossFoldFullPrompts": 0, "historicalCrossFoldFirstPrompts": 0}
    for row in candidates:
        split = row["split"]
        prompt, opening = prompt_hash(row["messages"][:-1]), prompt_hash(row["messages"][:1])
        target = row["messages"][-1]["content"]
        if prompt in prompts and prompts[prompt] != (split, target):
            raise ValueError("A full prompt crosses folds or has conflicting authored targets")
        if opening in openings and openings[opening] != split:
            raise ValueError("A first prompt crosses folds")
        if any(old != split for old in old_prompts.get(prompt, ())):
            raise ValueError("A full prompt conflicts with a historical fold")
        if any(old != split for old in old_openings.get(opening, ())):
            raise ValueError("A first prompt conflicts with a historical fold")
        prompts[prompt], openings[opening] = (split, target), split
    # Audit the complete bank before context checks or deduplication.
    for row in candidates:
        tokens, _ = dialogue_tokens(row["messages"], tokenizer)
        if len(tokens)>context_length+1:
            raise ValueError("A complete dialogue exceeds context; revise authored text, never truncate or move folds")
        exact = canonical(row["messages"])
        if exact in seen:
            reasons["within_world_exact_prefix_duplicates_removed"] += 1
            continue
        seen.add(exact)
        selected.append(row)
        lengths.append(len(tokens))
    flow_folds, paraphrases = defaultdict(set), defaultdict(lambda: defaultdict(set))
    world_folds = defaultdict(set)
    for row in selected:
        source = row["source"]
        flow_folds[source["flowId"]].add(row["split"])
        world_folds[row["groupId"]].add(row["split"])
        for event, style in zip(source["events"], source["paraphrases"]):
            paraphrases[event][row["split"]].add(style)
    if set(flow_folds) != set(FLOWS) or any(folds != set(SPLITS) for folds in flow_folds.values()):
        raise ValueError("Every flow must occur in every fold")
    if any(len(folds) != 1 for folds in world_folds.values()):
        raise ValueError("A world crosses folds")
    if any(set(folds) != set(SPLITS) or any(styles != set(range(8)) for styles in folds.values()) for folds in paraphrases.values()):
        raise ValueError("Every shared event paraphrase must occur in every fold")
    raw = b"".join(canonical(row)+b"\n" for row in selected)
    ordered = sorted(lengths)
    report = {"schemaVersion": 1, "dataset": DATASET, "purpose": "auxiliary-training-curriculum", "status": "FROZEN",
              "sha256": sha(raw), "seed": seed, "worldsRequested": worlds, "records": len(selected),
              "splits": dict(Counter(row["split"] for row in selected)),
              "groupSplits": {split: sum(folds == {split} for folds in world_folds.values()) for split in SPLITS},
              "flows": {flow: sorted(folds) for flow, folds in sorted(flow_folds.items())},
              "paraphrases": {event: {split: sorted(styles) for split, styles in sorted(folds.items())} for event, folds in sorted(paraphrases.items())},
              "firstPrompts": len(openings), "uniqueFullPrompts": len(prompts), "collisionAudit": collisions,
              "fullDialogueTokens": {"min": min(lengths), "median": ordered[len(ordered)//2], "p95": ordered[math.ceil(.95*len(ordered))-1], "max": max(lengths)},
              "tokenizer": {"sha256": sha(tokenizer_raw), "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
              "historicalCorpusAudit": old_report,
              "bankSha256": sha(canonical(bank_identity())), "generatorSha256": sha(Path(__file__).read_bytes()),
              "authoring": {"contextLength": context_length, "fullDialogueTokenLimit": context_length+1, "defaultSystemIncluded": True,
                            "eosIncluded": True, "dialoguesTruncated": False, "completeFlowsExcluded": 0,
                            "prefixSupervision": "last assistant response only", "openingPolicy": "natural explicit fictional facts; no fold-specific paraphrases",
                            "splitPolicy": "80/10/10 complete worlds at default; fixed before token/collision checks; all flows/paraphrases/prefixes stay together",
                            "valuePolicy": "globally unique names, aliases, cities and projects of one/two/three words drawn from common lexical pools",
                            "valueLengthPolicy": "randomized independently of initial/corrected role; exact values disjoint across all domains and folds",
                            "statePolicy": "only explicit user facts; independent attributes and objects; corrections replace their own attribute",
                            "inferenceUsesAuthoringState": False, "modelResponsesUsed": False, "modelScoresUsed": False,
                            "evaluationRead": False, "privateConversationUsed": False, "externalAIInferenceUsed": False},
              "revision": {"supersedes": "unidad-natural-diversity-v1", "reason": "Removed value-length correlation with initial/corrected role and cross-domain value overlaps; no evaluation was read", "previousPrivateArtifactsPreserved": True, "foldPolicyUnchanged": True},
              "review": {"reviewer": "development-assistant", "humanReviewClaimed": False, "scope": "fictional data design and independent semantic tests; no model competence claim"},
              "reasons": dict(reasons),
              "limitations": ["Finite authored dialogues and successful target tests do not demonstrate generated conversation competence.",
                              "Fictional places and titles are compositional practice values, not facts about the creator.",
                              "Context is checked with the unchanged supplied tokenizer and default system; no data or fold is adjusted to a measured model response."]}
    exclusive_write(output, raw)
    load_corpus(output)
    exclusive_write(manifest, canonical(report)+b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--historical-corpus", type=Path)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--worlds", type=int, default=120)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        report = build(args.tokenizer, args.output, args.manifest, args.seed, args.worlds, args.context_length, args.historical_corpus)
    except (ValueError, OSError) as exc: parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
