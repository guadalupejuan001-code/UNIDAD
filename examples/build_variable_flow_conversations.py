"""Author fictional conversations with varied causal event ordering.

State below authors training targets only. Nothing in model inference, the CLI
chat or Brain imports this helper. No weights, previous corpus, evaluations or
personal history are read. Whole worlds keep their fold across all flow orders,
phrasing variants and final-reply training prefixes.
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

from tokenizers import Tokenizer

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer


SPLITS = ("train", "validation", "test")
REAL_NAMES = ("Ada", "Ana", "Aroa", "Bruno", "Biel", "Berta", "Celia", "Ciro", "Dalia", "Diego", "Elena", "Eva",
              "Fabio", "Fiona", "Gael", "Gina", "Hugo", "Helena", "Inés", "Ivo", "Joel", "Julia", "Karla", "Kiko",
              "Lara", "Leo", "Mara", "Mario", "Nora", "Nuno", "Olga", "Omar", "Pablo", "Paz", "Quique", "Quena",
              "Rita", "Rubén", "Sara", "Saúl", "Tania", "Teo", "Ula", "Unai", "Vera", "Víctor", "Wanda", "Waldo",
              "Xenia", "Xavi", "Yara", "Yago", "Zoe", "Zeno")
ITEMS = ("cuaderno", "mapa", "libro", "bolso", "lápiz", "reloj", "vaso", "sobre", "plato", "cartel", "dibujo", "abrigo")
COLORS = ("rojo", "azul", "verde", "gris", "negro", "blanco", "amarillo", "morado")


# Each flow contains four to six USER exchanges (eight to twelve messages).
# Independent events are deliberately interleaved; correction/query chains
# have explicit causal antecedents. Identity is asked at positions one to six.
FLOWS = {
    "identity-between-names": ("greet", "ask_own_name", "introduce_real_name", "ask_real_name", "correct_real_name", "ask_real_name"),
    "greeting-after-intro": ("ask_own_name", "introduce_real_name", "greet", "correct_real_name", "ask_own_name", "ask_real_name"),
    "unknown-before-intro": ("ask_real_name", "greet", "introduce_real_name", "ask_own_name", "correct_real_name", "ask_real_name"),
    "real-versus-preferred": ("introduce_real_name", "ask_real_name", "set_preferred_alias", "ask_own_name", "ask_preferred_alias", "ask_real_name"),
    "alias-does-not-reveal-real": ("greet", "set_preferred_alias", "ask_real_name", "ask_preferred_alias", "introduce_real_name", "ask_real_name"),
    "topic-after-identity": ("ask_topic", "set_topic", "ask_own_name", "correct_topic", "greet", "ask_topic"),
    "topic-and-independent-city": ("set_topic", "ask_topic", "greet", "set_city", "correct_topic", "ask_city"),
    "unknown-and-new-city": ("ask_city", "greet", "set_city", "ask_own_name", "correct_city", "ask_city"),
    "listen-to-action": ("greet", "set_intent_listen", "ask_own_name", "ask_intent", "set_intent_act", "ask_intent"),
    "action-to-listening": ("set_intent_act", "greet", "set_topic", "set_intent_listen", "ask_topic", "ask_intent"),
    "two-colors-and-correction": ("ask_color", "set_colors", "greet", "ask_other_color", "correct_color", "ask_color"),
    "count-after-greeting": ("ask_count", "set_count", "ask_own_name", "correct_count", "greet", "ask_count"),
    "short-greeting-identity": ("greet", "introduce_real_name", "ask_own_name", "ask_real_name"),
    "short-name-correction": ("introduce_real_name", "greet", "correct_real_name", "ask_real_name"),
    "short-topic-current": ("greet", "set_topic", "ask_own_name", "ask_topic", "correct_topic"),
    "alias-and-identity-late": ("set_preferred_alias", "ask_real_name", "introduce_real_name", "ask_preferred_alias", "ask_own_name"),
    "identity-at-six": ("introduce_real_name", "set_topic", "greet", "correct_real_name", "ask_topic", "ask_own_name"),
    "name-at-five": ("greet", "set_city", "introduce_real_name", "correct_real_name", "ask_real_name", "ask_city"),
    "topic-at-three": ("set_topic", "greet", "ask_topic", "ask_own_name"),
    "city-at-two-and-four": ("set_city", "ask_city", "greet", "ask_city", "correct_city", "ask_city"),
    "city-at-three-and-five": ("greet", "set_city", "ask_city", "ask_own_name", "ask_city"),
}


def _name(index):
    if index % 3 == 0:
        return REAL_NAMES[(index // 3) % len(REAL_NAMES)]
    onset = ("B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y", "Z", "Br", "Cl", "Fr", "Gr", "Pr", "Tr", "Vr")
    vowels = ("a", "e", "i", "o", "u", "ia")
    endings = ("na", "ro", "la", "mi", "ri", "ne", "lo", "si", "ra", "ni")
    return onset[index % len(onset)] + vowels[(index // len(onset)) % len(vowels)] + endings[(index // (len(onset)*len(vowels))) % len(endings)]


def _topic(index):
    first = ("Mapa", "Libro", "Plan", "Taller", "Cuaderno", "Camino", "Puente", "Faro")
    second = ("Azul", "Norte", "Sur", "Claro", "Nuevo", "Breve", "Abierto", "Sereno", "Verde", "Simple", "Largo", "Corto")
    return first[index % len(first)] + " " + second[(index//len(first)) % len(second)]


def make_world(seed, index):
    rng = random.Random(seed + index*104729)
    k = rng.randrange(3000)
    name, replacement, alias = (_name(k + step) for step in (0, 113, 251))
    while len({name, replacement, alias}) != 3:
        k += 1
        name, replacement, alias = (_name(k + step) for step in (0, 113, 251))
    item, other = rng.sample(ITEMS, 2)
    color, other_color, replacement_color = rng.sample(COLORS, 3)
    count = rng.randrange(2,7)
    new_count = 2 + ((count-2 + rng.randrange(1,5)) % 5)
    return {"name": name, "new_name": replacement, "alias": alias,
            "topic": _topic(k), "new_topic": _topic(k+37),
            "city": "Villa " + _name(k+479), "new_city": "Puerto " + _name(k+683),
            "item": item, "other_item": other, "color": color, "other_color": other_color,
            "new_color": replacement_color, "count": count, "new_count": new_count}


USER_TEXT = {
    "greet": ("Hola, Unidad.", "Buenas, ¿charlamos?", "Hola. Estoy por aquí."),
    "ask_own_name": ("¿Cómo te llamas?", "Dime tu nombre.", "¿Quién eres?"),
    "ask_real_name": ("¿Cómo me llamo?", "¿Cuál es mi nombre real?", "Dime mi nombre, no tu nombre."),
    "introduce_real_name": ("Me llamo {name}.", "Mi nombre es {name}.", "Soy {name}."),
    "correct_real_name": ("Corrijo mi nombre: es {new_name}.", "Mi nombre correcto es {new_name}.", "Cambia mi nombre a {new_name}."),
    "set_preferred_alias": ("Llámame {alias}.", "Prefiero que me digas {alias}.", "Para esta charla, usa {alias} como apodo."),
    "ask_preferred_alias": ("¿Cómo quiero que me llames?", "¿Qué apodo elegí?", "Dime el nombre que prefiero usar aquí."),
    "set_topic": ("Quiero hablar de {topic}.", "El tema es {topic}.", "Hablemos del proyecto {topic}."),
    "correct_topic": ("Cambio de tema: {new_topic}.", "Ahora hablemos de {new_topic}.", "Prefiero el proyecto {new_topic} como tema."),
    "ask_topic": ("¿De qué tema hablamos?", "¿Cuál es el proyecto actual?", "Dime el tema que elegí ahora."),
    "set_city": ("Mi ciudad es {city}.", "Vivo en {city}.", "La ciudad donde vivo es {city}."),
    "correct_city": ("Corrijo la ciudad: {new_city}.", "Ahora vivo en {new_city}.", "Mi ciudad actual es {new_city}."),
    "ask_city": ("¿Cuál es mi ciudad?", "¿Dónde vivo?", "Dime la ciudad que indiqué."),
    "set_intent_listen": ("Sólo quiero que me escuches, sin consejos.", "Ahora quiero conversar sin preparar un plan.", "Por el momento, prefiero que me escuches."),
    "set_intent_act": ("Ahora quiero un primer paso práctico.", "Prefiero que propongas un paso sencillo.", "Quiero una propuesta para empezar."),
    "ask_intent": ("¿Qué quiero de esta charla ahora?", "¿Qué te pido ahora: escuchar o proponer?", "Dime mi intención actual."),
    "set_colors": ("El {item} es {color}; el {other_item} es {other_color}.", "Mi {item} tiene color {color} y mi {other_item}, {other_color}.", "Anoto dos colores: {item}, {color}; {other_item}, {other_color}."),
    "correct_color": ("El {item} ahora es {new_color}.", "Corrijo el color del {item}: es {new_color}.", "El color correcto del {item} es {new_color}."),
    "ask_color": ("¿De qué color es el {item}?", "Dime el color del {item}.", "Ahora pregunto por el {item}. ¿Qué color tiene?"),
    "ask_other_color": ("¿De qué color es el {other_item}?", "Dime el color del {other_item}.", "Ahora pregunto por el {other_item}. ¿Qué color tiene?"),
    "set_count": ("Tengo {count} fichas.", "Hay {count} fichas en mi caja.", "Mi caja contiene {count} fichas."),
    "correct_count": ("Corrijo la cantidad: tengo {new_count} fichas.", "Ahora son {new_count} fichas.", "La cantidad correcta es {new_count} fichas."),
    "ask_count": ("¿Cuántas fichas tengo?", "Dime la cantidad actual de fichas.", "¿Cuántas fichas hay en mi caja?"),
}


# Only an event that can begin a flow needs a fold-specific opening. The
# first text still expresses exactly the same event and contains no hidden
# fields. Interleaved later turns reuse ordinary phrasings across folds.
OPENING_TEXT = {
    "greet": {"train": USER_TEXT["greet"], "validation": ("Buen día. Quiero conversar.", "Buenas. Estoy listo para charlar.", "Hola, podemos hablar un momento."),
              "test": ("Saludos, Unidad.", "Buenas tardes. Vengo a hablar.", "Hola, tengo un rato para conversar.")},
    "ask_own_name": {"train": USER_TEXT["ask_own_name"], "validation": ("¿Cuál es tu nombre?", "Preséntate por tu nombre.", "¿Cómo puedo llamarte?"),
                     "test": ("Dime cómo puedo llamarte.", "¿Qué nombre tienes?", "Quiero conocer tu nombre antes de seguir.")},
    "ask_real_name": {"train": ("Dime cómo me llamo.", "¿Sabes mi nombre?", "Adivina mi nombre."),
                      "validation": ("¿Qué nombre mío conoces?", "Sin presentarme, ¿sabes mi nombre?", "Todavía no me presenté. Dime mi nombre."),
                      "test": ("¿Puedes decir mi nombre sin que lo comparta?", "No he dado mi nombre. ¿Lo sabes?", "¿Cómo me llamo si aún no me presenté?")},
    "introduce_real_name": {"train": USER_TEXT["introduce_real_name"], "validation": ("Mi nombre propio es {name}.", "Me presento: {name}.", "Para presentarme, mi nombre es {name}."),
                            "test": ("Aquí está mi nombre: {name}.", "Quien te habla se llama {name}.", "Te digo mi nombre: {name}.")},
    "set_preferred_alias": {"train": USER_TEXT["set_preferred_alias"], "validation": ("Para dirigirte a mí, prefiero {alias}.", "Elijo {alias} como apodo aquí.", "Te pido que me llames {alias}."),
                            "test": ("Por favor, llámame {alias} en esta conversación.", "Quiero usar el apodo {alias}.", "Mi elección para esta charla es el apodo {alias}.")},
    "ask_topic": {"train": ("¿Qué tema elegí?", "¿De qué quiero hablar?", "Dime el tema de esta conversación."),
                  "validation": ("¿Qué proyecto he mencionado?", "¿Ya indiqué un tema?", "Quiero saber qué tema he dado."),
                  "test": ("Sin que te lo diga, ¿qué tema quiero?", "¿Conoces ya el tema que elegí?", "No he dicho un tema. ¿Cuál es?")},
    "set_topic": {"train": USER_TEXT["set_topic"], "validation": ("Mi tema de conversación es {topic}.", "Quiero comentar el proyecto {topic}.", "Vamos a conversar sobre {topic}."),
                  "test": ("Me gustaría que el tema fuera {topic}.", "Te propongo hablar de {topic}.", "Hoy elijo el tema {topic}.")},
    "ask_city": {"train": ("¿En qué ciudad vivo?", "¿Sabes mi ciudad?", "Dime dónde vivo."),
                 "validation": ("¿Conoces la ciudad donde vivo?", "¿Qué ciudad he dicho?", "¿Ya sabes dónde vivo?"),
                 "test": ("No he indicado mi ciudad. ¿Cuál es?", "Sin decirte dónde vivo, ¿puedes saberlo?", "¿Tienes algún dato de mi ciudad?")},
    "set_city": {"train": USER_TEXT["set_city"], "validation": ("Te digo mi ciudad: {city}.", "Actualmente vivo en {city}.", "Mi lugar de residencia es {city}."),
                 "test": ("La ciudad donde resido es {city}.", "Ésta es mi ciudad: {city}.", "Para ubicarme, vivo en {city}.")},
    "set_intent_act": {"train": USER_TEXT["set_intent_act"], "validation": ("Me vendría bien un paso para empezar.", "Busco una propuesta concreta, no sólo hablar.", "Ayúdame a empezar con una idea práctica."),
                       "test": ("Necesito que sugieras un primer paso.", "Prefiero pasar a una propuesta sencilla.", "Quiero una idea para actuar, sin asumir mi objetivo.")},
    "ask_color": {"train": USER_TEXT["ask_color"], "validation": ("¿Conoces el color de mi {item}?", "¿Qué color del {item} te he dicho?", "¿Ya sabes de qué color es el {item}?"),
                  "test": ("Sin que lo indique, ¿sabes el color del {item}?", "Aún no di el color del {item}. ¿Cuál es?", "¿Qué dato de color tienes del {item}?")},
    "ask_count": {"train": USER_TEXT["ask_count"], "validation": ("¿Sabes la cantidad de fichas que tengo?", "¿Qué cantidad de fichas he dado?", "¿Ya indiqué cuántas fichas hay?"),
                  "test": ("No di la cantidad de fichas. ¿Cuántas son?", "Sin decirlo, ¿sabes cuántas fichas tengo?", "¿Tienes algún dato del número de fichas?")},
}


def _event(event):
    if isinstance(event, str):
        return event, 0
    if not isinstance(event, dict) or set(event)-{"type", "style"}:
        raise ValueError("Authoring event must be a type with optional style")
    kind, style = event.get("type"), event.get("style",0)
    if isinstance(style,bool) or not isinstance(style,int) or style not in (0,1,2):
        raise ValueError("Authoring style must be 0,1,2")
    return kind, style


def author(events, world, split):
    """Replay explicit authoring events; latent world values are never memory.

    Only declaration events populate state. Question events consult that state,
    not the world's hidden answer fields. The returned messages are synthetic
    training text; this function is never called by model generation.
    """
    if split not in SPLITS or not isinstance(events,(tuple,list)) or not 1<=len(events)<=6:
        raise ValueError("Authoring needs a known fold and one to six exchanges")
    state = {"real":None, "alias":None, "topic":None, "city":None, "intent":None, "count":None, "colors":{}}
    messages=[]
    for position,event in enumerate(events):
        kind,style = _event(event)
        if kind not in USER_TEXT:
            raise ValueError("Unknown authoring event")
        user = USER_TEXT[kind][style]
        if position==0:
            if kind not in OPENING_TEXT:
                raise ValueError("This event has no reviewed cold opening")
            user=OPENING_TEXT[kind][split][style]
        user=user.format(**world)
        if kind=="greet":
            reply=("Hola. Te leo.","Buenas. Podemos charlar.","Hola. ¿Qué tienes en mente?")[style]
        elif kind=="ask_own_name":
            reply=("Soy Unidad.","Me llamo Unidad.","Mi nombre es Unidad.")[style]
        elif kind in ("introduce_real_name","correct_real_name"):
            state["real"]=world["name" if kind=="introduce_real_name" else "new_name"]
            reply=f"De acuerdo, te llamas {state['real']}."
        elif kind=="ask_real_name":
            reply=f"Te llamas {state['real']}." if state["real"] is not None else "Todavía no has dicho tu nombre."
        elif kind=="set_preferred_alias":
            state["alias"]=world["alias"]
            reply=f"De acuerdo, te llamaré {state['alias']}."
        elif kind=="ask_preferred_alias":
            reply=f"Quieres que te llame {state['alias']}." if state["alias"] is not None else "No has indicado un apodo."
        elif kind in ("set_topic","correct_topic"):
            state["topic"]=world["topic" if kind=="set_topic" else "new_topic"]
            reply=f"Hablemos de {state['topic']}."
        elif kind=="ask_topic":
            reply=f"El tema actual es {state['topic']}." if state["topic"] is not None else "Todavía no has elegido un tema."
        elif kind in ("set_city","correct_city"):
            state["city"]=world["city" if kind=="set_city" else "new_city"]
            reply=f"Tu ciudad es {state['city']}."
        elif kind=="ask_city":
            reply=f"Tu ciudad es {state['city']}." if state["city"] is not None else "No has dicho tu ciudad."
        elif kind=="set_intent_listen":
            state["intent"]="listen"
            reply="Te escucho, sin proponer un plan."
        elif kind=="set_intent_act":
            state["intent"]="act"
            reply="¿Qué quieres empezar? Podemos buscar un paso sencillo."
        elif kind=="ask_intent":
            reply={"listen":"Quieres que te escuche.", "act":"Quieres que proponga un primer paso.", None:"Todavía no has indicado qué quieres de la charla."}[state["intent"]]
        elif kind=="set_colors":
            state["colors"][world["item"]]=world["color"]
            state["colors"][world["other_item"]]=world["other_color"]
            reply=f"El {world['item']} es {world['color']} y el {world['other_item']} es {world['other_color']}."
        elif kind=="correct_color":
            state["colors"][world["item"]]=world["new_color"]
            reply=f"El {world['item']} es {world['new_color']}."
        elif kind in ("ask_color","ask_other_color"):
            item=world["item" if kind=="ask_color" else "other_item"]
            value=state["colors"].get(item)
            reply=f"El {item} es {value}." if value is not None else f"No has dicho el color del {item}."
        elif kind in ("set_count","correct_count"):
            state["count"]=world["count" if kind=="set_count" else "new_count"]
            reply=f"Tienes {state['count']} fichas."
        elif kind=="ask_count":
            reply=f"Tienes {state['count']} fichas." if state["count"] is not None else "No has dicho cuántas fichas tienes."
        messages.extend(({"role":"user","content":user},{"role":"assistant","content":reply}))
    return messages


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def generate(seed=20261014,worlds=100):
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=2**32-1:
        raise ValueError("seed must be an integer in 0..4294967295")
    if isinstance(worlds,bool) or not isinstance(worlds,int) or not 30<=worlds<=1000:
        raise ValueError("worlds must be an integer in 30..1000")
    reserved=max(3,worlds//10)
    boundaries=(worlds-2*reserved,worlds-reserved)
    records=[]
    for index in range(worlds):
        split="train" if index<boundaries[0] else "validation" if index<boundaries[1] else "test"
        world=make_world(seed,index)
        world_id=sha(canonical(world))[:24]
        group="variable-flow-v1-"+world_id
        for flow_index,(flow_id,kinds) in enumerate(FLOWS.items()):
            events=[{"type":kind,"style":(index+flow_index+position)%3} for position,kind in enumerate(kinds)]
            messages=author(events,world,split)
            for prefix in range(1,len(events)+1):
                record={"id":f"{group}-{flow_id}-p{prefix}","groupId":group,"split":split,
                        "source":{"dataset":"unidad-variable-flow-v1","kind":"authored-synthetic","worldId":world_id,
                                  "flowId":flow_id,"prefixPairs":prefix,"fullFlowPairs":len(events),
                                  "events":[event["type"] for event in events[:prefix]],
                                  "questionEvent":events[prefix-1]["type"],"questionPosition":prefix,
                                  "fictional":True,"privateConversationUsed":False,"modelResponsesUsed":False},
                        "messages":messages[:2*prefix]}
                validate_record(record)
                records.append(record)
    return records


def exclusive_write(path,raw):
    path=Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Existing outputs are immutable; choose a new path")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="wb",prefix=".variable-flow-",dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary,path)
        except FileExistsError as exc:
            raise ValueError("Another writer created output; it was preserved") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(tokenizer_path,output_path,manifest_path,seed=20261014,worlds=100,context_length=256):
    tokenizer_path,output,manifest=Path(tokenizer_path),Path(output_path),Path(manifest_path)
    if isinstance(context_length,bool) or not isinstance(context_length,int) or not 32<=context_length<=4096:
        raise ValueError("context_length must be an integer in 32..4096")
    if len({path.resolve() for path in (tokenizer_path,output,manifest)})!=3:
        raise ValueError("Tokenizer and outputs must have distinct paths")
    if any(path.exists() or path.is_symlink() for path in (output,manifest)):
        raise ValueError("Existing outputs are immutable; choose new paths")
    raw_tokenizer=tokenizer_path.read_bytes()
    if len(raw_tokenizer)>16_777_216:
        raise ValueError("Tokenizer exceeds its local byte limit")
    tokenizer=validate_tokenizer(Tokenizer.from_str(raw_tokenizer.decode("utf-8")),strict=True)
    candidates=generate(seed,worlds)
    # Check all unfiltered inputs first. A world can have different orderings,
    # but must never collide with an unrelated fold or contradict its prompt.
    prompts={}
    for record in candidates:
        key=canonical(record["messages"][:-1])
        value=(record["split"],record["messages"][-1]["content"])
        if key in prompts and prompts[key]!=value:
            raise ValueError("A full prompt crosses folds or has contradictory targets")
        prompts[key]=value
    excluded_flows=set()
    for record in candidates:
        source=record["source"]
        if source["prefixPairs"]==source["fullFlowPairs"]:
            tokens,_=dialogue_tokens(record["messages"],tokenizer)
            if len(tokens)>context_length+1:
                excluded_flows.add((record["groupId"],source["flowId"]))
    seen,selected,lengths,reasons=set(),[],[],Counter()
    for record in candidates:
        source=record["source"]
        if (record["groupId"],source["flowId"]) in excluded_flows:
            reasons["records_in_over_context_complete_flows_omitted"]+=1
            continue
        exact=canonical(record["messages"])
        if exact in seen:
            reasons["within_fold_exact_prefix_duplicates_removed"]+=1
            continue
        tokens,_=dialogue_tokens(record["messages"],tokenizer)
        if len(tokens)>context_length+1:
            raise ValueError("A retained prefix exceeds its complete-flow token bound")
        seen.add(exact)
        selected.append(record)
        lengths.append(len(tokens))
    flow_splits,event_splits=defaultdict(Counter),defaultdict(Counter)
    positions=defaultdict(set)
    for record in selected:
        source=record["source"]
        flow_splits[source["flowId"]][record["split"]]+=1
        event_splits[source["questionEvent"]][record["split"]]+=1
        if source["questionEvent"].startswith("ask_"):
            positions[source["questionEvent"]].add(source["questionPosition"])
    if set(flow_splits)!=set(FLOWS) or any(set(folds)!=set(SPLITS) for folds in flow_splits.values()):
        raise ValueError("Every reviewed flow must remain represented in every fold")
    for event in ("ask_own_name","ask_real_name","ask_topic","ask_city"):
        if not {2,3,4,5,6}.issubset(positions[event]):
            raise ValueError("Required question positions were lost after context filtering")
    raw=b"".join(canonical(record)+b"\n" for record in selected)
    ordered=sorted(lengths)
    report={"schemaVersion":1,"dataset":"unidad-variable-flow-v1","purpose":"auxiliary-training-curriculum",
            "sha256":sha(raw),"seed":seed,"worldsRequested":worlds,"records":len(selected),
            "splits":dict(Counter(record["split"] for record in selected)),
            "groupSplits":{split:len({record["groupId"] for record in selected if record["split"]==split}) for split in SPLITS},
            "flows":{flow:dict(folds) for flow,folds in sorted(flow_splits.items())},
            "targetEvents":{event:dict(folds) for event,folds in sorted(event_splits.items())},
            "questionPositions":{event:sorted(values) for event,values in sorted(positions.items())},
            "prefixPairs":dict(Counter(str(record["source"]["prefixPairs"]) for record in selected)),
            "fullFlowPairs":dict(Counter(str(record["source"]["fullFlowPairs"]) for record in selected)),
            "fullDialogueTokens":{"min":min(lengths),"median":ordered[len(lengths)//2],"p95":ordered[math.ceil(.95*len(lengths))-1],"max":max(lengths)},
            "tokenizer":{"sha256":sha(raw_tokenizer),"vocabSize":tokenizer.get_vocab_size(),"refitted":False},
            "authoring":{"flowPrograms":len(FLOWS),"completeFlowExchangeRange":[4,6],"contextLength":context_length,
                         "fullDialogueTokenLimit":context_length+1,"excludedCompleteFlows":len(excluded_flows),
                         "dialoguesTruncated":False,"prefixSupervision":"last assistant response only",
                         "splitPolicy":"80/10/10 complete worlds; all flows/styles/prefixes for a world retain one fold",
                         "openingPolicy":"fold-specific equivalent openings; generic prefixes deduplicated only within their fold",
                         "statePolicy":"only explicit declaration events set known facts; queries never read hidden world answers",
                         "inferenceUsesAuthoringState":False,"existingCorpusRead":False,"evaluationRead":False,
                         "modelResponsesUsed":False,"modelScoresUsed":False,"privateConversationUsed":False,
                         "oldCorpusCollisionAudit":"Required separately before merge; preserve all prior folds"},
            "reasons":dict(reasons),"generatorSha256":sha(Path(__file__).read_bytes()),
            "limitations":["Finite fictional flow programs and authored history do not prove free conversation.",
                           "Tests of targets must use independent hand-written transition oracles, not replay this author to invent expected values.",
                           "Identical generic prototypes are retained once per fold; they are not counted as independent worlds.",
                           "No weights, development/final evaluation, owner biography or hidden memory were used to author this dataset."]}
    exclusive_write(output,raw)
    load_corpus(output)
    exclusive_write(manifest,canonical(report)+b"\n")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=20261014)
    parser.add_argument("--worlds",type=int,default=100)
    parser.add_argument("--context-length",type=int,default=256)
    args=parser.parse_args(argv)
    try:
        report=build(args.tokenizer,args.output,args.manifest,args.seed,args.worlds,args.context_length)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
