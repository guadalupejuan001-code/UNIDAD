"""Author fictional cold starts and simple natural Spanish conversations.

No previous corpus, weights or evaluation responses are read. An optional
collision-provenance file contains only metadata and occupied prompts. Templates
author training data only; inference never routes through this helper. Equivalent
cold-start phrasings are reserved by fold. Generic prefixes are deduplicated
within their own fold, retaining a prototype rather than deleting every copy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.conversation.tokenizer import dialogue_tokens, validate_tokenizer

if __package__:
    from .build_open_fact_conversations import _name, _world
else:
    from build_open_fact_conversations import _name, _world


SPLITS = ("train", "validation", "test")
ACTIVITIES = ("leer", "pasear", "cocinar", "dibujar", "escribir", "correr", "descansar", "estudiar",
              "caminar", "cantar", "pintar", "ordenar", "practicar", "conversar", "bailar", "nadar")


def _template(template_id, opening, reply, second, third):
    return {"templateId": template_id, "openingByFold": dict(zip(SPLITS, opening)),
            "turns": [(None, reply), second, third]}


def templates():
    """Twenty reviewed patterns; each opening tuple is train/validation/test."""
    return {
        "greeting": [
            _template("greeting-start", ("Hola, Unidad.", "Buenas, aquí estoy.", "Hey, empecemos a charlar."),
                "Hola. ¿Qué tienes en mente?",
                ("Soy {name} y me apetece {activity}.", "Hola, {name}. ¿Qué te apetece contarme de {activity}?"),
                ("Mejor prefiero {other_activity}.", "Ahora prefieres {other_activity}.")),
            _template("greeting-availability", ("¿Estás ahí, Unidad?", "¿Podemos conversar ahora?", "¿Me lees por aquí?"),
                "Sí, aquí estoy. Te leo.",
                ("Me llamo {name}.", "Hola, {name}."),
                ("¿Cómo me llamo?", "Te llamas {name}.")),
            _template("greeting-unhurried", ("Hola. Tengo un rato libre.", "Buenas. Vengo a charlar un poco.", "Qué tal, tengo ganas de conversar."),
                "Hola. Podemos charlar sin prisa.",
                ("Estoy pensando en el proyecto {title}.", "¿Qué te gustaría contar sobre {title}?"),
                ("Hoy sólo quiero hablar de él, sin hacer un plan.", "Claro. Podemos hablar sin preparar un plan.")),
            _template("greeting-introduction", ("Buenos días, Unidad.", "Buenas tardes. ¿Charlamos?", "Buenas noches, estoy por aquí."),
                "Hola. ¿Cómo quieres empezar?",
                ("Puedes llamarme {alias}.", "De acuerdo, {alias}."),
                ("¿Qué nombre elegí para esta charla?", "Elegiste {alias}.")),
        ],
        "self_identity": [
            _template("identity-name", ("¿Cómo te llamas?", "¿Cuál es tu nombre?", "Dime cómo puedo llamarte."),
                "Soy Unidad.",
                ("Me llamo {name}.", "Hola, {name}."),
                ("¿Cómo me llamo yo?", "Te llamas {name}.")),
            _template("identity-roles", ("¿Quién eres?", "Preséntate con tu nombre.", "Quiero saber quién está conversando conmigo."),
                "Soy Unidad, un modelo de lenguaje.",
                ("Yo soy {name}.", "Hola, {name}."),
                ("¿Cuál es tu nombre, no el mío?", "Mi nombre es Unidad.")),
            _template("identity-distinction", ("Dime tu nombre.", "Quiero conocer tu nombre primero.", "¿Con qué nombre te presentas?"),
                "Me llamo Unidad.",
                ("El mío es {name}.", "Hola, {name}."),
                ("¿Yo soy Unidad o tú?", "Unidad soy yo; tú te presentaste como {name}.")),
            _template("identity-alias", ("¿Qué nombre uso para dirigirme a ti?", "¿Cómo debería llamarte en esta charla?", "Necesito saber tu nombre para saludarte."),
                "Puedes llamarme Unidad.",
                ("Llámame {alias}.", "De acuerdo, {alias}."),
                ("¿Sabes mi nombre real?", "No has dicho tu nombre real; sólo que quieres que te llame {alias}.")),
        ],
        "unknown_user_name": [
            _template("unknown-name-introduced-corrected", ("Dime cómo me llamo.", "¿Cuál es mi nombre?", "Dime el nombre de quien te está hablando."),
                "Todavía no me has dicho tu nombre.",
                ("Me llamo {name}.", "Hola, {name}."),
                ("Corrijo mi nombre: es {new_name}.", "Entendido, te llamas {new_name}.")),
            _template("unknown-name-later-retrieved", ("¿Sabes mi nombre?", "¿Ya conoces mi nombre?", "¿Te he dicho cómo me llamo?"),
                "Aún no has dicho tu nombre en esta charla.",
                ("Soy {name}.", "Hola, {name}."),
                ("¿Cómo me llamo entonces?", "Te llamas {name}.")),
            _template("unknown-no-guess", ("Adivina mi nombre.", "Intenta decir cómo me llamo sin que te lo diga.", "¿Puedes saber mi nombre sin que lo comparta?"),
                "No conozco tu nombre. Puedes decírmelo.",
                ("Mi nombre es {name}.", "Gracias, {name}."),
                ("En realidad, el nombre correcto es {new_name}.", "De acuerdo, el nombre correcto es {new_name}.")),
            _template("unknown-real-name-versus-alias", ("¿Qué nombre mío sabes hasta ahora?", "¿Conoces algún nombre mío?", "¿Cómo puedes llamarme si no me he presentado?"),
                "Todavía no has compartido un nombre.",
                ("Prefiero que me llames {alias}.", "De acuerdo, {alias}."),
                ("Mi nombre es {name}. ¿Cuál es mi nombre y qué apodo elegí?", "Tu nombre es {name} y elegiste {alias} como apodo.")),
        ],
        "listen_or_act": [
            _template("listen-then-step", ("Sólo quiero conversar, sin consejos.", "Quiero que me escuches sin proponer soluciones.", "Hoy necesito hablar, no resolver nada."),
                "Te escucho. Puedes contarlo a tu ritmo.",
                ("Me cuesta empezar a {activity}. Ahora sí, dame un primer paso.", "Puedes dedicar unos minutos a {activity}, sin exigirte hacerlo perfecto."),
                ("Volvamos a conversar, sin pasos que cumplir.", "Claro. Dejamos los pasos y te escucho.")),
            _template("act-then-listen", ("Quiero empezar algo con un paso pequeño.", "Necesito una idea sencilla para empezar.", "Ayúdame a dar un primer paso."),
                "¿Qué te gustaría empezar?",
                ("Quiero {activity}.", "Podemos empezar con unos minutos para {activity}."),
                ("Mejor no quiero un plan ahora, sólo hablar.", "Dejamos el plan. Podemos hablar sin decidir nada.")),
            _template("listen-without-decision", ("Escúchame un rato, por favor.", "¿Puedes escuchar lo que quiero contar?", "Necesito contar algo sin decidir todavía."),
                "Sí. Cuéntame lo que te apetezca.",
                ("Quiero contarte mi idea sobre {title}. ¿Hace falta decidir ya?", "No. Podemos hablar de {title} sin decidir todavía."),
                ("Ahora quiero dar un paso para {activity}.", "Podemos reservar unos minutos para {activity}.")),
            _template("listen-current-topic", ("Prefiero una charla tranquila, sin planes.", "Hablemos con calma, sin preparar tareas.", "Por ahora quiero conversar, sin organizar nada."),
                "De acuerdo. Podemos conversar con calma.",
                ("Pienso en {activity}. Hazme una pregunta corta sobre eso.", "¿Qué te atrae de {activity}?"),
                ("No hace falta resumirlo; sólo quiero seguir contándolo.", "Te escucho. Puedes seguir.")),
        ],
        "goal_change": [
            _template("goal-project-change", ("Tengo una idea.", "Quiero hablar de un proyecto.", "Traigo una idea para trabajar."),
                "Cuéntame de qué se trata.",
                ("Quiero trabajar en {title}.", "Quieres trabajar en {title}."),
                ("Cambié mi objetivo: ahora trabajaré en {other_title}.", "Ahora quieres trabajar en {other_title}.")),
            _template("goal-activity-change", ("Ayúdame a ordenar una idea.", "Quiero aclarar qué hacer.", "Necesito poner en claro una intención."),
                "Cuéntame qué buscas conseguir.",
                ("Mi idea es {activity}.", "Quieres {activity}."),
                ("Ya no quiero eso; ahora prefiero {other_activity}.", "Ahora prefieres {other_activity}.")),
            _template("goal-work-to-talk", ("Quiero empezar algo, pero aún no lo expliqué.", "Tengo ganas de empezar un proyecto.", "Quiero comenzar y necesito contarte qué."),
                "¿Qué quieres empezar?",
                ("El proyecto {title}.", "Quieres empezar {title}."),
                ("No voy a empezarlo hoy. Ahora sólo quiero charlar.", "Dejamos el proyecto para después. Podemos charlar.")),
            _template("goal-clarify-then-switch", ("Quiero retomar un tema.", "Sigamos con un asunto que tengo en mente.", "Quiero continuar algo, pero no lo he explicado aquí."),
                "¿Qué tema quieres retomar?",
                ("El tema es {title}.", "El tema que quieres retomar es {title}."),
                ("Cambio de tema: hablemos de {other_title}.", "Ahora hablemos de {other_title}.")),
        ],
    }


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def render(template, world, split):
    messages = []
    for index, (user, reply) in enumerate(template["turns"]):
        text = template["openingByFold"][split] if index == 0 else user
        messages.extend(({"role": "user", "content": text.format(**world)},
                         {"role": "assistant", "content": reply.format(**world)}))
    return messages


def generate(seed=20261013, episodes_per_skill=100):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise ValueError("seed must be an integer in 0..4294967295")
    if isinstance(episodes_per_skill, bool) or not isinstance(episodes_per_skill, int) or not 20 <= episodes_per_skill <= 1000:
        raise ValueError("episodes_per_skill must be an integer in 20..1000")
    reserved = max(2, episodes_per_skill//10)
    quotas = {"train": episodes_per_skill-2*reserved, "validation": reserved, "test": reserved}
    records, counters, families = [], Counter(), {}
    for skill, patterns in sorted(templates().items()):
        for split, count in quotas.items():
            for _ in range(count):
                index = counters[split]
                counters[split] += 1
                world = _world(seed, split, index)
                world["activity"] = ACTIVITIES[(index + seed) % len(ACTIVITIES)]
                world["other_activity"] = ACTIVITIES[(index + seed + 7) % len(ACTIVITIES)]
                identity = {"skill": skill, "world": world}
                world_id = sha(canonical(identity))[:24]
                if world_id in families and families[world_id] != split:
                    raise ValueError("A semantic episode occurs in multiple folds")
                families[world_id] = split
                group = "natural-basics-v2-" + world_id
                for pattern in patterns:
                    messages = render(pattern, world, split)
                    for prefix in (1,2,3):
                        record = {"id": f"{group}-{pattern['templateId']}-p{prefix}", "groupId": group,
                                  "split": split, "source": {"dataset": "unidad-natural-basics-v2", "kind": "authored-synthetic",
                                  "skill": skill, "templateId": pattern["templateId"], "worldId": world_id,
                                  "prefixPairs": prefix, "coldOpeningFold": split, "fictional": True,
                                  "privateConversationUsed": False, "modelResponsesUsed": False},
                                  "messages": messages[:2*prefix]}
                        validate_record(record)
                        records.append(record)
    return records


def exclusive_write(path, raw):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Existing outputs are immutable; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".natural-basics-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("Another writer created output; existing evidence was preserved") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(tokenizer_path, output_path, manifest_path, seed=20261013, episodes_per_skill=100, context_length=256,
          collision_audit_path=None):
    tokenizer_path, output, manifest = Path(tokenizer_path), Path(output_path), Path(manifest_path)
    if isinstance(context_length, bool) or not isinstance(context_length, int) or not 32 <= context_length <= 4096:
        raise ValueError("context_length must be an integer in 32..4096")
    if len({path.resolve() for path in (tokenizer_path, output, manifest)}) != 3:
        raise ValueError("Tokenizer and outputs must have distinct paths")
    if any(path.exists() or path.is_symlink() for path in (output, manifest)):
        raise ValueError("Existing outputs are immutable; choose new paths")
    raw_tokenizer = tokenizer_path.read_bytes()
    if len(raw_tokenizer) > 16_777_216:
        raise ValueError("Tokenizer exceeds the local byte limit")
    tokenizer = validate_tokenizer(Tokenizer.from_str(raw_tokenizer.decode("utf-8")), strict=True)
    candidates = generate(seed, episodes_per_skill)
    provenance = None
    if collision_audit_path is not None:
        audit_path = Path(collision_audit_path)
        if audit_path.resolve() in {path.resolve() for path in (output, manifest)}:
            raise ValueError("The provenance file must be distinct from outputs")
        audit_raw = audit_path.read_bytes()
        if len(audit_raw) > 1_048_576:
            raise ValueError("Collision provenance exceeds its byte limit")
        audit = json.loads(audit_raw.decode("utf-8"))
        if audit.get("purpose") != "data_provenance_collision_audit_no_model_outputs" or not isinstance(audit.get("collisions"), list):
            raise ValueError("Only prompt/metadata collision provenance is supported")
        occupied = defaultdict(set)
        for collision in audit["collisions"]:
            if not isinstance(collision.get("messages"), list) or not isinstance(collision.get("old"), list):
                raise ValueError("Collision provenance needs prompt messages and old split metadata")
            for old in collision["old"]:
                if old.get("split") not in SPLITS:
                    raise ValueError("Collision provenance has an invalid occupied split")
                occupied[canonical(collision["messages"])].add(old["split"])
        for record in candidates:
            old_folds = occupied.get(canonical(record["messages"][:-1]), set())
            if old_folds and old_folds != {record["split"]}:
                raise ValueError("A generated prompt conflicts with an occupied split in the supplied provenance")
        provenance = {"name": audit_path.name, "sha256": sha(audit_raw), "recordedCollisions": len(audit["collisions"]),
                      "scope": "Metadata and occupied authored prompts only; no evaluation responses, scores or weights",
                      "check": "Generated full prompts checked against every occupied split listed in this file"}
    prompt_folds, target_by_prompt, seen = {}, {}, set()
    selected, lengths, reasons = [], [], Counter()
    for record in candidates:
        prompt = canonical(record["messages"][:-1])
        target = record["messages"][-1]["content"]
        if prompt in prompt_folds and prompt_folds[prompt] != record["split"]:
            raise ValueError("Identical full prompt occurs across folds; review the opening banks")
        if prompt in target_by_prompt and target_by_prompt[prompt] != target:
            raise ValueError("An identical full prompt has conflicting authored targets")
        prompt_folds[prompt], target_by_prompt[prompt] = record["split"], target
        exact = canonical(record["messages"])
        if exact in seen:
            reasons["within_fold_duplicate_prefixes_removed"] += 1
            continue
        tokens, _ = dialogue_tokens(record["messages"], tokenizer)
        if len(tokens) > context_length+1:
            reasons["full_dialogue_exceeds_context"] += 1
            continue
        seen.add(exact)
        selected.append(record)
        lengths.append(len(tokens))
    skills, patterns = defaultdict(Counter), defaultdict(Counter)
    for record in selected:
        skills[record["source"]["skill"]][record["split"]] += 1
        patterns[record["source"]["templateId"]][record["split"]] += 1
    expected = {pattern["templateId"] for bank in templates().values() for pattern in bank}
    if set(patterns) != expected or any(set(folds) != set(SPLITS) for folds in patterns.values()):
        raise ValueError("Every pattern must remain represented in every fold")
    raw = b"".join(canonical(record)+b"\n" for record in selected)
    ordered_lengths = sorted(lengths)
    report = {"schemaVersion": 1, "dataset": "unidad-natural-basics-v2", "purpose": "auxiliary-training-curriculum",
              "sha256": sha(raw), "seed": seed, "records": len(selected),
              "splits": dict(Counter(record["split"] for record in selected)),
              "groupSplits": {split: len({record["groupId"] for record in selected if record["split"]==split}) for split in SPLITS},
              "skills": {skill: dict(folds) for skill,folds in sorted(skills.items())},
              "templates": {pattern: dict(folds) for pattern,folds in sorted(patterns.items())},
              "prefixPairs": dict(Counter(str(record["source"]["prefixPairs"]) for record in selected)),
              "coldPrototypeRecords": sum(record["source"]["prefixPairs"]==1 for record in selected),
              "fullDialogueTokens": {"min": min(lengths), "median": ordered_lengths[len(lengths)//2],
                                    "p95": ordered_lengths[math.ceil(.95*len(lengths))-1], "max": max(lengths)},
              "tokenizer": {"sha256": sha(raw_tokenizer), "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
              "authoring": {"skills": 5, "patternsPerSkill": 4, "episodesPerSkill": episodes_per_skill,
                            "coldOpeningBanks": "20 semantic patterns with a distinct equivalent opening per fold; no exact full prompt across folds",
                            "genericPrefixPolicy": "Retain first within-fold prototype; remove only subsequent identical prefixes",
                            "splitPolicy": "80/10/10 episode quotas at defaults; all paraphrases and prefixes share world fold",
                            "contextLength": context_length, "fullDialogueTokenLimit": context_length+1,
                            "dialoguesTruncated": False, "existingCorpusRead": False,
                            "collisionProvenanceRead": provenance is not None,
                            "heldOutProvenancePromptsRead": provenance is not None,
                            "evaluationResponsesRead": False, "evaluationScoresRead": False, "evaluationRun": False,
                            "modelResponsesUsed": False, "modelScoresUsed": False, "privateConversationUsed": False,
                            "oldCorpusCollisionAudit": "Listed prompt conflicts checked here; full merge re-audit required separately; never move old folds"},
              "collisionProvenance": provenance,
              "authoringRevision": {"previousDataset": "unidad-natural-basics-v1", "changedOpening": "unknown-name-introduced-corrected/train",
                                    "reason": "Previous generic train opening was already assigned to an old test family",
                                    "newOpening": "Dime cómo me llamo.", "oldCorporaModified": False},
              "manualReview": {"reviewer": "development-assistant-manual-review", "blinded": False,
                               "patternsReviewed": 20, "modelCompetenceAssessed": False,
                               "scope": "Authored role identity, unknown facts, current intent and declaration/correction semantics"},
              "reasons": dict(reasons), "generatorSha256": sha(Path(__file__).read_bytes()),
              "sharedAuthoringHelperSha256": sha(Path(__file__).with_name("build_open_fact_conversations.py").read_bytes()),
              "limitations": ["Procedural authored history is not the model's real previous replies.",
                              "Repeated cold prototypes become only one record per fold after dedup; the merge/trainer controls sampling.",
                              "Different opening phrases by fold test paraphrase transfer, not unconstrained conversation.",
                              "The collision file lists known conflicts; it does not establish the absence of all possible conflicts in old corpora.",
                              "No final/development evaluation result, external memory, tool execution or consciousness is claimed."]}
    exclusive_write(output, raw)
    load_corpus(output)
    exclusive_write(manifest, canonical(report)+b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--episodes-per-skill", type=int, default=100)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--collision-audit", type=Path, help="Metadatos y prompts ocupados; nunca respuestas o métricas de evaluación")
    args = parser.parse_args(argv)
    try:
        result = build(args.tokenizer,args.output,args.manifest,args.seed,args.episodes_per_skill,args.context_length,
                       collision_audit_path=args.collision_audit)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    main()
