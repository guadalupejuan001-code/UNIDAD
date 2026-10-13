"""Prepare a small synthetic contrastive curriculum; never run a model or train.

The existing canonical corpus and native last-assistant-reply encoder are reused.
Each fictional family has two contexts with identical current requests but
different correct answers. Development has separate names, values, families and
authored wording. This is an inspectable learning diagnostic, not a final test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata


DATASET = "unidad-conversation-pattern-curriculum-v1"
COMPETENCIES = ("role_binding", "explicit_correction", "known_vs_unknown",
                "negation", "current_request_format", "relevant_context")
POOLS = {
    "train": {
        "names": (("Mirel", "Norel"), ("Tavén", "Lurea"), ("Azmi", "Serén"),
                  ("Darela", "Ivorel"), ("Nemira", "Valén"), ("Orvia", "Zarel"),
                  ("Avelis", "Mirón"), ("Senia", "Urel"), ("Naviel", "Almira"),
                  ("Irela", "Durén"), ("Velmia", "Tasor"), ("Evara", "Numel")),
        "values": (("cuaderno violeta", "mapa gris"), ("sobre verde", "libro blanco"),
                   ("lunes a las nueve", "martes a las diez"), ("jueves a las once", "sábado a las doce"),
                   ("viernes a las ocho", "domingo a las siete"), ("miércoles a las seis", "lunes a las cinco"),
                   ("sendero azul", "sendero rojo"), ("puerta central", "puerta lateral"),
                   ("sopa tibia", "arroz suave"), ("té claro", "jugo dulce"),
                   ("carpeta beige", "cinta dorada"), ("bolsa ancha", "lápiz fino")),
    },
    "validation": {
        "names": (("Osvira", "Leimar"), ("Keluna", "Brisel"), ("Zumira", "Caelun"),
                  ("Tirven", "Maelia"), ("Belira", "Saelor"), ("Ulmera", "Dovian")),
        "values": (("caja ámbar", "lámina turquesa"), ("viernes a las trece", "sábado a las catorce"),
                   ("martes a las diecisiete", "jueves a las diecinueve"),
                   ("pasaje estrecho", "pasaje amplio"), ("pan tostado", "fruta fresca"),
                   ("estuche naranja", "botella celeste")),
    },
}


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def msg(role, content):
    return {"role": role, "content": content}


def templates(split, competency):
    """Different authored wording and ordering for the development fold."""
    if split == "train":
        return {
            "role_binding": ("{a} tiene {x}; {b} tiene {y}.", "¿Qué tiene {a}?", "¿Qué tiene {b}?"),
            "explicit_correction": ("La cita de {a} era {old}. El dato válido ahora es {new}.",
                                    "¿Cuándo es ahora la cita de {a}?", "Responde sólo con el horario vigente."),
            "known_vs_unknown": ("La visita de {b} será {y}. {known}",
                                  "¿Cuándo será la visita de {a}?", "¿Cuándo será la visita de {b}?"),
            "negation": ("{a} no usará {blocked}; usará {allowed}.", "¿Qué usará {a}?",
                         "¿Qué opción descartó {a}?"),
            "current_request_format": ("La elección de {a} es {x}; la de {b} es {y}.",
                                       "Ahora escribe únicamente la elección de {a}, sin explicación.",
                                       "Escribe la elección de {b} en el formato nombre: elección."),
            "relevant_context": ("El adorno es negro. La lluvia fue leve. {a} guardó {x}; {b} guardó {y}.",
                                  "¿Qué guardó {a}?", "¿Qué guardó {b}?"),
        }[competency]
    return {
        "role_binding": ("En este reparto, {y} corresponde a {b}, mientras que {x} corresponde a {a}.",
                         "Dime la pertenencia de {a}, sin confundir a las dos personas.",
                         "Devuelve sólo la pertenencia de {b} precedida por su nombre y dos puntos."),
        "explicit_correction": ("Habíamos anotado {old} para {a}. Corrijo esa anotación: debe quedar {new}.",
                                "Con la rectificación aplicada, ¿qué horario le corresponde a {a}?",
                                "Para cerrar, conserva el horario corregido en una única línea, sin el anterior."),
        "known_vs_unknown": ("Tengo confirmado para {b}: {y}. Sobre {a}, {known}",
                              "¿Disponemos del horario de {a}? Si lo sabes, dilo; si falta, señálalo.",
                              "En una sola línea, indica el horario confirmado para {b} con nombre: horario."),
        "negation": ("Se mantiene {allowed} como opción de {a}; queda excluido {blocked}.",
                     "¿Cuál es la opción que sigue en pie para {a}?",
                     "Indica sólo lo excluido, con el prefijo descartado: y sin elegirlo."),
        "current_request_format": ("{b} ha seleccionado {y}. Por su parte, {a} ha seleccionado {x}.",
                                   "Cambio el formato anterior: entrega sólo lo que eligió {a}.",
                                   "Ahora cambia de persona: una sola línea con {b} | su elección, sin otros datos."),
        "relevant_context": ("Se habló de una pared blanca y un paraguas grande. Los objetos guardados son {y} de {b} y {x} de {a}.",
                              "Ignora los adornos: ¿qué objeto guardó {a}?",
                              "Sin hablar del clima ni los adornos, usa nombre: objeto para lo que guardó {b}."),
    }[competency]


def build_cases():
    cases = []
    for split, family_repeats in (("train", 2), ("validation", 1)):
        for competency_index, competency in enumerate(COMPETENCIES):
            for family_number in range(family_repeats):
                pool_index = competency_index * family_repeats + family_number
                a, b = POOLS[split]["names"][pool_index]
                x, y = POOLS[split]["values"][pool_index]
                family = f"pattern-v1:{split}:{competency}:{family_number}"
                template = templates(split, competency)
                for branch in (0, 1):
                    chosen, other = ((x, y) if branch == 0 else (y, x))
                    fields = {"a": a, "b": b, "x": chosen, "y": other,
                              "old": x, "new": chosen, "blocked": other, "allowed": chosen}
                    wrong_indices = []
                    if competency == "known_vs_unknown":
                        fields["y"] = y
                        fields["known"] = ((f"La visita de {a} será {x}." if branch == 0 else
                                             f"No me han dicho la hora de {a}.") if split == "train" else
                                            (f"el horario quedó fijado en {x}." if branch == 0 else
                                             "todavía no me dieron ningún horario."))
                        answers = [x if branch == 0 else (f"No me diste la hora de {a}." if split == "train" else
                                                         f"Falta el horario de {a}."),
                                   y if split == "train" else f"{b}: {y}"]
                    elif competency == "explicit_correction":
                        # Both branches genuinely change the original value. The
                        # wrong assistant is context; no prefix ends in this error.
                        fields["old"] = "un horario pendiente" if split == "train" else "una hora provisional"
                        seed = [msg("user", f"La cita de {a} todavía no tiene horario confirmado."),
                                msg("assistant", f"La cita de {a} será {other}."),
                                msg("user", template[0].format(**fields)),
                                msg("assistant", "Tomo la corrección.")]
                        wrong_indices = [1]
                        answers = [f"La cita de {a} es {chosen}.", chosen]
                    elif competency == "negation":
                        answers = [chosen, other if split == "train" else f"descartado: {other}"]
                    elif competency == "current_request_format":
                        seed = [msg("user", template[0].format(**fields) +
                                    (" Explica ambas elecciones en un párrafo." if split == "train" else
                                     " Quería una lista larga con comentarios de las dos elecciones.")),
                                msg("assistant", "Puedo usar el formato que pidas.")]
                        answers = [chosen, f"{b}: {other}" if split == "train" else f"{b} | {other}"]
                    else:
                        answers = [chosen, other if split == "train" else f"{b}: {other}"]
                    if competency not in ("explicit_correction", "current_request_format"):
                        seed = [msg("user", template[0].format(**fields)), msg("assistant", "Anotado.")]
                    queries = [text.format(**fields) for text in template[1:]]
                    cases.append({"id": family + f":branch:{branch}", "groupId": family,
                                  "split": split, "competency": competency, "branch": branch,
                                  "contrastPairId": family, "phraseTemplateId": f"pattern-v1:{split}:{competency}",
                                  "phraseTemplate": list(template), "initialMessages": seed,
                                  "userTurns": queries, "expectedResponses": answers,
                                  "wrongHistoryAssistantMessageIndices": wrong_indices,
                                  "names": [a, b], "values": [x, y], "recordIds": []})
    return cases


def build_records(cases):
    records = []
    for case in cases:
        messages = list(case["initialMessages"])
        for turn, (query, answer) in enumerate(zip(case["userTurns"], case["expectedResponses"])):
            messages = [*messages, msg("user", query), msg("assistant", answer)]
            identity = case["id"] + f":turn:{turn}"
            case["recordIds"].append(identity)
            records.append({"id": identity, "groupId": case["groupId"], "split": case["split"],
                            "source": {"dataset": DATASET, "kind": "authored-synthetic", "license": "CC0-1.0",
                                       "method": "Deterministic authored fictional state transitions; no external model or runtime answers",
                                       "author": "Codex engineering agent; AI-authored, not human-reviewed",
                                       "competency": case["competency"], "scenarioId": case["id"],
                                       "branch": case["branch"], "turn": turn, "contrastPairId": case["contrastPairId"],
                                       "phraseTemplateId": case["phraseTemplateId"],
                                       "wrongHistoryAssistantMessageIndices": case["wrongHistoryAssistantMessageIndices"]},
                            "messages": list(messages)})
    return records


def audit_splits(cases, records):
    from ia_propia.conversation.data import validate_record
    for record in records:
        validate_record(record)
    for field in ("groupId", "id", "phraseTemplateId"):
        folds = [{c[field] for c in cases if c["split"] == split} for split in ("train", "validation")]
        if folds[0] & folds[1]:
            raise ValueError("Cross-fold leakage: " + field)
    for field in ("names", "values"):
        folds = [{normalize(v) for c in cases if c["split"] == split for v in c[field]}
                 for split in ("train", "validation")]
        if folds[0] & folds[1]:
            raise ValueError("Cross-fold normalized lexical overlap: " + field)
    template_folds = [{compact(c["phraseTemplate"]) for c in cases if c["split"] == split}
                      for split in ("train", "validation")]
    if template_folds[0] & template_folds[1]:
        raise ValueError("Cross-fold authored template overlap")
    exact_dialogues = {}
    for record in records:
        key = compact(record["messages"])
        if key in exact_dialogues:
            raise ValueError("Exact duplicate dialogue")
        exact_dialogues[key] = record["split"]
    pairs = {}
    for case in cases:
        pairs.setdefault(case["contrastPairId"], []).append(case)
    pair_receipts = []
    for identity, pair in pairs.items():
        if len(pair) != 2 or pair[0]["split"] != pair[1]["split"]:
            raise ValueError("Incomplete or cross-fold contrast pair")
        if pair[0]["userTurns"] != pair[1]["userTurns"]:
            raise ValueError("Contrast pair changed the current request")
        if pair[0]["initialMessages"] == pair[1]["initialMessages"]:
            raise ValueError("Contrast pair did not change real context")
        if pair[0]["expectedResponses"][0] == pair[1]["expectedResponses"][0]:
            raise ValueError("Contrast pair has a fixed first answer")
        pair_receipts.append({"id": identity, "split": pair[0]["split"], "sameCurrentRequests": True,
                              "differentInitialContexts": True, "differentFirstCorrectResponse": True})
    return {"schemaVersion": 1, "pairs": pair_receipts, "pairsCount": len(pairs),
            "wholeFamilyAndBranchesInOneFold": True, "templatesDisjointAcrossFolds": True,
            "normalizedNamesAndValuesDisjointAcrossFolds": True, "exactDuplicateDialogues": 0,
            "trainRecords": 48, "validationRecords": 24, "testRecords": 0,
            "limitation": "Both folds are small authored templates. Lexical and phrase separation alone does not establish broad natural-language generalization."}


def distribution(values):
    values = sorted(values)
    return {"count": len(values), "min": min(values), "median": values[len(values)//2], "max": max(values)}


def audit_supervision(records, tokenizer, context_length):
    from ia_propia.conversation.contract import IGNORE_INDEX
    from ia_propia.conversation.tokenizer import ASSISTANT_ID, EOS_ID, dialogue_tokens
    from ia_propia.conversation.training import _record_windows
    receipts = []
    for record in records:
        tokens, _ = dialogue_tokens(record["messages"], tokenizer)
        windows = list(_record_windows(record, tokenizer, context_length, "sliding_context", objective="assistant_last_reply"))
        if len(tokens) - 1 > context_length or len(windows) != 1:
            raise ValueError("Complete context must fit one native window: " + record["id"])
        inputs, labels = windows[0]
        final_role = max(i for i, token in enumerate(inputs) if token == ASSISTANT_ID)
        if any(value != IGNORE_INDEX for value in labels[:final_role]):
            raise ValueError("History got accidental positive supervision")
        expected = [*tokenizer.encode(record["messages"][-1]["content"], add_special_tokens=False).ids, EOS_ID]
        actual = [value for value in labels if value != IGNORE_INDEX]
        if actual != expected or len(inputs) != len(labels):
            raise ValueError("Native last reply target/EOS mismatch")
        wrong = record["source"]["wrongHistoryAssistantMessageIndices"]
        for index in wrong:
            if index >= len(record["messages"]) - 1 or record["messages"][index]["role"] != "assistant":
                raise ValueError("Wrong assistant is not exclusively history")
        receipts.append({"id": record["id"], "split": record["split"], "inputTokens": len(inputs),
                         "targetTokensIncludingEos": len(actual), "targetSha256": digest(compact(actual).encode()),
                         "wrongAssistantHistoryMessages": len(wrong), "allHistoryLabelsIgnored": True,
                         "fullContextRetained": True, "windows": 1})
    return {"schemaVersion": 1, "objective": "assistant_last_reply", "encoder": "ia_propia.conversation.training._record_windows",
            "windowPolicy": "sliding_context", "contextLength": context_length,
            "recordAudits": receipts, "historyPositiveTargets": 0,
            "wrongAssistantHistoryOccurrencesIgnored": sum(r["wrongAssistantHistoryMessages"] for r in receipts),
            "inputTokens": distribution([r["inputTokens"] for r in receipts]),
            "targetTokensIncludingEos": distribution([r["targetTokensIncludingEos"] for r in receipts]),
            "parametersTrained": False, "modelQualityMeasured": False}


def prepare(output, parent):
    from tokenizers import Tokenizer
    from ia_propia.conversation.tokenizer import validate_tokenizer
    output, parent = Path(output), Path(parent)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a new output directory; previous data is immutable")
    parent_manifest_path = parent / "manifest.json"
    parent_bytes = parent_manifest_path.read_bytes()
    parent_manifest = json.loads(parent_bytes)
    if not parent_manifest["modelId"].startswith("unidad-conversation-"):
        raise ValueError("Require a Unidad conversation artifact")
    lineage = parent_manifest.get("training", {}).get("lineage", {})
    if lineage.get("rootOrigin") != "random_weights":
        raise ValueError("This diagnostic requires verified own random-initialization lineage")
    tokenizer_path = parent / "tokenizer.json"
    tokenizer_raw = tokenizer_path.read_bytes()
    if digest(tokenizer_raw) != parent_manifest["files"]["tokenizer.json"]["sha256"]:
        raise ValueError("Parent tokenizer hash mismatch")
    tokenizer = validate_tokenizer(Tokenizer.from_str(tokenizer_raw.decode()), strict=True)
    context_length = parent_manifest["config"]["context_length"]
    cases = build_cases()
    records = build_records(cases)
    splits = audit_splits(cases, records)
    supervision = audit_supervision(records, tokenizer, context_length)
    if parent_manifest_path.read_bytes() != parent_bytes or tokenizer_path.read_bytes() != tokenizer_raw:
        raise ValueError("Parent changed while preparing data")
    corpus_raw = b"".join((compact(record) + "\n").encode("utf-8") for record in records)
    design = {"schemaVersion": 1, "dataset": DATASET,
              "hypothesis": "The current model can learn context-sensitive bindings and corrections when small contrastive families prevent fixed-answer shortcuts. Failure to overfit this control would motivate a training/capacity diagnosis before more broad data.",
              "intervention": "Small contrastive conversation data; unchanged own architecture, tokenizer, native last-reply loss and generation",
              "parentModelId": parent_manifest["modelId"], "parentManifestSha256": digest(parent_bytes),
              "successCriterionProposedBeforeTraining": {"selection": "One complete family per competency and fold; both contrast branches. Select the latest turn with distinct expected responses within each pair (--prefix-turn contrast), yielding 12 TRAIN and 12 VALIDATION turns; known_vs_unknown selects turn 0.",
                                                         "control": "At least 10 of 12 selected TRAIN turns fully correct with EOS; report both-branch discrimination. This measures learning/memorization only.",
                                                         "development": "At least 8 of 12 selected VALIDATION turns fully correct with EOS, improvement over the identical frozen parent, and two generated-history dialogues correct after the authored initial context. Report both-branch discrimination, zero technical failures, inference errors or dropped history turns.",
                                                         "acceptance": "No model promotion from this diagnostic. Preserve the separate new final criterion of 36/40, 8/10 and zero critical/technical failures."},
              "trainingStatus": "not_run_not_queued", "modelResponsesUsedToAuthorTargets": False,
              "externalModelsCalled": False, "reservedFinalRead": False,
              "oldCorporaModified": False, "parameterLearningNotPersistentMemory": True}
    cases_blob = {"schemaVersion": 1, "dataset": DATASET, "cases": cases,
                  "executionContract": "Free-history runs use only initialMessages and userTurns, appending actual raw model responses. expectedResponses are scoring evidence, never model inputs. Prefix/teacher-forced runs are a separately labeled diagnostic."}
    files = {"corpus.jsonl": corpus_raw,
             "development-cases.json": (json.dumps(cases_blob, ensure_ascii=False, indent=2) + "\n").encode(),
             "split-and-contrast-audit.json": (json.dumps(splits, ensure_ascii=False, indent=2) + "\n").encode(),
             "last-reply-audit.json": (json.dumps(supervision, ensure_ascii=False, indent=2) + "\n").encode(),
             "preregister.json": (json.dumps(design, ensure_ascii=False, indent=2) + "\n").encode(),
             "LICENSE-DATA.md": ("These AI-authored fictional examples are dedicated under CC0-1.0 to the extent the author can grant rights.\nThey contain no real personal facts and no external-model generations.\nhttps://creativecommons.org/publicdomain/zero/1.0/\n").encode(),
             "README.md": ("# Currículo pequeño de patrones conversacionales\n\n48 TRAIN y 24 VALIDATION, en 12/6 familias completas con dos contextos contrastantes y dos respuestas correctas por rama. Son ejemplos ficticios redactados por un agente de ingeniería de IA; no son datos humanos ni un catálogo del español. Se preparan para comprobar aprendizaje contextual, sin cambiar la inferencia ni responder con plantillas en ejecución.\n\nCubre atribución de hechos a personas, correcciones, conocimiento ausente, negación, formato de la petición vigente e información relevante entre distractores. La validación usa nombres, valores, familias y formulaciones separados y combina seguimiento de hechos con formatos. No constituye aceptación final ni acredita generalización amplia.\n\nEl entrenador nativo etiqueta únicamente la última respuesta correcta y EOS. Los asistentes erróneos de la familia de correcciones están exclusivamente en historia y todos sus labels son ignorados; cada prefijo positivo termina en una respuesta corregida. El manifiesto y las auditorías conservan las entradas y targets completos.\n\n`development-cases.json` separa contexto inicial, turnos de usuario y respuestas esperadas. En historia generada se añaden respuestas reales del modelo; los targets nunca se inyectan. La historia ideal de los prefijos es sólo un diagnóstico por separado.\n\nPreparación desde la raíz: `PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 examples/prepare_conversation_pattern_curriculum.py --parent RUTA_MODELO_PROPIO --output-dir RUTA_NUEVA`. No entrena, no encola, no toca datos anteriores y rechaza sobrescribir una entrega.\n").encode()}
    manifest = {"schemaVersion": 1, "dataset": DATASET, "provenance": {"kind": "authored-synthetic", "license": "CC0-1.0",
                 "fictional": True, "author": "Codex engineering agent", "humanReviewed": False,
                 "method": "Explicit deterministic authored contexts and state transitions; no external model calls, no mined private/evaluation data"},
                "parent": {"modelId": parent_manifest["modelId"], "manifestSha256": digest(parent_bytes),
                           "weightsSha256": parent_manifest["files"]["weights.safetensors"]["sha256"],
                           "rootOrigin": lineage["rootOrigin"]},
                "tokenizer": {"sha256": digest(tokenizer_raw), "vocabSize": tokenizer.get_vocab_size(), "refitted": False},
                "contextLength": context_length, "objective": "assistant_last_reply", "partitionPolicy": "train_validation_only",
                "file": {"path": "corpus.jsonl", "sha256": digest(corpus_raw), "bytes": len(corpus_raw)},
                "corpusSha256": digest(corpus_raw),
                "contentSha256": digest(b"".join((compact({"groupId": r["groupId"], "messages": r["messages"]}) + "\n").encode() for r in records)),
                "contentSha256Method": "SHA256 over ordered UTF-8 JSONL of {groupId,messages}, sorted keys, compact JSON, ensure_ascii=False, one trailing LF per record; excludes IDs, splits and source metadata",
                "splitCounts": dict(Counter(r["split"] for r in records)),
                "splits": dict(Counter(r["split"] for r in records)),
                "sources": {DATASET: len(records)}, "testUsedForTraining": False,
                "families": {s: len({r["groupId"] for r in records if r["split"] == s}) for s in ("train", "validation")},
                "byCompetency": {s: dict(Counter(r["source"]["competency"] for r in records if r["split"] == s)) for s in ("train", "validation")},
                "distinctFinalAnswers": {s: len({r["messages"][-1]["content"] for r in records if r["split"] == s}) for s in ("train", "validation")},
                "distinctPhraseTemplates": {s: len({c["phraseTemplateId"] for c in cases if c["split"] == s}) for s in ("train", "validation")},
                "inputTokens": supervision["inputTokens"], "targetTokensIncludingEos": supervision["targetTokensIncludingEos"],
                "totalTrainSupervisedTokensOnePass": sum(r["targetTokensIncludingEos"] for r in supervision["recordAudits"] if r["split"] == "train"),
                "weightsTrained": False, "reservedFinalRead": False,
                "files": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if sum(map(len, files.values())) > 4 * 1024 * 1024:
        raise ValueError("Small-curriculum output exceeds its 4 MiB bound")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    return {"outputDir": str(output), "corpusSha256": digest(corpus_raw),
            "manifestSha256": digest(files["manifest.json"]), "splits": manifest["splits"],
            "totalTrainSupervisedTokensOnePass": manifest["totalTrainSupervisedTokensOnePass"],
            "inputTokens": supervision["inputTokens"], "parametersTrained": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.output_dir, args.parent), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
