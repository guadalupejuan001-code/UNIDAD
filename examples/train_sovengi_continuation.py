"""Bounded continuation from a declared own-weight parent, with frozen evidence.

A new stage keeps learned weights and starts a new optimizer. --resume keeps
the optimizer and persistent budget of this SAME stage. Completed stages are
never silently restarted. Authored examples are teaching data, not runtime code.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading
import time

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.sovengi.application import SYSTEM_PROMPT
from train_sovengi_memory import evaluate, encoded, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "train": "Acacia Azalea Begonia Dalia Enebro Gardenia Hortensia Jacinto Lavanda Magnolia Mimosa Petunia Romero Salvia Tulipan Verbena".split(),
    "validation": "Amatista Berilo Granate Turquesa".split(),
    "development": "Nimbo Mirador Cometa Estela".split(),
}
DAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


def record(group, split, suffix, exchanges):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for question, answer in exchanges:
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return {"id": group + "/" + suffix, "groupId": group, "split": split,
            "source": {"dataset": "sovengi-dialogue-continuation-v1", "kind": "authored-synthetic",
                       "author": "Codex", "personalData": False, "modelResponsesUsed": False,
                       "description": "Fictional entity families; counterfactual known/absent facts and real-history evaluation."},
            "messages": messages}


def authored_rows():
    rows = []
    for split in ("train", "validation"):
        for index, name in enumerate(FAMILIES[split]):
            group = f"sovengi-dialogue-v1/{split}/{name}"
            entity = "librería " + name
            day, changed = DAYS[index % 7], DAYS[(index + 3) % 7]
            fact = f"La {entity} abre el {day}."
            rows.append(record(group, split, "unknown-day", [
                (f"¿Qué día abre la {entity}? No te he dicho el horario.", "No lo sé; no me has dicho el horario.")]))
            rows.append(record(group, split, "given-day", [
                (fact + f" ¿Qué día abre la {entity}?", fact)]))
            rows.append(record(group, split, "correction", [
                (fact + f" ¿Qué día abre la {entity}?", fact),
                (f"Corrección: ahora abre el {changed}. ¿Qué día abre?", f"La {entity} abre el {changed}.")]))
            rows.append(record(group, split, "unknown-place", [
                (f"¿Dónde está la {entity}? No te he dicho su ubicación.", "No lo sé; no me has dicho su ubicación.")]))
            rows.append(record(group, split, "given-place", [
                (f"La {entity} está en la plaza. ¿Dónde está?", f"La {entity} está en la plaza.")]))
            rows.append(record(group, split, "contrast", [
                (f"La {entity} abre el {day} y la cafetería del barrio abre el {changed}. ¿Qué día abre la {entity}?", fact)]))
            rows.append(record(group, split, "unknown-owner", [
                (f"¿De quién es la carpeta {name}? No te he dicho su dueño.", "No lo sé; no me has dicho su dueño.")]))
            rows.append(record(group, split, "after-absence", [
                (f"¿Qué día abre la {entity}? No te he dicho el horario.", "No lo sé; no me has dicho el horario."),
                (fact + " ¿Qué día abre entonces?", fact)]))

    identity = [
        "¿Cómo te llamas?", "¿Cuál es tu nombre?", "Dime cómo te llamas.",
        "¿Quién eres?", "Preséntate en pocas palabras.", "Quiero saber tu nombre.",
        "Hola, ¿quién eres?", "Buenas, ¿cómo te llamas?", "¿Me dices tu nombre?",
        "¿Cómo debería llamarte?", "¿Con quién estoy hablando?", "Dime tu nombre.",
    ]
    for i, question in enumerate(identity):
        rows.append(record("sovengi-dialogue-v1/train/identity", "train", str(i), [(question, "Soy SOVENGI.")]))
    names = "Adela Bruno Celia Dario Elisa Fabio Greta Hugo Ines Julio Karen Leo".split()
    for name in names:
        group = "sovengi-dialogue-v1/train/person-" + name
        first = (f"Hola, me llamo {name}. ¿Cómo te llamas?", f"Hola, {name}. Soy SOVENGI.")
        rows.append(record(group, "train", "greeting", [first]))
        rows.append(record(group, "train", "name", [first, ("¿Y cómo me llamo yo?", f"Te llamas {name}.")]))
        rows.append(record(group, "train", "missing-neighbor", [first,
            ("¿Cómo se llama mi vecino?", "No lo sé; no me has dicho su nombre.")]))
    simple = [
        [("Hola.", "Hola, ¿en qué te puedo ayudar?"), ("Quiero hablar un rato.", "Claro. ¿De qué te gustaría hablar?")],
        [("Buenas tardes.", "Buenas tardes. ¿En qué te puedo ayudar?"), ("No sé por dónde empezar.", "Cuéntame qué quieres conseguir y empezamos por ahí.")],
        [("Ayúdame con esto.", "Claro. ¿Con qué necesitas ayuda?"), ("Con un texto.", "Pásame el texto y dime qué quieres cambiar.")],
        [("Quiero aprender a dibujar.", "¿Qué te gustaría dibujar?"), ("Paisajes.", "Puedes empezar con un paisaje sencillo y practicar sus formas principales.")],
        [("No entendí lo que dijiste.", "¿Qué parte quieres que te explique?"), ("La última frase.", "Claro. Dime cuál es y te la explico con palabras sencillas.")],
        [("Gracias por tu ayuda.", "De nada. Me alegra ayudarte."), ("Hasta luego.", "Hasta luego.")],
    ]
    for i, dialogue in enumerate(simple):
        group = f"sovengi-dialogue-v1/train/social-{i}"
        for j in range(len(dialogue)):
            rows.append(record(group, "train", str(j), dialogue[:j + 1]))
    return rows


def development_probes():
    # Frozen before any candidate generation; neither prompts nor gold replies
    # are appended to the training corpus. These are development, not final TEST.
    dialogues = [
        ("identity", "Buenas, me dicen Iker. ¿Cómo te llamas?", "Hola, Iker. Soy SOVENGI.",
         "¿Recuerdas cómo me llamo?", "Te llamas Iker."),
        ("identity-alone", "Dime tu nombre, por favor.", "Soy SOVENGI.",
         "¿Cuál es mi nombre? Todavía no te lo he dicho.", "No lo sé; no me has dicho tu nombre."),
        ("absent-day", "¿Qué día abre la librería Nimbo? Todavía no te he dado el horario.", "No lo sé; no me has dicho el horario.",
         "Ahora te lo digo: abre el sábado. ¿Qué día abre la librería Nimbo?", "La librería Nimbo abre el sábado."),
        ("correction", "La librería Mirador abre el martes. ¿Cuándo abre?", "La librería Mirador abre el martes.",
         "Lo anterior estaba mal: abre el domingo. ¿Qué día abre ahora?", "La librería Mirador abre el domingo."),
        ("two-entities", "La librería Cometa abre el lunes y la librería Estela abre el viernes. ¿Qué día abre Estela?", "La librería Estela abre el viernes.",
         "¿Y Cometa?", "La librería Cometa abre el lunes."),
        ("absent-place", "¿Dónde queda la librería Nimbo? Aún no te he dicho dónde está.", "No lo sé; no me has dicho su ubicación.",
         "Está junto al puente. ¿Dónde queda la librería Nimbo?", "La librería Nimbo está junto al puente."),
        ("absent-person", "¿Sabes cómo se llama mi profesora? No te he dicho su nombre.", "No lo sé; no me has dicho su nombre.",
         "Mi profesora se llama Nuria. ¿Cómo se llama?", "Tu profesora se llama Nuria."),
        ("clarification", "Necesito una mano, pero aún no te digo con qué.", "Claro. ¿Con qué necesitas ayuda?",
         "Quiero corregir un párrafo.", "Pásame el párrafo y dime qué quieres cambiar."),
        ("social", "Hola, tengo ganas de conversar.", "Hola. ¿De qué te gustaría hablar?",
         "Estoy empezando a pintar.", "¿Qué te gustaría pintar?"),
        ("location-retention", "El museo Nimbo está al lado del río. ¿Dónde está el museo?", "El museo Nimbo está al lado del río.",
         "Me confundí: está junto al parque. ¿Dónde está entonces?", "El museo Nimbo está junto al parque."),
    ]
    return [{"id": "dialogue-development/" + key, "groupId": "dialogue-development/" + key,
             "kind": "fresh_dialogue", "note": None,
             "messages": [{"role": "user", "content": question}], "expected": answer,
             "followup": followup, "followupExpected": final}
            for key, question, answer, followup, final in dialogues]


def validate_isolation(rows, probes, originals):
    ids, folds = set(), {}
    for row in rows:
        validate_record(row)
        if row["split"] not in ("train", "validation"):
            raise ValueError("TEST cannot enter this stage")
        if row["id"] in ids or folds.setdefault(row["groupId"], row["split"]) != row["split"]:
            raise ValueError("Duplicate identity or cross-fold family")
        ids.add(row["id"])
    lookup = {row["id"]: row for row in rows}
    if any(lookup.get(row["id"]) != row for row in originals):
        raise ValueError("Previous records or splits were changed")
    questions = {m["content"] for row in rows for m in row["messages"] if m["role"] == "user"}
    for probe in probes:
        if probe["id"].startswith("dialogue-development/"):
            if probe["messages"][0]["content"] in questions or probe["followup"] in questions:
                raise ValueError("A held-out development prompt entered teaching data")
        if probe.get("groupId") in folds:
            raise ValueError("Development family entered training/validation")


def paths(config):
    name = config["experimentId"]
    if not isinstance(name, str) or not re.fullmatch(r"sovengi-[a-z0-9-]{1,80}", name):
        raise ValueError("Invalid experiment identity")
    return (ROOT / "examples" / name, ROOT / "docs" / name,
            ROOT / "models" / name / "candidate-best", ROOT / "private" / name / "run")


def prepare(config):
    data, report, candidate, run_dir = paths(config)
    for path_key, hash_key in (("retentionCorpus", "retentionSha256"), ("retentionProbes", "retentionProbesSha256")):
        if sha(ROOT / config[path_key]) != config[hash_key]:
            raise ValueError("Frozen input changed: " + path_key)
    originals = list(load_corpus(ROOT / config["retentionCorpus"]))
    rows = originals + authored_rows()
    probes = json.loads((ROOT / config["retentionProbes"]).read_text()) + development_probes()
    validate_isolation(rows, probes, originals)
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    files = {"corpus.jsonl": content, "development-cases.json": encoded(probes)}
    manifest = {"schemaVersion": 1, "corpusSha256": hashlib.sha256(content.encode()).hexdigest(),
                "splitCounts": dict(Counter(r["split"] for r in rows)), "retainedRows": len(originals),
                "retentionRecordsUnmodified": True, "testRecords": 0, "personalMessagesUsed": False,
                "heldOutEntityFamilies": FAMILIES["development"], "newDevelopmentTurns": 20,
                "developmentCasesSha256": hashlib.sha256(files["development-cases.json"].encode()).hexdigest(),
                "stageConfigSha256": hashlib.sha256(encoded(config).encode()).hexdigest(),
                "sourceCommit": config["sourceCommit"], "generatorSha256": sha(__file__),
                "finalAcceptanceTest": False}
    files["manifest.json"] = encoded(manifest)
    if data.exists():
        if any(not (data / name).is_file() or (data / name).read_text() != value for name, value in files.items()):
            raise ValueError("Frozen experiment differs; choose a new experimentId")
    else:
        data.mkdir(parents=True)
        for name, value in files.items():
            (data / name).write_text(value, encoding="utf-8")
    return data / "corpus.jsonl", probes, manifest


def run(config, *, resume=False, budget_seconds=None):
    import torch
    from ia_propia.conversation.training import train, export_best
    torch.set_num_threads(config["training"]["torch_threads"])
    data, report, candidate, run_dir = paths(config)
    parent = ROOT / config["initialModel"]
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    if parent_manifest["modelId"] != config["expectedModelId"] or sha(parent / "weights.safetensors") != config["expectedWeightsSha256"]:
        raise ValueError("Declared own-weight parent identity differs")
    if (report / "summary.json").exists():
        raise ValueError("Stage already evaluated; do not repeat it as new progress")
    if (run_dir / "status.json").exists() and not resume:
        raise ValueError("Stage already exists; use --resume without resetting its budget")
    corpus, probes, manifest = prepare(config)
    before_path = report / "before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())
        if before["modelId"] != parent_manifest["modelId"]:
            raise ValueError("Existing baseline belongs to another parent")
    else:
        before = evaluate(parent, probes)
        write_json(before_path, before)
    stopped = threading.Event()

    def progress():
        while not stopped.wait(30):
            path = run_dir / "status.json"
            if path.exists():
                state = json.loads(path.read_text())
                print(json.dumps({"event": "training_progress", **{k: state.get(k) for k in
                    ("status", "step", "elapsedSeconds", "reason", "bestValidationNll")}}), flush=True)

    observer = threading.Thread(target=progress, daemon=True)
    observer.start()
    try:
        status = train(corpus, run_dir, initial_model=None if resume else parent, resume=resume,
                       training_config=config["training"], budget_seconds=budget_seconds,
                       expected_corpus_sha256=manifest["corpusSha256"], expected_model_config=parent_manifest["config"])
    finally:
        stopped.set()
        observer.join(timeout=1)
    if status["status"] != "completed":
        print(encoded({"resumable": True, "status": status}), flush=True)
        return
    trained = export_best(run_dir, candidate)
    after = evaluate(candidate, probes)
    write_json(report / "after.json", after)
    selection = deepcopy(json.loads((ROOT / "configs/sovengi-memory-candidate.json").read_text()))
    selection["brain"].update(conversation_model=str(candidate.relative_to(ROOT)), expected_model_id=trained["modelId"])
    config_path = ROOT / "configs" / (config["experimentId"] + "-candidate.json")
    write_json(config_path, selection)
    summary = {"schemaVersion": 1, "parentModelId": parent_manifest["modelId"],
               "candidateModelId": trained["modelId"], "selectedStep": trained["training"]["selectedStep"],
               "trainingStatus": status, "newOptimizerStage": True, "resumedThisStage": resume,
               "stageConfig": config, "corpusManifest": manifest, "candidateConfig": str(config_path.relative_to(ROOT)),
               "automaticPromotion": False, "productionAccepted": False, "finalAcceptanceTest": False,
               "referenceWeightsUnchanged": sha(parent / "weights.safetensors") == config["expectedWeightsSha256"],
               "scores": {}}
    for label, predicate in (("retention", lambda x: x["kind"] != "fresh_dialogue"),
                             ("freshDialogue", lambda x: x["kind"] == "fresh_dialogue")):
        left = [x for x in before["outputs"] if predicate(x)]
        right = [x for x in after["outputs"] if predicate(x)]
        summary["scores"][label] = {"turns": len(left), "beforeExactTextAndEos": sum(x["exactTextAndEos"] for x in left),
                                    "afterExactTextAndEos": sum(x["exactTextAndEos"] for x in right)}
    for filename in ("status.json", "events.jsonl"):
        if (run_dir / filename).exists():
            report.mkdir(parents=True, exist_ok=True)
            (report / filename).write_bytes((run_dir / filename).read_bytes())
    write_json(report / "summary.json", summary)
    print(encoded({k: summary[k] for k in ("candidateModelId", "selectedStep", "scores", "referenceWeightsUnchanged")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sovengi-dialogue-stage.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.prepare_only:
        print(encoded(prepare(config)[2]))
    else:
        run(config, resume=args.resume, budget_seconds=args.budget_seconds)
