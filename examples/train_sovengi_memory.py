"""Finite own-weight experiment: retrieved facts, corrections and conversation retention.

Authored fictional examples are teaching data, never runtime answer templates.
The held-out development probes are not the project's final acceptance TEST.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import threading
import time

from ia_propia.conversation.data import load_corpus, validate_record
from ia_propia.memory_search import recall_snapshot
from ia_propia.sovengi.application import MEMORY_NOTICE, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/sovengi-memory-20261013"
REPORT = ROOT / "docs/sovengi-memory-20261013"
CANDIDATE = ROOT / "models/sovengi-memory-20261013/candidate-best"
RUN = ROOT / "private/sovengi-memory-20261013/run"
SEED = 20261013
DAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
NAMES = {
    "train": "Arce Abeto Cedro Olivo Pino Roble Fresno Sauce Haya Nogal Almendro Cerezo Peral Ciruelo Naranjo Limonero Laurel Pera Mora Lima Mango Cacao Café Nopal Agave Lino Trigo Centeno Avena Maíz Arroz Palma Bambú Junco Brezo Cardo Musgo Helecho Trébol Loto".split(),
    "validation": "Zafiro Ópalo Rubí Ágata Jaspe Topacio Jade Cuarzo".split(),
    "development": "Lucerna Candela Brújula Albor Marea Prisma Duna Caleta Astro Sendero".split(),
}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded(value), encoding="utf-8")


def memory_prompt(note, query, key):
    """Use the production retrieval/serialization contract with fictional IDs."""
    task_id = hashlib.sha256((key + "task").encode()).hexdigest()[:32]
    event_id = hashlib.sha256((key + "event").encode()).hexdigest()[:32]
    task = {"id": task_id, "eventId": event_id, "goal": note, "context": "",
            "createdAt": 0, "attempts": []}
    recalled = recall_snapshot({"tasks": {task_id: task}}, query)["tasks"]
    if not recalled:
        raise ValueError("Fictional note must be retrieved by the production policy")
    data = json.dumps(recalled, ensure_ascii=False).replace("<", "\\u003c")
    return SYSTEM_PROMPT + MEMORY_NOTICE + data


def authored_cases():
    rows, probes = [], []
    for split, names in NAMES.items():
        for index, name in enumerate(names):
            entity = "taller " + name
            group = f"sovengi-memory-v1/{split}/{name}"
            for branch in range(2):
                day = DAYS[(index + branch * 3) % len(DAYS)]
                previous = DAYS[(index + branch * 3 + 1) % len(DAYS)]
                note = f"El {entity} abre el {day}."
                questions = ([f"¿Qué día abre el {entity}?", f"¿Cuándo abre el {entity}?",
                              f"Recuérdame el día de apertura del {entity}."] if split == "train" else
                             [f"¿En qué día de la semana abre el {entity}?"])
                for variant, question in enumerate(questions):
                    key = f"{group}/{branch}/{variant}"
                    messages = [{"role": "system", "content": memory_prompt(note, question, key)},
                                {"role": "user", "content": question}]
                    if split == "development":
                        probes.append({"id": key, "groupId": group, "kind": "unseen_memory",
                            "note": note, "messages": messages[1:], "expected": note})
                    else:
                        rows.append({"id": key, "groupId": group, "split": split,
                            "source": {"dataset": "sovengi-fictional-memory-v1", "kind": "authored-synthetic",
                                       "author": "Codex", "personalData": False, "contrastBranch": branch},
                            "messages": messages + [{"role": "assistant", "content": note}]})
                if split != "development":
                    # All variants, facts and corrections for an entity stay in one fold.
                    rows.append({"id": f"{group}/{branch}/correction", "groupId": group, "split": split,
                        "source": {"dataset": "sovengi-fictional-memory-v1", "kind": "authored-synthetic",
                                   "author": "Codex", "personalData": False, "contrastBranch": branch},
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"El {entity} abre el {previous}."},
                            {"role": "assistant", "content": f"Entendido, el {entity} abre el {previous}."},
                            {"role": "user", "content": f"Me equivoqué: abre el {day}, no el {previous}. ¿Qué día abre?"},
                            {"role": "assistant", "content": note}]})
    return rows, probes


def retention_rows(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("The frozen conversation retention corpus has changed")
    groups = defaultdict(list)
    for record in load_corpus(path):
        if record["split"] not in ("train", "validation"):
            raise ValueError("No TEST data can enter this experiment")
        groups[(record["split"], record["groupId"])].append(record)
    selected, counts = [], Counter()
    for (split, group), records in sorted(groups.items(), key=lambda item: hashlib.sha256(item[0][1].encode()).hexdigest()):
        target = 160 if split == "train" else 24
        # Keep whole conversations and their original source, identity and fold.
        if counts[split] < target and len(records) <= 32:
            selected.extend(records)
            counts[split] += len(records)
    if counts["train"] < 160 or counts["validation"] < 24:
        raise ValueError("Insufficient complete retention groups")
    return selected


def prepare(config):
    if DATA.exists():
        raise ValueError("Use a clean experiment checkout; do not rewrite a frozen corpus")
    rows, probes = authored_cases()
    retention = retention_rows(ROOT / config["retentionCorpus"], config["retentionSha256"])
    rows.extend(retention)
    ids, groups = set(), {}
    for row in rows:
        validate_record(row)
        if row["id"] in ids or groups.get(row["groupId"], row["split"]) != row["split"]:
            raise ValueError("Duplicate identity or cross-fold conversation")
        ids.add(row["id"])
        groups[row["groupId"]] = row["split"]
    probes.extend([
        {"id": "retention/location", "kind": "unseen_live_fact", "note": None,
         "messages": [{"role": "user", "content": "El faro Niebla está en Puerto Coral. ¿Dónde está el faro Niebla?"}],
         "expected": "El faro Niebla está en Puerto Coral."},
        {"id": "retention/owner", "kind": "unseen_live_fact", "note": None,
         "messages": [{"role": "user", "content": "La libreta violeta es de Teresa. ¿De quién es la libreta violeta?"}],
         "expected": "La libreta violeta es de Teresa."},
        {"id": "retention/correction", "kind": "actual_history", "note": None,
         "messages": [{"role": "user", "content": "La cita es el lunes. ¿Qué día es la cita?"}],
         "expected": "La cita es el lunes.",
         "followup": "Cambio de planes: ahora la cita es el viernes. ¿Qué día es?",
         "followupExpected": "La cita es el viernes."},
        {"id": "retention/unknown", "kind": "missing_fact", "note": None,
         "messages": [{"role": "user", "content": "¿Cómo se llama mi vecino? No te he dicho su nombre."}],
         "expected": "No lo sé; no me has dicho su nombre."},
    ])
    DATA.mkdir(parents=True)
    corpus = DATA / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
    write_json(DATA / "development-cases.json", probes)
    manifest = {"schemaVersion": 1, "corpusSha256": sha(corpus),
        "splitCounts": dict(Counter(row["split"] for row in rows)), "testRecords": 0,
        "groupCounts": dict(Counter(groups.values())), "authoredBy": "Codex; fictional teaching examples",
        "retentionRows": len(retention), "retentionSource": config["retentionCorpus"],
        "retentionSourceSha256": config["retentionSha256"], "retentionRecordsUnmodified": True,
        "heldOutEntityFamilies": NAMES["development"], "developmentCasesSha256": sha(DATA / "development-cases.json"),
        "personalMessagesUsed": False, "finalAcceptanceTest": False,
        "sourceCommit": config["sourceCommit"], "generatorSha256": sha(__file__)}
    write_json(DATA / "manifest.json", manifest)
    return corpus, probes, manifest


def evaluate(model_path, probes):
    from ia_propia.sovengi import Sovengi
    from ia_propia.sovengi.__main__ import _request
    manifest = json.loads((model_path / "manifest.json").read_text())
    config = json.loads((ROOT / "configs/gnosis.json").read_text())
    config["projectRoot"] = str(ROOT)
    config["brain"].update(conversation_model=str(model_path), expected_model_id=manifest["modelId"])
    outputs = []
    with tempfile.TemporaryDirectory() as temporary:
        config["brain"]["state_dir"] = str(Path(temporary) / "brain")
        config_path = Path(temporary) / "config.json"
        write_json(config_path, config)
        app = Sovengi(config_path)
        try:
            for index, case in enumerate(probes):
                session = app.connect("case" + str(index))
                if case["note"]:
                    _request(session, "memory.task.add", {"goal": case["note"]})
                messages = deepcopy(case["messages"])
                expected = [case["expected"]]
                if "followup" in case:
                    expected.append(case["followupExpected"])
                for turn, target in enumerate(expected):
                    result = _request(session, "conversation.reply", {"messages": messages,
                        "max_new_tokens": 64, "temperature": 0, "seed": 42})
                    outputs.append({"caseId": case["id"], "kind": case["kind"], "turn": turn,
                        "inputMessages": deepcopy(messages), "savedFictionalNote": case["note"],
                        "expected": target, "reply": result,
                        "exactTextAndEos": (result["text"].strip() == target and result["stopReason"] == "eos"
                                            and result["discardedPromptTokens"] == 0)})
                    print(json.dumps({"event": "development_reply", "modelId": manifest["modelId"],
                        "caseId": case["id"], "turn": turn, "text": result["text"]}, ensure_ascii=False), flush=True)
                    if turn == 0 and "followup" in case:
                        if not result["text"].strip():
                            raise ValueError("Cannot insert an invented assistant reply into history")
                        messages += [{"role": "assistant", "content": result["text"]},
                                     {"role": "user", "content": case["followup"]}]
        finally:
            app.close()
    return {"modelId": manifest["modelId"], "outputs": outputs,
            "exactTextAndEos": sum(x["exactTextAndEos"] for x in outputs), "turns": len(outputs),
            "metricScope": "strict wording diagnostic; inspect raw replies for semantic correctness",
            "finalAcceptanceTest": False}


def run(config):
    import torch
    from ia_propia.conversation.training import train, export_best
    torch.set_num_threads(config["training"]["torch_threads"])
    parent = ROOT / config["initialModel"]
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["modelId"] != config["expectedModelId"] or sha(parent / "weights.safetensors") != config["expectedWeightsSha256"]:
        raise ValueError("Refusing to train a different parent")
    corpus, probes, data_manifest = prepare(config)
    before = evaluate(parent, probes)
    write_json(REPORT / "before.json", before)
    stopped = threading.Event()

    def progress():
        while not stopped.wait(30):
            path = RUN / "status.json"
            if path.exists():
                state = json.loads(path.read_text())
                print(json.dumps({"event": "training_progress", "state": state}), flush=True)

    observer = threading.Thread(target=progress, daemon=True)
    observer.start()
    started = time.monotonic()
    try:
        status = train(corpus, RUN, initial_model=parent, training_config=config["training"],
                       expected_corpus_sha256=data_manifest["corpusSha256"],
                       expected_model_config=manifest["config"])
    finally:
        stopped.set()
        observer.join(timeout=1)
    trained = export_best(RUN, CANDIDATE)
    after = evaluate(CANDIDATE, probes)
    write_json(REPORT / "after.json", after)
    summary = {"schemaVersion": 1, "parentModelId": manifest["modelId"],
        "candidateModelId": trained["modelId"], "trainingStatus": status,
        "selectedStep": trained["training"]["selectedStep"], "trainingAndEvaluationSeconds": time.monotonic() - started,
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "beforeExactTextAndEos": before["exactTextAndEos"], "afterExactTextAndEos": after["exactTextAndEos"],
        "developmentTurns": after["turns"], "corpusSha256": data_manifest["corpusSha256"],
        "sourceCommit": config["sourceCommit"], "newOptimizerStage": True, "exactResume": False,
        "automaticPromotion": False, "productionAccepted": False, "finalAcceptanceTest": False,
        "referenceWeightsUnchanged": sha(parent / "weights.safetensors") == config["expectedWeightsSha256"]}
    write_json(REPORT / "summary.json", summary)
    for filename in ("status.json", "events.jsonl"):
        if (RUN / filename).exists():
            (REPORT / filename).write_bytes((RUN / filename).read_bytes())
    print(encoded(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sovengi-memory-stage.json")
    args = parser.parse_args()
    run(json.loads((ROOT / args.config).read_text()))
