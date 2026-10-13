"""Finite own-selector training from unchanged SOVENGI candidate generations.

Run collect, fit, then evaluate. Collection opens only TRAIN and VALIDATION;
the final evaluation opens TEST after verifying the frozen selector hash.
The oracle checks narrow literal compliance, not general conversation quality.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CORPUS = ROOT / "examples/sovengi-coordination-20261013"
RESULTS = ROOT / "docs/sovengi-coordination-20261013"
MODEL = ROOT / "models/coordination-reference-v1/model.json"
SEED = 20261013
ENTITIES = ("Brezunil", "Calmoria", "Dorsavel", "Ebrunata", "Felsoria", "Garnudel",
            "Hespirona", "Itramiel", "Jorazuma", "Keldavira", "Lumbrinel", "Mestoria",
            "Nerbalia", "Ostraviel", "Peldamira", "Qoriventa", "Reldunara", "Salmivora", "Tervandil", "Urdavena")
COLORS = ("azul", "verde", "roja", "amarilla", "blanca", "negra")
DAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
PROTOCOL = {"name": "literal-compliance-v1", "semanticGrader": False,
    "location": "Require whole words caja and supplied color; reject any other listed color.",
    "correction": "Require corrected weekday; reject old weekday.",
    "missing_name": "Require explicit name-related uncertainty or 'no lo sé'; reject name assertion markers.",
    "missingNamePatterns": [r"\bno (?:me )?has (?:dicho|dado) (?:el |su )?nombre\b",
        r"\bno (?:conozco|tengo) (?:el |su )nombre\b", r"\bno (?:se (?:el |su )nombre|lo se)\b"],
    "nameAssertionMarkers": [r"\bse llama\b", r"\b(?:el|su) nombre es\b"],
    "limitations": "Conservative literal checks can reject valid paraphrases and do not assess factual understanding or prose quality."}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    paths = [Path(__file__), ROOT / "src/ia_propia/conversation/selection.py",
        ROOT / "src/ia_propia/conversation/ensemble.py", ROOT / "src/ia_propia/sovengi/application.py",
        ROOT / "src/ia_propia/brain/models.py", ROOT / "configs/gnosis.json", ROOT / "configs/conversation-models.json"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def family_checks(manifest):
    families = {name: set(value["families"]) for name, value in manifest["partitions"].items()}
    if [len(families[name]) for name in ("train", "validation", "test")] != [12, 4, 4]:
        raise ValueError("Expected twelve/four/four complete entity families")
    if any(families[first] & families[second] for first, second in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Entity families overlap between partitions")
    command = ["rg", "-il", "--glob", "*.json*", "--glob", "*.py", "--glob", "*.md",
        "--glob", "!train_sovengi_coordination.py", "--glob", "!**/sovengi-coordination-20261013/**",
        "|".join(ENTITIES), "examples", "docs", "src/ia_propia/context"]
    checked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if checked.returncode not in (0, 1):
        raise ValueError("Prior-family text audit failed")
    collisions = checked.stdout.splitlines()
    if collisions:
        raise ValueError("New fictional families collide with prior project evidence: " + ", ".join(collisions))
    return {"entityFamiliesDisjoint": True, "priorProjectFamilyCollisionFiles": [],
            "auditScope": "previous examples/docs/context sources; new pipeline and results excluded"}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def norm(text):
    return "".join(char for char in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(char) != "Mn")


def utility(case, text):
    normalized = norm(text)
    words = set(re.findall(r"[^\W_]+", normalized))
    target = norm(case["expected"])
    if case["task"] == "location":
        return int("caja" in words and target in words and not any(norm(color) in words for color in COLORS if color != case["expected"]))
    if case["task"] == "correction":
        return int(target in words and norm(case["rejected"]) not in words)
    return int(any(re.search(pattern, normalized) for pattern in PROTOCOL["missingNamePatterns"])
               and not any(re.search(pattern, normalized) for pattern in PROTOCOL["nameAssertionMarkers"]))


def create_corpus():
    manifest_path = CORPUS / "corpus.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    partitions = {name: [] for name in ("train", "validation", "test")}
    for index, entity in enumerate(ENTITIES):
        partition = "train" if index < 12 else "validation" if index < 16 else "test"
        color, old, day = COLORS[index % len(COLORS)], DAYS[index % 7], DAYS[(index + 3) % 7]
        definitions = [("location", [f"La libreta {entity} está en caja {color}. ¿Dónde está la libreta {entity}?"], color, None),
            ("correction", [f"El taller {entity} abre el {old}.",
                f"Corrección: el taller {entity} abre el {day}. ¿Qué día abre el taller {entity}?"], day, old),
            ("missing_name", [f"No te he dado el nombre del farero de {entity}. ¿Cómo se llama?"], "explicit_uncertainty", None)]
        for task, messages, expected, rejected in definitions:
            partitions[partition].append({"id": f"coordination-v1/{entity.lower()}/{task}", "family": entity,
                "split": partition, "task": task, "messages": [{"role": "system", "content": "Eres SOVENGI. Responde en español."}]
                    + [{"role": "user", "content": text} for text in messages], "expected": expected, "rejected": rejected})
    descriptors = {}
    for name, rows in partitions.items():
        path = CORPUS / (name + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        path.write_bytes(raw)
        descriptors[name] = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "cases": len(rows), "families": list(dict.fromkeys(row["family"] for row in rows))}
    source_files = [Path(__file__), ROOT / "src/ia_propia/conversation/selection.py",
        ROOT / "src/ia_propia/conversation/ensemble.py", ROOT / "src/ia_propia/sovengi/application.py",
        ROOT / "src/ia_propia/brain/models.py", ROOT / "configs/gnosis.json", ROOT / "configs/conversation-models.json"]
    manifest = {"schemaVersion": 1, "frozenAt": now(), "totalCases": 60, "entityFamilies": 20,
        "source": "new authored fictional entity families; no personal conversation or prior evaluation cases",
        "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "priorIntegrationSourceCommit": "37177", "partitions": descriptors, "protocol": PROTOCOL,
        "protocolSha256": digest(PROTOCOL), "codeSha256": {str(path.relative_to(ROOT)): sha(path) for path in source_files},
        "generation": {"torchThreads": 1, "temperature": 0, "seed": SEED, "maxNewTokens": 24,
            "catalogModels": 35, "cohortSize": 4, "maxResident": 4, "criticCount": 2, "maxScoreCalls": 8},
        "training": {"epochs": 1000, "learningRate": .3, "l2": .001, "maxSeconds": 120}}
    write(manifest_path, manifest)
    return manifest


def rows_for(manifest, split):
    descriptor = manifest["partitions"][split]
    path = CORPUS / descriptor["file"]
    if sha(path) != descriptor["sha256"]:
        raise ValueError("Frozen corpus partition hash changed")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def infer(rows, *, selector=None, max_seconds=180):
    import torch
    from ia_propia.conversation.ensemble import ConversationEnsemble
    from ia_propia.conversation.runtime import ConversationEngine
    from ia_propia.conversation.selection import candidate_features
    from ia_propia.sovengi import Sovengi

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    recorded, prepared_inputs, output = [], [], []
    original_reply, original_ensemble = ConversationEngine.reply, ConversationEnsemble.reply
    def capture(engine, *args, **kwargs):
        result = original_reply(engine, *args, **kwargs)
        recorded.append({"modelId": engine.model_id, "messages": deepcopy(kwargs.get("messages", args[0] if args else None)),
            "reply": deepcopy(result), "textSha256": hashlib.sha256(result["text"].encode()).hexdigest(),
            "generatedTokenIdsSha256": digest(result["generatedTokenIds"])})
        return result
    def facade(engine, *args, **kwargs):
        prepared_inputs.append(deepcopy(kwargs.get("messages", args[0] if args else None)))
        return original_ensemble(engine, *args, **kwargs)
    started = time.perf_counter()
    app = None
    try:
        with tempfile.TemporaryDirectory(prefix="sovengi-coordination-") as temporary:
            config = json.loads((ROOT / "configs/gnosis.json").read_text())
            config["projectRoot"] = str(ROOT)
            config["brain"]["state_dir"] = str(Path(temporary) / "brain")
            config["brain"]["language_backend"] = "numpy"
            config["brain"].pop("conversation_selector", None)
            config["brain"]["conversation_ensemble"].update(max_candidates=4, min_candidates=2,
                max_resident=4, max_seconds=60, total_new_tokens=96, critic_count=2, max_score_calls=8)
            if selector is not None:
                config["brain"]["conversation_selector"] = str(selector)
            config_path = Path(temporary) / "config.json"
            write(config_path, config)
            app = Sovengi(config_path)
            with patch.object(ConversationEngine, "reply", capture), patch.object(ConversationEnsemble, "reply", facade):
                for index, case in enumerate(rows):
                    if time.perf_counter() - started >= max_seconds:
                        raise TimeoutError("Finite collection/evaluation deadline reached")
                    before = len(recorded)
                    session = app.connect(f"case-{index}", context_session_id=f"case-{index}")
                    response = session.request({"schemaVersion": 1, "operation": "conversation.reply", "input": {
                        "messages": deepcopy(case["messages"]), "max_new_tokens": 24, "temperature": 0, "seed": SEED}})
                    calls = deepcopy(recorded[before:])
                    row = {"id": case["id"], "family": case["family"], "task": case["task"], "split": case["split"],
                        "inputMessages": case["messages"], "preparedMessages": deepcopy(prepared_inputs[-1]) if prepared_inputs else None,
                        "rawResponse": response, "actualGenerations": calls, "candidates": []}
                    if response["status"] == "ok":
                        result = response["result"]
                        for metadata in result["ensemble"]["candidates"]:
                            matching = [call for call in calls if call["modelId"] == metadata["modelId"]
                                and call["textSha256"] == metadata["textSha256"]]
                            if len(matching) != 1:
                                raise ValueError("Candidate provenance does not match exactly one actual generation")
                            raw = matching[0]["reply"]
                            row["candidates"].append({**deepcopy(metadata), "text": raw["text"],
                                "generatedTokenIds": raw["generatedTokenIds"], "utility": utility(case, raw["text"])})
                        usable = [candidate for candidate in row["candidates"] if candidate["text"].strip()]
                        features = candidate_features(row["preparedMessages"], usable)
                        row["cohort"] = {"cohortId": case["id"], "features": features,
                            "utilities": [candidate["utility"] for candidate in usable]}
                        row["featureCandidateNames"] = [candidate["name"] for candidate in usable]
                        baseline_id = result["ensemble"].get("ruleSelectedModelId", result["ensemble"]["selectedModelId"])
                        baseline = next(candidate for candidate in row["candidates"] if candidate["modelId"] == baseline_id)
                        row.update(baselineUtility=baseline["utility"], actualWinnerUtility=utility(case, result["text"]),
                            oracleUtility=max(candidate["utility"] for candidate in usable),
                            exactOriginalWinner=any(result["text"] == call["reply"]["text"]
                                and result["generatedTokenIds"] == call["reply"]["generatedTokenIds"]
                                and result["modelId"] == call["modelId"] for call in calls))
                    output.append(row)
                    write(RESULTS / ("evaluation-progress.json" if selector else "collection-progress.json"),
                        {"updatedAt": now(), "selector": selector is not None, "elapsedSeconds": time.perf_counter() - started,
                         "completedCases": len(output), "cases": output})
                    print(json.dumps({"phase": "evaluate" if selector else "collect", "case": case["id"],
                        "completed": len(output), "status": response["status"], "baseline": row.get("baselineUtility"),
                        "winner": row.get("actualWinnerUtility"), "oracle": row.get("oracleUtility"),
                        "elapsedSeconds": time.perf_counter() - started}), flush=True)
        return {"cases": output, "elapsedSeconds": time.perf_counter() - started, "completedAt": now()}
    finally:
        if app is not None:
            app.close()
        torch.set_num_threads(previous_threads)


def summaries(records):
    return {split: {"cases": sum(row["split"] == split for row in records),
        "baselinePasses": sum(row.get("baselineUtility", 0) for row in records if row["split"] == split),
        "selectedPasses": sum(row.get("actualWinnerUtility", 0) for row in records if row["split"] == split),
        "oraclePasses": sum(row.get("oracleUtility", 0) for row in records if row["split"] == split),
        "technicalErrors": sum(row["rawResponse"]["status"] != "ok" for row in records if row["split"] == split)}
        for split in dict.fromkeys(row["split"] for row in records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("collect", "fit", "evaluate"), required=True)
    args = parser.parse_args(argv)
    manifest = create_corpus()
    separation = family_checks(manifest)
    collection_path, freeze_path = RESULTS / "collection.json", RESULTS / "selector-freeze.json"
    if args.phase == "collect":
        if collection_path.exists():
            raise ValueError("Collection is frozen; do not silently regenerate development predictions")
        rows = rows_for(manifest, "train") + rows_for(manifest, "validation")
        result = infer(rows, max_seconds=180)
        result.update(corpusSha256=sha(CORPUS / "corpus.json"), protocolSha256=manifest["protocolSha256"],
            partitionsOpened=["train", "validation"], testOpened=False, summary=summaries(result["cases"]), familyChecks=separation)
        write(collection_path, result)
    elif args.phase == "fit":
        from ia_propia.conversation.selection import LearnedSelector, save_selector, train_selector
        if freeze_path.exists() or MODEL.exists():
            raise ValueError("Selector already frozen; no training after a held-out run")
        collection = json.loads(collection_path.read_text())
        train = [row["cohort"] for row in collection["cases"] if row["split"] == "train" and "cohort" in row]
        validation = [row["cohort"] for row in collection["cases"] if row["split"] == "validation" and "cohort" in row]
        try:
            artifact = train_selector(train, validation, epochs=1000, learning_rate=.3, l2=.001,
                max_seconds=120, seed=SEED, provenance={"sourceCommit": manifest["sourceCommit"],
                    "corpusSha256": sha(CORPUS / "corpus.json"), "protocolSha256": manifest["protocolSha256"],
                    "collectionSha256": sha(collection_path), "personalData": False, "testOpened": False})
        except ValueError as error:
            write(RESULTS / "report.json", {"schemaVersion": 1, "status": "no_trainable_preference_signal",
                "reason": str(error), "corpus": manifest, "collection": collection,
                "transformerWeightsChanged": False, "selectorTrained": False, "testOpened": False,
                "defaultPromoted": False})
            raise
        save_selector(MODEL, artifact)
        predictor = LearnedSelector(artifact)
        for row in collection["cases"]:
            if "cohort" not in row:
                continue
            candidates = [candidate for candidate in row["candidates"] if candidate["text"].strip()]
            prediction = predictor.select(row["preparedMessages"], candidates)
            row["learnedPrediction"] = prediction
            row["actualWinnerUtility"] = next(candidate["utility"] for candidate in candidates
                if candidate["name"] == prediction["selectedName"])
        freeze = {"frozenAt": now(), "modelId": artifact["modelId"], "modelSha256": sha(MODEL),
            "collectionSha256": sha(collection_path), "corpusSha256": sha(CORPUS / "corpus.json"), "testOpened": False,
            "training": artifact["training"], "developmentSummary": summaries(collection["cases"]),
            "familyChecks": separation, "codeSnapshotHashes": {"collection": manifest["codeSha256"], "fit": code_hashes()}}
        write(freeze_path, freeze)
        write(RESULTS / "development-predictions.json", collection)
        print(json.dumps({"phase": "fit", "modelId": artifact["modelId"], "training": artifact["training"],
                          "summary": freeze["developmentSummary"]}), flush=True)
    else:
        if (RESULTS / "report.json").exists():
            raise ValueError("Held-out report exists; do not rerun TEST to tune the candidate")
        freeze = json.loads(freeze_path.read_text())
        if sha(MODEL) != freeze["modelSha256"] or sha(CORPUS / "corpus.json") != freeze["corpusSha256"]:
            raise ValueError("Frozen selector or corpus changed before TEST")
        evaluation_code = code_hashes()
        evaluated = infer(rows_for(manifest, "test"), selector=MODEL, max_seconds=180)
        learned_executed = all(row["rawResponse"].get("result", {}).get("ensemble", {}).get("learnedSelectorApplied", False)
                              for row in evaluated["cases"])
        report = {"schemaVersion": 1, "status": "heldout_completed", "corpus": manifest,
            "selectorFreeze": freeze, "artifact": json.loads(MODEL.read_text()),
            "collection": json.loads((RESULTS / "development-predictions.json").read_text()),
            "evaluation": evaluated, "testSummary": summaries(evaluated["cases"]),
            "learnedSelectorUsedInActualRuntime": learned_executed, "testUsedForTraining": False,
            "familyChecks": separation, "codeSnapshotHashes": {**freeze["codeSnapshotHashes"], "evaluation": evaluation_code},
            "transformerWeightsChanged": False, "defaultPromoted": False,
            "qualityScope": "predeclared conservative literal compliance on fictional tasks, not conversation acceptance"}
        write(RESULTS / "report.json", report)
        print(json.dumps({"phase": "evaluate", "summary": report["testSummary"], "runtimeLearned": learned_executed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
