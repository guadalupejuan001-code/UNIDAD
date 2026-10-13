"""Reproducible, bounded SGD for the independent context relevance model.

The authored fictional corpus measures retrieval of supplied text. It contains
no user conversation, imported weights, generated answers or language training.
Whole entity families are held apart; test is read only after weight selection.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .model import ContextRanker, FEATURE_NAMES, features, sigmoid

CORPUS_VERSION = "context-fictional-retrieval-v1"
FAMILIES = {
    "train": ("Amapola", "Brisa", "Cedro", "Duna", "Estrella", "Faro", "Granito", "Horizonte", "Isla", "Jade", "Kiwi", "Lirio"),
    "validation": ("Marea", "Nube", "Olivo", "Prado"),
    "test": ("Quilla", "Risco", "Sauce", "Trebol"),
}


def fictional_corpus():
    records = []
    for split, entities in FAMILIES.items():
        for entity in entities:
            sources = [
                ("goal", "user", f"Mi objetivo es organizar una visita al taller {entity}."),
                ("constraint", "user", f"La condición es no publicar datos personales del taller {entity}."),
                ("fact", "user", f"El taller {entity} abre el jueves."),
                ("fact", "user", f"La ubicación del taller {entity} está junto a la plaza."),
                ("fact", "user", f"La biblioteca Alterna{entity} abre el martes."),
                ("correction", "user", f"Corrijo la apertura del taller {entity}: abre el lunes."),
                ("utterance", "assistant", f"El taller {entity} abre el jueves."),
                ("topic", "user", "Ahora hablamos de una canción que escuché ayer."),
            ]
            cases = [
                ("opening", f"¿Qué día abre el taller {entity}?", "", 5),
                ("other-entity", f"¿Qué día abre la biblioteca Alterna{entity}?", "", 4),
                ("short-followup", "¿Y eso?", f"Revisemos el día de apertura del taller {entity}.", 5),
                ("goal", f"¿Cuál es el objetivo del taller {entity}?", "", 0),
                ("constraint", f"¿Cuál es la condición del taller {entity}?", "", 1),
                ("location", f"¿Cuál es la ubicación del taller {entity}?", "", 3),
                ("goal-followup", "¿Qué objetivo teníamos?", f"Queríamos organizar una visita al taller {entity}.", 0),
                ("constraint-followup", "¿Y el límite?", f"Hablábamos de publicar datos personales del taller {entity}.", 1),
                ("question-history", f"¿Qué día abre el taller {entity}?", "", 5),
                ("assistant-confusion", f"¿Cuál es la ubicación del taller {entity}?", "", 3),
                ("distant-goal", f"Retomemos el objetivo de organizar la visita al taller {entity}.", "", 0),
                ("correction-paraphrase", f"Dime la apertura del taller {entity}.", "", 5),
            ]
            for name, query, thread, target in cases:
                candidates = [{"id": f"{entity}-{index}", "text": text, "role": role,
                               "kind": kind, "turn_index": index}
                              for index, (kind, role, text) in enumerate(sources)]
                if name == "question-history":
                    candidates[7] = {"id": f"{entity}-7", "role": "user", "kind": "utterance", "turn_index": 7,
                                     "text": f"¿Qué día abre el taller {entity}?"}
                elif name == "assistant-confusion":
                    candidates[6]["text"] = f"La ubicación del taller {entity} está junto a la plaza."
                elif name == "distant-goal":
                    candidates[7]["text"] = "Mi objetivo nuevo para otra tarea es organizar una fiesta."
                    candidates[7]["kind"] = "goal"
                records.append({"id": f"{entity}-{name}", "groupId": f"entity-{entity}", "split": split,
                                "source": {"dataset": CORPUS_VERSION, "kind": "authored-fictional-by-Codex",
                                           "scope": "supplied source ranking; repeated templates with held-out entity families"},
                                "query": query, "thread": thread, "candidates": candidates,
                                "relevantIds": [candidates[target]["id"]]})
    return records


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def corpus_sha256(records):
    return hashlib.sha256(_canonical(records)).hexdigest()


def validate_corpus(records):
    if type(records) is not list or not records or len(records) > 10000:
        raise ValueError("A bounded nonempty context corpus is required")
    ids, groups = set(), {}
    ranker = ContextRanker([0.] * len(FEATURE_NAMES), 0.)
    for record in records:
        if (type(record) is not dict or record.get("split") not in FAMILIES
                or type(record.get("id")) is not str or not record["id"]
                or type(record.get("groupId")) is not str or not record["groupId"]):
            raise ValueError("Context corpus requires explicit record IDs, families and partitions")
        if record["id"] in ids or (record["groupId"] in groups and groups[record["groupId"]] != record["split"]):
            raise ValueError("Duplicate context record or family partition leakage")
        ids.add(record["id"])
        groups[record["groupId"]] = record["split"]
        ranker.score_candidates(record["query"], record["candidates"], thread=record.get("thread", ""))
        candidate_ids = [candidate.get("id") for candidate in record["candidates"]]
        relevant = record.get("relevantIds")
        if (any(type(value) is not str or not value for value in candidate_ids)
                or len(set(candidate_ids)) != len(candidate_ids) or type(relevant) is not list or not relevant
                or len(set(relevant)) != len(relevant) or not set(relevant).issubset(candidate_ids)):
            raise ValueError("Context labels must reference distinct supplied sources")
        source = record.get("source")
        if type(source) is not dict or not source.get("dataset") or not source.get("kind"):
            raise ValueError("Context training requires explicit provenance")
    if set(groups.values()) != set(FAMILIES):
        raise ValueError("Train, validation and held-out test families are all required")


def _examples(records, split):
    result = []
    for record in records:
        if record["split"] != split:
            continue
        maximum = max(candidate.get("turn_index", 0) for candidate in record["candidates"])
        for candidate in record["candidates"]:
            result.append((features(record["query"], candidate, thread=record.get("thread", ""), max_turn=maximum),
                           float(candidate["id"] in record["relevantIds"])))
    return result


def _loss(weights, bias, examples):
    total = 0.
    for values, label in examples:
        logit = bias + sum(weight * value for weight, value in zip(weights, values))
        total += max(logit, 0) - label * logit + math.log1p(math.exp(-abs(logit)))
    return total / len(examples)


def evaluate(ranker, records, split):
    """Measure source-ranking accuracy, without generating conversational answers."""
    learned_correct, baseline_correct, reciprocal, baseline_reciprocal = 0, 0, 0., 0.
    learned_recall, baseline_recall, outputs = 0., 0., []
    for record in records:
        if record["split"] != split:
            continue
        scored = ranker.score_candidates(record["query"], record["candidates"], thread=record.get("thread", ""))
        learned = sorted(scored, key=lambda candidate: -candidate["score"])
        baseline = sorted(scored, key=lambda candidate: -(0.85 * candidate["featureValues"]["query_coverage"]
                                                        + 0.15 * candidate["featureValues"]["recency"]))
        relevant = set(record["relevantIds"])
        learned_ids, baseline_ids = [item["id"] for item in learned], [item["id"] for item in baseline]
        learned_correct += learned_ids[0] in relevant
        baseline_correct += baseline_ids[0] in relevant
        reciprocal += 1 / next(index for index, value in enumerate(learned_ids, 1) if value in relevant)
        baseline_reciprocal += 1 / next(index for index, value in enumerate(baseline_ids, 1) if value in relevant)
        learned_recall += len(relevant & set(learned_ids[:3])) / len(relevant)
        baseline_recall += len(relevant & set(baseline_ids[:3])) / len(relevant)
        outputs.append({"id": record["id"], "relevantIds": sorted(relevant), "learnedIds": learned_ids,
                        "baselineIds": baseline_ids, "scores": {item["id"]: item["score"] for item in scored}})
    if not outputs:
        raise ValueError("The requested context evaluation split is empty")
    count = len(outputs)
    return {"split": split, "cases": count,
            "learned": {"top1Correct": learned_correct, "top1Accuracy": learned_correct / count,
                        "meanReciprocalRank": reciprocal / count, "recallAt3": learned_recall / count},
            "baseline": {"method": "0.85*query_word_coverage+0.15*recency",
                         "top1Correct": baseline_correct, "top1Accuracy": baseline_correct / count,
                         "meanReciprocalRank": baseline_reciprocal / count, "recallAt3": baseline_recall / count},
            "outputs": outputs, "measures": "retrieval of labeled supplied source; not language understanding or conversation"}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


def train_ranker(records, *, epochs=2000, seed=20261013, learning_rate=0.5, l2=0.001,
                 checkpoint=None, resume=False):
    """Full-batch SGD from own random weights, selected only by validation loss.

    Repeated calls with an unchanged corpus/configuration resume exactly at an
    epoch boundary. There is no optimizer momentum or post-initialization RNG.
    """
    validate_corpus(records)
    if type(epochs) is not int or not 1 <= epochs <= 5000 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Context training requires bounded epochs and seed")
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1
            or type(l2) not in (int, float) or not math.isfinite(l2) or not 0 <= l2 <= 1 or type(resume) is not bool):
        raise ValueError("Context training requires bounded learning rate, L2 and boolean resume")
    config = {"seed": seed, "learningRate": learning_rate, "l2": l2,
              "corpusSha256": corpus_sha256(records), "featureNames": list(FEATURE_NAMES)}
    train = _examples(records, "train")
    validation = _examples(records, "validation")
    started = time.monotonic()
    if resume:
        if checkpoint is None:
            raise ValueError("Resume requires a checkpoint path")
        raw = Path(checkpoint).read_bytes()
        if len(raw) > 1048576:
            raise ValueError("Context checkpoint exceeds its byte limit")
        saved = json.loads(raw)
        digest = hashlib.sha256(_canonical({key: value for key, value in saved.items()
                                            if key != "checkpointSha256"})).hexdigest()
        if saved.get("checkpointSha256") != digest or saved.get("schemaVersion") != 1:
            raise ValueError("Context checkpoint integrity mismatch")
        if saved.get("config") != config:
            raise ValueError("Context checkpoint configuration or corpus changed")
        weights, bias, first_epoch = saved["weights"], saved["bias"], saved["epoch"]
        ContextRanker(weights, bias)
        best, history = saved["best"], saved["history"]
        prior_seconds = saved["trainingSeconds"]
        ContextRanker(best["weights"], best["bias"])
        if (type(first_epoch) is not int or not 0 <= first_epoch <= 5000
                or type(best["epoch"]) is not int or not 0 <= best["epoch"] <= first_epoch
                or type(best["validationLoss"]) not in (int, float) or not math.isfinite(best["validationLoss"])
                or best["validationLoss"] < 0 or type(history) is not list or len(history) > 201
                or type(prior_seconds) not in (int, float) or not math.isfinite(prior_seconds) or prior_seconds < 0):
            raise ValueError("Invalid context checkpoint state")
    else:
        rng = random.Random(seed)
        weights = [rng.gauss(0, 0.02) for _ in FEATURE_NAMES]
        bias, first_epoch = rng.gauss(0, 0.02), 0
        best = {"weights": list(weights), "bias": bias, "epoch": 0,
                "validationLoss": _loss(weights, bias, validation)}
        history, prior_seconds = [], 0.
    initial_rng = random.Random(seed)
    initial = [initial_rng.gauss(0, 0.02) for _ in FEATURE_NAMES]
    initial_loss = _loss(initial, initial_rng.gauss(0, 0.02), train)
    segment_initial_loss = _loss(weights, bias, train)
    for epoch in range(first_epoch + 1, epochs + 1):
        gradient, bias_gradient = [0.] * len(FEATURE_NAMES), 0.
        for values, label in train:
            probability = sigmoid(bias + sum(weight * value for weight, value in zip(weights, values)))
            error = probability - label
            bias_gradient += error
            for index, value in enumerate(values):
                gradient[index] += error * value
        weights = [weight - learning_rate * (value / len(train) + l2 * weight)
                   for weight, value in zip(weights, gradient)]
        bias -= learning_rate * bias_gradient / len(train)
        loss = _loss(weights, bias, validation)
        if loss < best["validationLoss"]:
            best = {"weights": list(weights), "bias": bias, "epoch": epoch, "validationLoss": loss}
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "trainLoss": _loss(weights, bias, train), "validationLoss": loss})
            if checkpoint is not None:
                state = {"schemaVersion": 1, "config": config, "weights": weights,
                    "bias": bias, "epoch": epoch, "best": best, "history": history,
                    "trainingSeconds": prior_seconds + time.monotonic() - started}
                state["checkpointSha256"] = hashlib.sha256(_canonical(state)).hexdigest()
                _atomic_json(checkpoint, state)
    metadata = {"training": "own random initialization; full-batch SGD; binary source relevance",
                "corpusVersion": CORPUS_VERSION, "provenance": "authored fictional by Codex",
                "corpusSha256": config["corpusSha256"], "seed": seed, "learningRate": learning_rate, "l2": l2,
                "selectedEpoch": best["epoch"], "completedEpochs": max(first_epoch, epochs),
                "selection": "minimum validation binary cross entropy; test excluded",
                "validationLoss": best["validationLoss"], "trainSources": len(train),
                "validationSources": len(validation), "limits": "lexical features and known markers; held-out entity templates; no semantic comprehension claim"}
    ranker = ContextRanker(best["weights"], best["bias"], metadata=metadata)
    return ranker, {"schemaVersion": 1, "config": config, "completedEpochs": max(first_epoch, epochs),
                    "selectedEpoch": best["epoch"], "trainingSeconds": prior_seconds + time.monotonic() - started,
                    "initialTrainLoss": initial_loss, "segmentInitialTrainLoss": segment_initial_loss,
                    "selectedTrainLoss": _loss(ranker.weights, ranker.bias, train),
                    "selectedValidationLoss": best["validationLoss"], "history": history}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train own independent context source ranker on declared fiction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    corpus_path, checkpoint = args.output / "corpus.json", args.output / "checkpoint.json"
    if args.resume:
        records = json.loads(corpus_path.read_bytes())
    else:
        if any(path.exists() for path in (corpus_path, checkpoint, args.output / "ranker.json")):
            raise ValueError("A new context training must use an unused output directory")
        records = fictional_corpus()
        _atomic_json(corpus_path, records)
    ranker, receipt = train_ranker(records, epochs=args.epochs, seed=args.seed, checkpoint=checkpoint, resume=args.resume)
    # Weights and selection are frozen before held-out test labels are inspected.
    artifact = ranker.artifact()
    _atomic_json(args.output / "ranker.json", artifact)
    receipt["modelId"] = artifact["modelId"]
    receipt["artifactSha256"] = artifact["artifactSha256"]
    receipt["partitionCounts"] = {split: sum(record["split"] == split for record in records) for split in FAMILIES}
    _atomic_json(args.output / "training.json", receipt)
    evaluation = {"schemaVersion": 1, "modelId": ranker.model_id,
                  "frozenBeforeTest": True, "validation": evaluate(ranker, records, "validation"),
                  "test": evaluate(ranker, records, "test")}
    _atomic_json(args.output / "evaluation.json", evaluation)
    print(json.dumps({"modelId": ranker.model_id, "parameters": ranker.parameter_count,
                      "trainingSeconds": receipt["trainingSeconds"], "test": {key: value for key, value in evaluation["test"].items()
                      if key != "outputs"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
