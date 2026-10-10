"""A tanh recurrent network with explicit BPTT, Adam and no pretrained parameters.

State is reset for every document chunk. Evaluation is next-character negative
log likelihood with exactly the same truncated context, not a comprehension test.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time

import numpy as np


class LanguageError(ValueError):
    """Invalid data, configuration, model, or numerical operation."""


MODEL_TYPE = "ia-propia-char-rnn"
UNKNOWN = "<UNK>"
WEIGHT_NAMES = ("Wxh", "Whh", "Why", "bh", "by")
ARTIFACT_KEYS = {"schemaVersion", "modelType", "modelId", "config", "tokenizer", "weights",
                 "training", "evaluation", "artifactSha256"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def integer(value, name, minimum=0, maximum=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise LanguageError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LanguageError(f"{name} must be a JSON number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise LanguageError(f"{name} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise LanguageError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def read_documents(corpus_path):
    raw = Path(corpus_path).read_bytes()
    if not raw or len(raw) > 2_000_000:
        raise LanguageError("Corpus must contain 1 to 2,000,000 bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LanguageError("Corpus must be UTF-8") from error
    # Explicit boundaries: no next-character target spans a separator or document.
    documents = [part.strip() for part in re.split(r"(?m)^\s*---\s*$", text)]
    if not 3 <= len(documents) <= 1000 or any(len(document) < 2 for document in documents):
        raise LanguageError("At least three nonempty documents with two characters each are required; separate them with a line ---")
    return documents, hashlib.sha256(raw).hexdigest()


def split_documents(documents, train_fraction=0.6, validation_fraction=0.2):
    train_fraction = number(train_fraction, "trainFraction", positive=True)
    validation_fraction = number(validation_fraction, "validationFraction", positive=True)
    if train_fraction + validation_fraction >= 1:
        raise LanguageError("Fractions must leave a nonempty test split")
    count = len(documents)
    if count < 3:
        raise LanguageError("Three documents are required for disjoint train/validation/test")
    train_count = max(1, min(count - 2, int(count * train_fraction)))
    validation_count = max(1, min(count - train_count - 1, int(count * validation_fraction)))
    indices = {"train": list(range(train_count)),
               "validation": list(range(train_count, train_count + validation_count)),
               "test": list(range(train_count + validation_count, count))}
    seen = {}
    for name, selected in indices.items():
        for index in selected:
            if documents[index] in seen and seen[documents[index]] != name:
                raise LanguageError("Identical documents cannot occur in different splits; deduplicate the corpus")
            seen[documents[index]] = name
    return {name: [documents[index] for index in selected] for name, selected in indices.items()}, indices


def tokenizer(train_documents):
    characters = sorted(set("".join(train_documents)))
    if len(characters) < 2 or len(characters) > 1023:
        raise LanguageError("Training vocabulary must contain 2 to 1,023 distinct characters")
    return {"kind": "train-only-character-v1", "unknownId": 0, "tokens": [UNKNOWN] + characters}


def encode(text, vocabulary):
    mapping = {character: index for index, character in enumerate(vocabulary["tokens"])}
    return np.asarray([mapping.get(character, 0) for character in text], dtype=np.int64)


def chunks(documents, vocabulary, sequence_length):
    result = []
    for document in documents:
        tokens = encode(document, vocabulary)
        for start in range(0, len(tokens) - 1, sequence_length):
            end = min(start + sequence_length + 1, len(tokens))
            result.append((tokens[start:end - 1], tokens[start + 1:end]))
    if not result:
        raise LanguageError("Every split must contain next-character targets")
    return result


def initialize(vocabulary_size, hidden_size, seed):
    rng = np.random.default_rng(seed)
    return {"Wxh": rng.normal(0, 0.08, (hidden_size, vocabulary_size)),
            "Whh": rng.normal(0, 1 / math.sqrt(hidden_size), (hidden_size, hidden_size)),
            "Why": rng.normal(0, 0.08, (vocabulary_size, hidden_size)),
            "bh": np.zeros(hidden_size, dtype=np.float64),
            "by": np.zeros(vocabulary_size, dtype=np.float64)}


def _log_softmax(logits):
    if not np.all(np.isfinite(logits)):
        raise LanguageError("Nonfinite language logits")
    shifted = logits - logits.max()
    return shifted - math.log(float(np.exp(shifted).sum()))


def loss_and_gradients(parameters, inputs, targets, *, gradients=True):
    """Mean next-character NLL; gradient covers the complete current chunk."""
    if len(inputs) == 0 or len(inputs) != len(targets):
        raise LanguageError("Inputs and targets need the same nonzero length")
    current = np.zeros_like(parameters["bh"])
    # Evaluation needs only the current state; retain history only for BPTT.
    hidden = [current] if gradients else None
    output_gradients = [] if gradients else None
    loss = 0.0
    for index, target in zip(inputs, targets):
        activation = parameters["Wxh"][:, index] + parameters["Whh"] @ current + parameters["bh"]
        if not np.all(np.isfinite(activation)):
            raise LanguageError("Nonfinite language hidden activation")
        current = np.tanh(activation)
        log_probabilities = _log_softmax(parameters["Why"] @ current + parameters["by"])
        loss -= float(log_probabilities[target])
        if gradients:
            hidden.append(current)
            derivative = np.exp(log_probabilities)
            derivative[target] -= 1.0
            output_gradients.append(derivative / len(inputs))
    loss /= len(inputs)
    if not math.isfinite(loss):
        raise LanguageError("Nonfinite language loss")
    if not gradients:
        return loss, None
    result = {name: np.zeros_like(value) for name, value in parameters.items()}
    next_hidden_gradient = np.zeros_like(parameters["bh"])
    for step in reversed(range(len(inputs))):
        derivative = output_gradients[step]
        result["Why"] += np.outer(derivative, hidden[step + 1])
        result["by"] += derivative
        hidden_gradient = parameters["Why"].T @ derivative + next_hidden_gradient
        activation_gradient = (1 - hidden[step + 1] ** 2) * hidden_gradient
        result["Wxh"][:, inputs[step]] += activation_gradient
        result["Whh"] += np.outer(activation_gradient, hidden[step])
        result["bh"] += activation_gradient
        next_hidden_gradient = parameters["Whh"].T @ activation_gradient
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise LanguageError("Nonfinite language gradient")
    return loss, result


def evaluate(parameters, sequences, baseline):
    targets = sum(len(target) for _, target in sequences)
    total = 0.0
    baseline_total = 0.0
    unknown_targets = 0
    for inputs, target in sequences:
        nll, _ = loss_and_gradients(parameters, inputs, target, gradients=False)
        total += nll * len(target)
        baseline_total -= float(np.log(baseline[target]).sum())
        unknown_targets += int((target == 0).sum())
    nll = total / targets
    unigram_nll = baseline_total / targets
    return {"targetCount": targets, "unknownTargetCount": unknown_targets,
            "rnnNllNats": nll, "unigramNllNats": unigram_nll,
            "rnnPerplexity": math.exp(nll) if nll < 700 else None,
            "unigramPerplexity": math.exp(unigram_nll) if unigram_nll < 700 else None}


def _weight_lists(parameters):
    return {name: parameters[name].tolist() for name in WEIGHT_NAMES}


def model_id(artifact):
    return "language-" + digest({key: artifact[key] for key in ("modelType", "config", "tokenizer", "weights")})


def seal_artifact(artifact):
    artifact["modelId"] = model_id(artifact)
    artifact["artifactSha256"] = digest({key: value for key, value in artifact.items() if key != "artifactSha256"})
    return artifact


def train(corpus_path, *, hidden_size=24, sequence_length=48, epochs=20, learning_rate=0.015,
          clip_norm=1.0, patience=6, seed=42, train_fraction=0.6, validation_fraction=0.2):
    integer(hidden_size, "hiddenSize", 1, 128)
    integer(sequence_length, "sequenceLength", 1, 256)
    integer(epochs, "epochs", 1, 200)
    integer(patience, "patience", 1, 200)
    integer(seed, "seed")
    learning_rate = number(learning_rate, "learningRate", positive=True)
    clip_norm = number(clip_norm, "clipNorm", positive=True)
    documents, dataset_hash = read_documents(corpus_path)
    split, indices = split_documents(documents, train_fraction, validation_fraction)
    vocabulary = tokenizer(split["train"])
    sequences = {name: chunks(value, vocabulary, sequence_length) for name, value in split.items()}
    vocab_size = len(vocabulary["tokens"])
    target_counts = np.zeros(vocab_size, dtype=np.float64)
    for _, target in sequences["train"]:
        target_counts += np.bincount(target, minlength=vocab_size)
    # Add-one smoothing includes UNK without reading holdout characters.
    baseline = (target_counts + 1) / (target_counts.sum() + vocab_size)
    parameters = initialize(vocab_size, hidden_size, seed)
    initial_hash = digest(_weight_lists(parameters))
    initial_train = evaluate(parameters, sequences["train"], baseline)["rnnNllNats"]
    best_loss = evaluate(parameters, sequences["validation"], baseline)["rnnNllNats"]
    initial_validation = best_loss
    best_parameters = {name: value.copy() for name, value in parameters.items()}
    best_epoch = 0
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    rng = np.random.default_rng(seed)
    updates = 0
    stale = 0
    clipped_updates = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        for index in rng.permutation(len(sequences["train"])):
            inputs, target = sequences["train"][index]
            _, gradients = loss_and_gradients(parameters, inputs, target)
            norm = math.sqrt(sum(float(np.sum(value * value)) for value in gradients.values()))
            if not math.isfinite(norm):
                raise LanguageError("Gradient norm overflow; no artifact produced")
            if norm > clip_norm:
                clipped_updates += 1
                gradients = {name: value * (clip_norm / norm) for name, value in gradients.items()}
            updates += 1
            for name in WEIGHT_NAMES:
                first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradients[name]
                second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradients[name] ** 2
                numerator = first_moment[name] / (1 - 0.9 ** updates)
                denominator = np.sqrt(second_moment[name] / (1 - 0.999 ** updates)) + 1e-8
                parameters[name] -= learning_rate * numerator / denominator
            if not all(np.all(np.isfinite(value)) for value in parameters.values()):
                raise LanguageError("Training produced nonfinite parameters")
        training_nll = evaluate(parameters, sequences["train"], baseline)["rnnNllNats"]
        validation_nll = evaluate(parameters, sequences["validation"], baseline)["rnnNllNats"]
        history.append({"epoch": epoch, "trainNllNats": training_nll, "validationNllNats": validation_nll})
        if validation_nll < best_loss:
            best_loss = validation_nll
            best_epoch = epoch
            best_parameters = {name: value.copy() for name, value in parameters.items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    # Test is first evaluated after checkpoint selection has finished.
    scores = {name: evaluate(best_parameters, selected, baseline) for name, selected in sequences.items()}
    training = {"datasetSha256": dataset_hash, "initializedFrom": "random weights; zero biases; no pretrained parameters",
                "initialWeightsSha256": initial_hash, "selectedWeightsSha256": digest(_weight_lists(best_parameters)),
                "seed": seed, "optimizer": "manual BPTT and Adam with global gradient-norm clipping",
                "learningRate": learning_rate, "clipNorm": clip_norm, "clippedUpdates": clipped_updates,
                "updates": updates, "epochsRequested": epochs, "epochsCompleted": len(history),
                "patience": patience, "bestEpoch": best_epoch, "initialTrainNllNats": initial_train,
                "initialValidationNllNats": initial_validation, "trainedAt": int(time.time() * 1000),
                "trainingSeconds": time.monotonic() - started, "pythonVersion": platform.python_version(),
                "numpyVersion": np.__version__, "checkpointSelection": "minimum validation NLL; test never used",
                "splitPolicy": "contiguous disjoint document indices; no transitions between documents",
                "contextPolicy": "zero hidden state for each chunk; full BPTT within sequenceLength",
                "splitDocumentIndices": indices,
                "documents": [{"index": index, "characterCount": len(document),
                               "textSha256": hashlib.sha256(document.encode("utf-8")).hexdigest()}
                              for index, document in enumerate(documents)],
                "unigram": {"fit": "train next-character targets only", "additiveSmoothing": 1.0,
                            "probabilities": baseline.tolist()}}
    evaluation = {"experimental": True, "productionAccepted": False, "unit": "nats per next character",
                  "splits": scores, "testNllBelowUnigram": scores["test"]["rnnNllNats"] < scores["test"]["unigramNllNats"],
                  "limitations": ["Small character predictor; no comprehension, reasoning, instruction following, or worker acceptance.",
                                  "Vocabulary comes from training only; unseen holdout characters are evaluated as UNK.",
                                  "Loss uses chunks with reset state, not unlimited context or a chatbot benchmark.",
                                  "A narrow authored demonstration corpus does not show performance on real user tasks."]}
    artifact = {"schemaVersion": 1, "modelType": MODEL_TYPE, "modelId": "pending",
                "config": {"hiddenSize": hidden_size, "sequenceLength": sequence_length},
                "tokenizer": vocabulary, "weights": _weight_lists(best_parameters),
                "training": training, "evaluation": evaluation, "artifactSha256": "pending"}
    seal_artifact(artifact)
    validate_artifact(artifact)
    return artifact, {"schemaVersion": 1, "modelId": artifact["modelId"], "artifactSha256": artifact["artifactSha256"],
                      "training": training, "evaluation": evaluation, "learningHistory": history}


def validate_artifact(artifact):
    try:
        if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
            raise LanguageError("Language artifact keys must match schema 1 exactly")
        integer(artifact["schemaVersion"], "schemaVersion", 1, 1)
        if artifact["modelType"] != MODEL_TYPE:
            raise LanguageError("Unsupported language model type")
        config = artifact["config"]
        if not isinstance(config, dict) or set(config) != {"hiddenSize", "sequenceLength"}:
            raise LanguageError("Invalid language configuration keys")
        hidden_size = integer(config["hiddenSize"], "hiddenSize", 1, 128)
        integer(config["sequenceLength"], "sequenceLength", 1, 256)
        vocabulary = artifact["tokenizer"]
        if not isinstance(vocabulary, dict) or set(vocabulary) != {"kind", "unknownId", "tokens"}:
            raise LanguageError("Invalid tokenizer keys")
        if vocabulary["kind"] != "train-only-character-v1":
            raise LanguageError("Unsupported tokenizer")
        integer(vocabulary["unknownId"], "unknownId", 0, 0)
        tokens = vocabulary["tokens"]
        if not isinstance(tokens, list) or not 3 <= len(tokens) <= 1024 or tokens[0] != UNKNOWN:
            raise LanguageError("Invalid character vocabulary size or UNK token")
        if any(not isinstance(character, str) or len(character) != 1 for character in tokens[1:]) or len(set(tokens)) != len(tokens):
            raise LanguageError("Character tokens must be unique one-character strings")
        if tokens[1:] != sorted(tokens[1:]):
            raise LanguageError("Character vocabulary must be ordered")
        weights = artifact["weights"]
        if not isinstance(weights, dict) or set(weights) != set(WEIGHT_NAMES):
            raise LanguageError("Invalid language weight keys")
        dimensions = {"Wxh": (hidden_size, len(tokens)), "Whh": (hidden_size, hidden_size),
                      "Why": (len(tokens), hidden_size), "bh": (hidden_size,), "by": (len(tokens),)}
        for name, shape in dimensions.items():
            values = weights[name]
            if not isinstance(values, list):
                raise LanguageError(f"{name} must be an array")
            if len(shape) == 2:
                if len(values) != shape[0] or any(not isinstance(row, list) or len(row) != shape[1] for row in values):
                    raise LanguageError(f"Invalid {name} matrix shape")
                for row in values:
                    for value in row:
                        number(value, name)
            else:
                if len(values) != shape[0]:
                    raise LanguageError(f"Invalid {name} vector shape")
                for value in values:
                    number(value, name)
        for name in ("training", "evaluation"):
            if not isinstance(artifact[name], dict):
                raise LanguageError(f"{name} must be an object")
        training = artifact["training"]
        for key in ("datasetSha256", "initialWeightsSha256", "selectedWeightsSha256"):
            value = training.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise LanguageError(f"Training metadata requires SHA-256 {key}")
        integer(training.get("seed"), "training.seed")
        integer(training.get("trainedAt"), "training.trainedAt")
        integer(training.get("bestEpoch"), "training.bestEpoch", 0, 200)
        integer(training.get("epochsCompleted"), "training.epochsCompleted", 1, 200)
        if training["bestEpoch"] > training["epochsCompleted"]:
            raise LanguageError("Checkpoint epoch cannot exceed completed epochs")
        if training["selectedWeightsSha256"] != digest(weights):
            raise LanguageError("Selected weight digest differs from the actual weights")
        unigram = training.get("unigram")
        if not isinstance(unigram, dict) or not isinstance(unigram.get("probabilities"), list):
            raise LanguageError("Training unigram probabilities are required")
        probabilities = unigram["probabilities"]
        if len(probabilities) != len(tokens) or not math.isclose(sum(number(value, "unigram probability", positive=True) for value in probabilities), 1.0, abs_tol=1e-10):
            raise LanguageError("Training unigram must match vocabulary and sum to one")
        documents = training.get("documents")
        partitions = training.get("splitDocumentIndices")
        if not isinstance(documents, list) or not 3 <= len(documents) <= 1000 or not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
            raise LanguageError("Document manifest and train/validation/test partitions are required")
        ordered = []
        for name in ("train", "validation", "test"):
            values = partitions[name]
            if not isinstance(values, list) or not values:
                raise LanguageError("Every document partition must be nonempty")
            ordered.extend(integer(value, "document index", 0, len(documents) - 1) for value in values)
        if ordered != list(range(len(documents))):
            raise LanguageError("Document partitions must be contiguous, disjoint, and cover the manifest")
        for index, document in enumerate(documents):
            if not isinstance(document, dict) or document.get("index") != index or isinstance(document.get("index"), bool):
                raise LanguageError("Invalid document manifest index")
            integer(document.get("characterCount"), "document characterCount", 2, 2_000_000)
            if not isinstance(document.get("textSha256"), str) or re.fullmatch(r"[0-9a-f]{64}", document["textSha256"]) is None:
                raise LanguageError("Document manifest requires a text digest")
        evaluation = artifact["evaluation"]
        if evaluation.get("experimental") is not True or evaluation.get("productionAccepted") is not False or evaluation.get("unit") != "nats per next character":
            raise LanguageError("Language evaluation must explicitly retain its experimental status and unit")
        scores = evaluation.get("splits")
        if not isinstance(scores, dict) or set(scores) != {"train", "validation", "test"}:
            raise LanguageError("Language evaluation requires three split scores")
        for score in scores.values():
            if not isinstance(score, dict):
                raise LanguageError("Invalid language score")
            count = integer(score.get("targetCount"), "targetCount", 1)
            integer(score.get("unknownTargetCount"), "unknownTargetCount", 0, count)
            for key in ("rnnNllNats", "unigramNllNats"):
                if number(score.get(key), key) < 0:
                    raise LanguageError("Negative language NLL is invalid")
        # Hashing validates JSON finiteness recursively, including stored metrics.
        expected_hash = digest({key: value for key, value in artifact.items() if key != "artifactSha256"})
        if artifact["artifactSha256"] != expected_hash:
            raise LanguageError("Language artifact content hash mismatch")
        if artifact["modelId"] != model_id(artifact):
            raise LanguageError("Language model identity does not match tokenizer, configuration, and weights")
    except (TypeError, ValueError, OverflowError, KeyError) as error:
        if isinstance(error, LanguageError):
            raise
        raise LanguageError(f"Invalid language artifact: {error}") from error
    return artifact


def load_artifact(path):
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 16_000_000:
            raise LanguageError("Language artifact exceeds 16 MB")
        artifact = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except (OSError, ValueError) as error:
        raise LanguageError(f"Cannot load language artifact: {error}") from error
    return validate_artifact(artifact)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LanguageError(f"Duplicate language JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise LanguageError(f"Nonfinite JSON constant: {value}")


def generate(artifact, prompt="", *, length=200, temperature=0.8, seed=42):
    validate_artifact(artifact)
    integer(length, "length", 0, 4096)
    integer(seed, "seed")
    temperature = number(temperature, "temperature")
    if not 0 <= temperature <= 2:
        raise LanguageError("temperature must be in [0, 2]; zero means greedy")
    if not isinstance(prompt, str) or len(prompt) > 4096:
        raise LanguageError("Prompt must be a string of at most 4,096 characters")
    weights = {name: np.asarray(artifact["weights"][name], dtype=np.float64) for name in WEIGHT_NAMES}
    hidden = np.zeros(artifact["config"]["hiddenSize"], dtype=np.float64)
    initial = encode(prompt, artifact["tokenizer"])
    effective_prompt = prompt
    # An empty prompt starts from train's most frequent target token.
    if len(initial) == 0:
        probabilities = artifact["training"].get("unigram", {}).get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(artifact["tokenizer"]["tokens"]):
            raise LanguageError("Empty-prompt generation requires the recorded training unigram")
        baseline = np.asarray([number(value, "unigram probability", positive=True) for value in probabilities])
        initial = np.asarray([int(np.argmax(baseline[1:])) + 1])
        effective_prompt = artifact["tokenizer"]["tokens"][int(initial[0])]
    for token in initial:
        activation = weights["Wxh"][:, token] + weights["Whh"] @ hidden + weights["bh"]
        if not np.all(np.isfinite(activation)):
            raise LanguageError("Language generation hidden activation overflow")
        hidden = np.tanh(activation)
    rng = np.random.default_rng(seed)
    continuation = []
    for _ in range(length):
        logits = weights["Why"] @ hidden + weights["by"]
        if not np.all(np.isfinite(logits)):
            raise LanguageError("Language generation logits overflow")
        # UNK is an evaluation placeholder, not an emitted character.
        usable = logits[1:]
        if temperature == 0:
            token = int(np.argmax(usable)) + 1
        else:
            with np.errstate(over="raise", invalid="raise"):
                try:
                    log_probabilities = _log_softmax(usable / temperature)
                except FloatingPointError as error:
                    raise LanguageError("Generation temperature caused numeric overflow") from error
            token = int(rng.choice(len(usable), p=np.exp(log_probabilities))) + 1
        continuation.append(artifact["tokenizer"]["tokens"][token])
        activation = weights["Wxh"][:, token] + weights["Whh"] @ hidden + weights["bh"]
        if not np.all(np.isfinite(activation)):
            raise LanguageError("Language generation hidden activation overflow")
        hidden = np.tanh(activation)
    return {"modelId": artifact["modelId"], "artifactSha256": artifact["artifactSha256"],
            "prompt": prompt, "effectivePrompt": effective_prompt, "continuation": "".join(continuation), "generatedCharacters": length,
            "seed": seed, "temperature": temperature, "experimental": True,
            "instructionFollowingValidated": False, "workerCapabilityValidated": False}


def write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
