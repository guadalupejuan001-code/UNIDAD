"""A small own-trained preference selector for original model candidates.

The fixed features are lexical and diagnostic, not a semantic verifier. Training
accepts explicit preferences from separate development families. Inference never
receives those labels and returns a candidate name without changing its text.
This module uses only the standard library and does not load a Transformer.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from time import perf_counter
import unicodedata


MODEL_TYPE = "conversation-linear-selector-v1"
FEATURE_NAMES = (
    "critic_preference", "critic_available", "lexical_source_support",
    "query_coverage", "query_precision", "rnn_preference", "rnn_available",
    "rnn_known_character_ratio", "rnn_unknown_ratio_available",
    "character_length", "generated_token_length", "word_diversity",
    "repeated_word_fraction", "maximum_character_run", "letter_fraction",
    "character_diversity", "ends_sentence", "question_mark_fraction",
)
_MAX_TEXT_BYTES = 65536
_FEATURE_TEXT_CHARACTERS = 512
_MAX_ARTIFACT_BYTES = 131072
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_STOP = frozenset("a al con de del el en es la las lo los para por que se un una y".split())


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _distinct(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in selector artifact")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Selector artifact requires finite JSON numbers")


def _number(value, name, low=0.0, high=1.0):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and bounded")
    return float(value)


def _text(value, name, maximum=_MAX_TEXT_BYTES, *, nonempty=False):
    if type(value) is not str or (nonempty and not value.strip()):
        raise ValueError(f"{name} requires bounded text")
    try:
        if len(value.encode("utf-8", errors="strict")) > maximum:
            raise ValueError(f"{name} exceeds its byte limit")
    except UnicodeError as error:
        raise ValueError(f"{name} requires valid UTF-8") from error
    return value


def _terms(value):
    normalized = "".join(character for character in unicodedata.normalize("NFD", value.casefold())
                         if unicodedata.category(character) != "Mn")
    return [word for word in _WORDS.findall(normalized) if len(word) > 1 and word not in _STOP]


def _query(messages):
    if type(messages) is not list or not 1 <= len(messages) <= 128:
        raise ValueError("Selector messages must be a bounded nonempty list")
    query, size = None, 0
    for message in messages:
        if (type(message) is not dict or message.get("role") not in ("system", "user", "assistant")
                or "content" not in message):
            raise ValueError("Selector messages require explicit role and content")
        content = _text(message["content"], "message content")
        size += len(content.encode("utf-8"))
        if size > 262144:
            raise ValueError("Selector messages exceed their total byte limit")
        if message["role"] == "user":
            query = content[:_FEATURE_TEXT_CHARACTERS]
    if query is None:
        raise ValueError("Selector requires the current user question")
    return set(_terms(query))


def _ranks(values):
    """Midranks retain equal diagnostics and are invariant to affine scaling."""
    present = [value for value in values if value is not None]
    return [None if value is None else (
        sum(other < value for other in present) + (sum(other == value for other in present) - 1) / 2
    ) / max(1, len(present) - 1) for value in values]


def candidate_features(messages, candidates):
    """Return bounded model-agnostic signals in the original candidate order.

    Names identify results only. RNN likelihood is ranked inside this cohort;
    missing diagnostics have explicit flags rather than fabricated likelihood.
    Each candidate's lexical computation examines at most its first 512 chars.
    """
    query_terms = _query(messages)
    if type(candidates) is not list or not 1 <= len(candidates) <= 8:
        raise ValueError("Selector requires one to eight original candidates")
    names, prepared, rnn_values = set(), [], []
    for candidate in candidates:
        if type(candidate) is not dict:
            raise ValueError("Selector candidates must be objects")
        name = candidate.get("name")
        if type(name) is not str or _NAME.fullmatch(name) is None or name in names:
            raise ValueError("Selector candidate names must be unique bounded identifiers")
        names.add(name)
        text = _text(candidate.get("text"), "original candidate", nonempty=True)
        critic = candidate.get("criticRankCost")
        if critic is not None:
            critic = _number(critic, "critic rank")
        support = _number(candidate.get("lexicalSourceSupport"), "source support")
        tokens = candidate.get("generatedTokens")
        if type(tokens) is not int or not 0 <= tokens <= 1024:
            raise ValueError("Generated token count must be bounded")
        auxiliary = candidate.get("auxiliary", {})
        if type(auxiliary) is not dict:
            raise ValueError("Auxiliary diagnostics must be an object")
        likelihood = auxiliary.get("meanNll")
        if likelihood is not None:
            likelihood = _number(likelihood, "RNN likelihood", high=1000000)
        scoreable = auxiliary.get("scoreable", likelihood is not None)
        if type(scoreable) is not bool:
            raise ValueError("Auxiliary scoreable flag must be boolean")
        if not scoreable:
            likelihood = None
        unknown = auxiliary.get("unknownRatio")
        if unknown is not None:
            unknown = _number(unknown, "RNN unknown ratio")
        prepared.append((text, critic, support, tokens, unknown))
        rnn_values.append(likelihood)
    ranks, result = _ranks(rnn_values), []
    for (text, critic, support, tokens, unknown), rnn_rank in zip(prepared, ranks):
        prefix = text[:_FEATURE_TEXT_CHARACTERS]
        words = _terms(prefix)
        unique, characters = set(words), [character for character in prefix if not character.isspace()]
        overlap = len(unique & query_terms)
        counts = Counter(words)
        repeated = sum(count - 1 for count in counts.values()) / max(1, len(words))
        maximum_run, current_run, previous = 0, 0, None
        for character in prefix:
            current_run = current_run + 1 if character == previous else 1
            maximum_run = max(maximum_run, current_run)
            previous = character
        result.append([
            1 - critic if critic is not None else 0.0, float(critic is not None), support,
            overlap / max(1, len(query_terms)), overlap / max(1, len(unique)),
            1 - rnn_rank if rnn_rank is not None else 0.0, float(rnn_rank is not None),
            1 - unknown if unknown is not None else 0.0, float(unknown is not None),
            min(len(text), 512) / 512, min(tokens, 256) / 256,
            len(unique) / max(1, len(words)), repeated, min(maximum_run, 16) / 16,
            sum(character.isalpha() for character in characters) / max(1, len(characters)),
            min(len(set(characters)), 64) / max(1, min(len(characters), 64)),
            float(prefix.rstrip().endswith((".", "!", "?", "…"))),
            min(prefix.count("?") + prefix.count("¿"), 8) / 8,
        ])
    return result


def _vectors(values):
    if type(values) is not list or not 1 <= len(values) <= 8:
        raise ValueError("A cohort requires one to eight feature vectors")
    result = []
    for vector in values:
        if type(vector) not in (tuple, list) or len(vector) != len(FEATURE_NAMES):
            raise ValueError("Selector feature dimensions differ from their contract")
        result.append([_number(value, "selector feature") for value in vector])
    return result


def _cohorts(records, partition):
    if type(records) is not list or not 1 <= len(records) <= 1000:
        raise ValueError("Training requires bounded nonempty development cohorts")
    names, normalized, pairs = set(), [], []
    for record in records:
        if type(record) is not dict or set(record) != {"cohortId", "features", "utilities"}:
            raise ValueError("Training cohorts require IDs, fixed features and explicit utilities")
        identity = _text(record["cohortId"], "cohort ID", 128, nonempty=True)
        if identity in names:
            raise ValueError("Duplicate selector training cohort")
        names.add(identity)
        vectors = _vectors(record["features"])
        utilities = record["utilities"]
        if (type(utilities) is not list or len(utilities) != len(vectors)
                or any(type(value) not in (int, float) or value not in (0, 1) for value in utilities)):
            raise ValueError("Selector utilities must be explicit binary observations")
        normalized.append({"cohortId": identity, "features": vectors, "utilities": utilities})
        for high in range(len(vectors)):
            for low in range(len(vectors)):
                if utilities[high] > utilities[low]:
                    pairs.append([a - b for a, b in zip(vectors[high], vectors[low])])
    if not pairs:
        raise ValueError(f"No preference signal in selector {partition} cohorts")
    if not any(any(value != 0 for value in pair) for pair in pairs):
        raise ValueError(f"No distinguishable preference signal in selector {partition} features")
    return names, normalized, pairs


def _loss(weights, pairs):
    total = 0.0
    for pair in pairs:
        logit = math.fsum(weight * value for weight, value in zip(weights, pair))
        total += max(-logit, 0) + math.log1p(math.exp(-abs(logit)))
    return total / len(pairs)


def _identity(artifact):
    return hashlib.sha256(_canonical({key: value for key, value in artifact.items()
                                     if key not in ("artifactSha256", "modelId")})).hexdigest()


def train_selector(training_cohorts, validation_cohorts, *, epochs=1000,
                   learning_rate=0.3, l2=0.001, seed=20261013, max_seconds=120,
                   provenance=None):
    """Bounded pairwise SGD from own random weights, selected by validation.

    Utilities are observed properties of original candidates, not runtime rules.
    Tied utilities contribute no pair. Test cohorts have no argument or path in
    this API. The caller must also separate full fictional entity families.
    """
    if type(epochs) is not int or not 1 <= epochs <= 5000:
        raise ValueError("Selector epochs must be an integer from 1 to 5000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Selector seed must be a bounded integer")
    learning_rate = _number(learning_rate, "learning rate", low=0.000001)
    l2 = _number(l2, "L2")
    max_seconds = _number(max_seconds, "training seconds", low=0.001, high=120)
    if provenance is not None:
        if type(provenance) is not dict:
            raise ValueError("Selector provenance must be a bounded JSON object")
        try:
            encoded_provenance = _canonical(provenance)
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("Selector provenance must contain finite JSON values") from error
        if len(encoded_provenance) > 8192:
            raise ValueError("Selector provenance exceeds its byte limit")
        provenance = json.loads(encoded_provenance)
    train_names, train_records, train_pairs = _cohorts(training_cohorts, "training")
    validation_names, validation_records, validation_pairs = _cohorts(validation_cohorts, "validation")
    if train_names & validation_names:
        raise ValueError("Training and validation cohort IDs overlap")
    rng = random.Random(seed)
    weights = [rng.gauss(0, 0.02) for _ in FEATURE_NAMES]
    initial_loss = _loss(weights, train_pairs)
    best_weights, best_epoch, best_loss = list(weights), 0, _loss(weights, validation_pairs)
    started, completed, history = perf_counter(), 0, []
    for epoch in range(1, epochs + 1):
        if perf_counter() - started >= max_seconds:
            break
        gradient = [0.0] * len(FEATURE_NAMES)
        for pair in train_pairs:
            logit = math.fsum(weight * value for weight, value in zip(weights, pair))
            if logit >= 0:
                exponential = math.exp(-logit)
                error = -exponential / (1 + exponential)
            else:
                error = -1 / (1 + math.exp(logit))
            for index, value in enumerate(pair):
                gradient[index] += error * value
        weights = [weight - learning_rate * (value / len(train_pairs) + l2 * weight)
                   for weight, value in zip(weights, gradient)]
        validation_loss = _loss(weights, validation_pairs)
        completed = epoch
        if validation_loss < best_loss:
            best_weights, best_epoch, best_loss = list(weights), epoch, validation_loss
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "trainLoss": _loss(weights, train_pairs),
                            "validationLoss": validation_loss})
    training = {
        "rootOrigin": "random_weights", "objective": "pairwise_preference_logistic",
        "selection": "minimum_validation_pairwise_loss", "testUsedForTraining": False,
        "seed": seed, "learningRate": learning_rate, "l2": l2,
        "epochsRequested": epochs, "completedEpochs": completed, "selectedEpoch": best_epoch,
        "maxSeconds": max_seconds, "trainingSeconds": perf_counter() - started,
        "stopReason": "epochs_completed" if completed == epochs else "time_budget",
        "trainingCohorts": len(train_records), "validationCohorts": len(validation_records),
        "trainingPairs": len(train_pairs), "validationPairs": len(validation_pairs),
        "trainingCohortsSha256": hashlib.sha256(_canonical(train_records)).hexdigest(),
        "validationCohortsSha256": hashlib.sha256(_canonical(validation_records)).hexdigest(),
        "initialTrainLoss": initial_loss, "selectedTrainLoss": _loss(best_weights, train_pairs),
        "selectedValidationLoss": best_loss, "history": history,
        "limitations": "Fixed lexical and diagnostic features; preferences are limited to observed authored tasks; not a semantic or factual verifier.",
    }
    if provenance is not None:
        training["provenance"] = provenance
    artifact = {"schemaVersion": 1, "modelType": MODEL_TYPE,
        "featureNames": list(FEATURE_NAMES), "weights": best_weights,
        "parameterCount": len(best_weights), "training": training,
        "experimental": True, "generalConversationValidated": False, "generatesAnswers": False}
    digest = _identity(artifact)
    return {**artifact, "artifactSha256": digest, "modelId": "unidad-selector-" + digest}


def _validate_artifact(artifact):
    keys = {"schemaVersion", "modelType", "featureNames", "weights", "parameterCount", "training",
            "experimental", "generalConversationValidated", "generatesAnswers", "artifactSha256", "modelId"}
    if (type(artifact) is not dict or set(artifact) != keys
            or type(artifact["schemaVersion"]) is not int or artifact["schemaVersion"] != 1
            or artifact["modelType"] != MODEL_TYPE or artifact["featureNames"] != list(FEATURE_NAMES)
            or artifact["experimental"] is not True or artifact["generalConversationValidated"] is not False
            or artifact["generatesAnswers"] is not False):
        raise ValueError("Invalid learned conversation selector artifact")
    weights = artifact["weights"]
    if type(weights) is not list or len(weights) != len(FEATURE_NAMES):
        raise ValueError("Selector parameter dimensions differ from their contract")
    for weight in weights:
        _number(weight, "selector weight", low=-1000000, high=1000000)
    if type(artifact["parameterCount"]) is not int or artifact["parameterCount"] != len(weights):
        raise ValueError("Selector parameter count mismatch")
    training = artifact["training"]
    if (type(training) is not dict or training.get("rootOrigin") != "random_weights"
            or training.get("objective") != "pairwise_preference_logistic"
            or training.get("selection") != "minimum_validation_pairwise_loss"
            or training.get("testUsedForTraining") is not False):
        raise ValueError("Selector training provenance is invalid")
    digest = _identity(artifact)
    if artifact["artifactSha256"] != digest or artifact["modelId"] != "unidad-selector-" + digest:
        raise ValueError("Selector artifact hash or identity mismatch")
    if len(_canonical(artifact)) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Selector artifact exceeds its byte limit")
    return artifact


def save_selector(path, artifact):
    """Atomically persist verified own selector parameters and provenance."""
    _validate_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(artifact) + b"\n")
    temporary.replace(path)
    return artifact["modelId"]


def load_selector(path):
    """Read a bounded artifact, checking its whole content and fixed contract."""
    with Path(path).open("rb") as stream:
        raw = stream.read(_MAX_ARTIFACT_BYTES + 1)
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Selector artifact exceeds its byte limit")
    try:
        artifact = json.loads(raw, object_pairs_hook=_distinct, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Selector artifact must be UTF-8 JSON") from error
    return deepcopy(_validate_artifact(artifact))


class LearnedSelector:
    """Apply the own-trained scores to a bounded original candidate cohort."""

    def __init__(self, artifact):
        self._artifact = deepcopy(_validate_artifact(artifact))
        self._weights = tuple(float(weight) for weight in self._artifact["weights"])

    @property
    def model_id(self):
        return self._artifact["modelId"]

    @property
    def parameter_count(self):
        return len(self._weights)

    def select(self, messages, candidates):
        started = perf_counter()
        vectors = candidate_features(messages, candidates)
        scores = [math.fsum(weight * value for weight, value in zip(self._weights, vector))
                  for vector in vectors]
        # max preserves original order for a genuine numerical tie.
        winner = max(range(len(scores)), key=lambda index: scores[index])
        return {"selectedName": candidates[winner]["name"], "modelId": self.model_id,
                "candidateScores": [{"name": candidate["name"], "score": score}
                                    for candidate, score in zip(candidates, scores)],
                "learned": True, "selectorSeconds": perf_counter() - started}
