"""Optional, resumable priorities from observed TRAIN-window losses only.

This module owns no RNG, order, model or corpus. The trainer supplies detached
mean assistant-token CE (including EOS) computed before the optimizer update,
then commits observations only after a successful step. Refreshing sampled
orders, preserving the current order/cursor and checking corpus hashes belong
to the trainer. Absence of this option must bypass this module entirely.
"""

from collections import defaultdict
import math


MAX_WINDOWS = 1_000_000
MAX_BATCH = 256
MAX_COUNT = 2**63 - 1
MAX_LOSS = 1_000_000.0
DEFAULT_CONFIG = {
    "mode": "train_loss_ema", "ema_decay": .9, "exponent": .5,
    "coverage": .25, "min_priority": .5, "max_priority": 2.,
    "epsilon": 1e-8, "refresh_every_steps": 64,
}
_STATE_KEYS = {"schemaVersion", "windowIds", "baseWeights", "config", "scores", "counts"}


def _number(value, name, low, high, *, positive=False):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid adaptive {name}")
    return float(value)


def validate_config(config):
    """Normalize an explicit opt-in config; None means the legacy bypass.

    Priority bounds apply BEFORE normalizing the hard-example distribution;
    they are not bounds on the final base-weight multiplier.
    """
    if config is None:
        return None
    if type(config) is not dict or set(config) - set(DEFAULT_CONFIG) or config.get("mode") != "train_loss_ema":
        raise ValueError("Adaptive sampling requires explicit mode train_loss_ema and known fields")
    result = {**DEFAULT_CONFIG, **config}
    for key, low, high in (("ema_decay", 0, .99999), ("exponent", 0, 2),
                           ("coverage", .25, 1), ("min_priority", .05, 1),
                           ("max_priority", 1, 8), ("epsilon", 1e-12, .01)):
        result[key] = _number(result[key], key, low, high)
    refresh = result["refresh_every_steps"]
    if type(refresh) is not int or not 1 <= refresh <= 100_000:
        raise ValueError("Invalid adaptive refresh_every_steps")
    return result


def _identifier(value):
    if type(value) is not str or not value.strip() or len(value) > 512 or any(ord(char) < 32 for char in value):
        raise ValueError("Adaptive window IDs must be nonempty strings of at most 512 characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("Adaptive window IDs must be valid UTF-8") from exc
    return value


class AdaptiveSampler:
    """EMA cache and deterministic weights for a fixed TRAIN-window ordering.

    A duplicate index within one batch contributes its mean CE once to EMA;
    counts still include every draw. No validation/test loss may be supplied.
    Scores do not imply quality: irreducible errors may remain difficult. This
    opt-in curriculum changes the expected training objective without importance
    correction; it is not equivalent to the original sampling distribution.
    """

    def __init__(self, window_ids, base_weights, config, state=None):
        normalized = validate_config(config)
        if normalized is None:
            raise ValueError("Construct an AdaptiveSampler only for an explicit enabled config")
        if type(window_ids) not in (list, tuple) or not 1 <= len(window_ids) <= MAX_WINDOWS:
            raise ValueError("Adaptive sampling needs 1 to 1000000 TRAIN-window IDs")
        ids = tuple(_identifier(value) for value in window_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Adaptive TRAIN-window IDs must be unique")
        if type(base_weights) not in (list, tuple) or len(base_weights) != len(ids):
            raise ValueError("Adaptive base weights must match the TRAIN-window IDs")
        self._window_ids = ids
        self._base_weights = tuple(_number(value, "base weight", 0, 1e12, positive=True) for value in base_weights)
        self._config = normalized
        self._scores = [None] * len(ids)
        self._counts = [0] * len(ids)
        if state is not None:
            self._restore(state)

    @property
    def config(self):
        return dict(self._config)

    def _restore(self, state):
        if type(state) is not dict or set(state) != _STATE_KEYS or type(state["schemaVersion"]) is not int or state["schemaVersion"] != 1:
            raise ValueError("Invalid adaptive checkpoint schema")
        if type(state["windowIds"]) is not list or tuple(state["windowIds"]) != self._window_ids:
            raise ValueError("Adaptive checkpoint TRAIN-window IDs/order changed")
        bases = state["baseWeights"]
        if type(bases) is not list or len(bases) != len(self._base_weights):
            raise ValueError("Invalid adaptive checkpoint base weights")
        restored_bases = tuple(_number(value, "checkpoint base weight", 0, 1e12, positive=True) for value in bases)
        if restored_bases != self._base_weights:
            raise ValueError("Adaptive checkpoint provenance base weights changed")
        if type(state["config"]) is not dict or set(state["config"]) != set(self._config) or validate_config(state["config"]) != self._config:
            raise ValueError("Adaptive checkpoint configuration changed")
        scores, counts = state["scores"], state["counts"]
        if type(scores) is not list or type(counts) is not list or len(scores) != len(self._window_ids) or len(counts) != len(self._window_ids):
            raise ValueError("Adaptive checkpoint cache lengths changed")
        restored_scores, restored_counts = [], []
        for score, count in zip(scores, counts):
            if type(count) is not int or not 0 <= count <= MAX_COUNT or ((score is None) != (count == 0)):
                raise ValueError("Invalid adaptive checkpoint score/count consistency")
            restored_scores.append(None if score is None else _number(score, "checkpoint score", 0, MAX_LOSS))
            restored_counts.append(count)
        self._scores, self._counts = restored_scores, restored_counts

    def _prepare_update(self, indices, losses, *, split):
        if type(split) is not str or split != "train":
            raise ValueError("Adaptive observations are TRAIN-only")
        if type(indices) not in (list, tuple) or type(losses) not in (list, tuple) or not 1 <= len(indices) <= MAX_BATCH or len(indices) != len(losses):
            raise ValueError("Adaptive updates need 1 to 256 matching indices and detached scalar losses")
        observations = defaultdict(list)
        for index, loss in zip(indices, losses):
            if type(index) is not int or not 0 <= index < len(self._window_ids):
                raise ValueError("Invalid adaptive TRAIN-window index")
            observations[index].append(_number(loss, "observed loss", 0, MAX_LOSS))
        pending = []
        decay = self._config["ema_decay"]
        for index, values in observations.items():
            count = self._counts[index] + len(values)
            if count > MAX_COUNT:
                raise ValueError("Adaptive observation count overflow")
            observed = math.fsum(values) / len(values)
            previous = self._scores[index]
            score = observed if previous is None else previous + (1-decay) * (observed-previous)
            pending.append((index, _number(score, "updated score", 0, MAX_LOSS), count))
        return pending

    def validate_update(self, indices, losses, *, split="train"):
        """Pure preflight before backward/optimizer; does not commit scores.

        The trainer must retain its own stable scalar/index snapshot until the
        corresponding successful step is committed with update().
        """
        self._prepare_update(indices, losses, split=split)

    def update(self, indices, losses, *, split="train"):
        """Atomically commit a successful step's detached TRAIN observations.

        Invalid data or overflowing counters leave the complete cache unchanged.
        The caller must not invoke this after failed/skipped optimizer steps.
        """
        pending = self._prepare_update(indices, losses, split=split)
        for index, score, count in pending:
            self._scores[index], self._counts[index] = score, count

    def weights(self):
        """Return the real mixture of base and normalized hard distributions.

        p_i = coverage * p_base_i + (1-coverage) * p_hard_i.
        The result is rescaled by sum(base weights) for random.choices. Unseen
        windows have raw priority 1, not necessarily final multiplier 1. With
        default bounds, final/base ratios lie in [.4375, 3.25]. Coverage is a
        probability floor relative to the base distribution, not a guarantee
        that every window is drawn in a finite sampling cycle.
        """
        observed = [score for score in self._scores if score is not None]
        if not observed or self._config["coverage"] == 1 or self._config["exponent"] == 0:
            return list(self._base_weights)
        mean = math.fsum(observed) / len(observed)
        epsilon = self._config["epsilon"]
        low, high = self._config["min_priority"], self._config["max_priority"]
        priorities = [1. if score is None else min(high, max(low, ((score+epsilon)/(mean+epsilon))**self._config["exponent"]))
                      for score in self._scores]
        # Scaling by the largest base avoids overflow/underflow in the weighted
        # normalization. Provenance factors remain multiplicative per window.
        maximum = max(self._base_weights)
        scaled = [base/maximum for base in self._base_weights]
        average = math.fsum(base*priority for base, priority in zip(scaled, priorities)) / math.fsum(scaled)
        coverage = self._config["coverage"]
        result = [base*(coverage+(1-coverage)*priority/average)
                  for base, priority in zip(self._base_weights, priorities)]
        if any(not math.isfinite(weight) or weight <= 0 for weight in result):
            raise ValueError("Adaptive weights are not representable as positive finite floats")
        return result

    def state_dict(self):
        """Independent JSON-safe snapshot; the trainer saves its RNG/order too."""
        return {"schemaVersion": 1, "windowIds": list(self._window_ids), "baseWeights": list(self._base_weights),
                "config": dict(self._config), "scores": list(self._scores), "counts": list(self._counts)}

    def summary(self):
        """Descriptive cache statistics; no claim of improved learning quality."""
        weights = self.weights()
        ratios = [weight/base for weight, base in zip(weights, self._base_weights)]
        observed = [score for score in self._scores if score is not None]
        return {"mode": self._config["mode"], "windows": len(self._window_ids), "observedWindows": len(observed),
                "observationDraws": sum(self._counts), "meanObservedScore": None if not observed else math.fsum(observed)/len(observed),
                "coverage": self._config["coverage"], "minActualBaseRatio": min(ratios), "maxActualBaseRatio": max(ratios),
                "refreshEverySteps": self._config["refresh_every_steps"], "qualityImprovementValidated": False}
