"""Pure architecture validation and exact parameter accounting.

Planning never imports Torch or allocates model tensors. Larger architectures
require an explicit host-selected budget; old configurations retain their
original limits and serialized fields.
"""

import math


DEFAULT_CONFIG = {"d_model": 128, "n_layers": 4, "n_heads": 4, "context_length": 256,
                  "dropout": 0.0, "ffn_multiplier": 4, "seed": 42}
MAX_MODEL_PARAMETERS = 20_000_000
MAX_PARAMETER_BUDGET = 1_200_000_000
MAX_SCALE_PARAMETERS = MAX_PARAMETER_BUDGET
MAX_CONTEXT_LENGTH = 4096
MAX_VOCAB_SIZE = 16_384


class ConversationError(ValueError):
    """Invalid conversational configuration, input, or portable artifact."""


def _integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConversationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _parameter_count(config):
    d = config["d_model"]
    multiplier = config["ffn_multiplier"]
    count = config["vocab_size"] * d + config["context_length"] * d + 2 * d
    count += config["n_layers"] * (4 * d * d + 2 * multiplier * d * d + (9 + multiplier) * d)
    if "copy_head" in config:
        count += 2 * d * config["copy_head"]["rank"] + d + 1
    return count


def validate_config(config):
    if (type(config) is not dict or "vocab_size" not in config
            or set(config) - (set(DEFAULT_CONFIG) | {"vocab_size", "copy_head", "parameter_budget"})):
        raise ConversationError("Model configuration requires vocabulary size and known Transformer fields")
    result = {**DEFAULT_CONFIG, **config}
    expanded = "parameter_budget" in result
    budget = (_integer(result["parameter_budget"], "parameter_budget", MAX_MODEL_PARAMETERS,
                       MAX_PARAMETER_BUDGET) if expanded else MAX_MODEL_PARAMETERS)
    _integer(result["vocab_size"], "vocab_size", 262, MAX_VOCAB_SIZE)
    d = _integer(result["d_model"], "d_model", 8, 4096 if expanded else 1024)
    _integer(result["n_layers"], "n_layers", 1, 64 if expanded else 24)
    heads = _integer(result["n_heads"], "n_heads", 1, 64 if expanded else 16)
    _integer(result["context_length"], "context_length", 2, MAX_CONTEXT_LENGTH)
    _integer(result["ffn_multiplier"], "ffn_multiplier", 1, 8)
    _integer(result["seed"], "seed")
    dropout = result["dropout"]
    if type(dropout) not in (int, float):
        raise ConversationError("dropout must be a finite number")
    try:
        dropout = float(dropout)
    except OverflowError as error:
        raise ConversationError("dropout must be a finite number") from error
    if not math.isfinite(dropout) or not 0 <= dropout <= 0.5:
        raise ConversationError("dropout must be a finite number in [0, 0.5]")
    result["dropout"] = dropout
    if d % heads:
        raise ConversationError("d_model must be divisible by n_heads")
    if "copy_head" in result:
        head = result["copy_head"]
        if type(head) is not dict or set(head) != {"version", "rank", "seed"}:
            raise ConversationError("copy_head requires version, rank and seed")
        _integer(head["version"], "copy_head.version", 1, 1)
        _integer(head["rank"], "copy_head.rank", 32, 32)
        _integer(head["seed"], "copy_head.seed", 0, 2**32 - 1)
        result["copy_head"] = dict(head)
    if _parameter_count(result) > budget:
        raise ConversationError("Transformer exceeds the configured parameter budget")
    return result


def count_config_parameters(config):
    """Return the exact tied-embedding parameter count without allocating tensors."""
    return _parameter_count(validate_config(config))
