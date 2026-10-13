#!/usr/bin/env python3
"""Export an untrained positional extension of a pinned own CPU artifact.

Only the new position rows are initialized. This is a new-stage artifact, not
an optimizer resume, a replacement backbone, or evidence of learned capacity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
MAX_SOURCE_CONTEXT = 512
MAX_EXTENDED_CONTEXT = 2048
MAX_PARAMETERS = 20_000_000
MAX_OPERATION_SECONDS = 120
POSITION_KEY = "position_embedding.weight"


def _integer(value, name, minimum, maximum):
    from ia_propia.conversation.configuration import ConversationError
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConversationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _new_destination(source, output, context_length):
    from ia_propia.conversation.configuration import ConversationError
    target = (Path(output) if output is not None else source.with_name(
        f"{source.name}-context{context_length}-untrained-{uuid.uuid4().hex[:12]}"))
    if target.exists() or target.is_symlink():
        raise ConversationError("Context extension requires a new nonexistent output path")
    resolved = target.resolve()
    if resolved == source.resolve() or source.resolve() in resolved.parents:
        raise ConversationError("Context extension output cannot be inside the source artifact")
    return target


def _own_lineage(manifest):
    from ia_propia.conversation.configuration import ConversationError
    from ia_propia.conversation.training import _validate_lineage
    metadata = manifest["training"]
    origin = metadata.get("trainedFrom")
    if origin == "own_checkpoint":
        _validate_lineage(metadata.get("lineage"))
    elif origin != "random_weights" or "lineage" in metadata:
        raise ConversationError("Source must retain Unidad's own random-weight lineage")
    lineage = {"parentModelId": manifest["modelId"],
               "parentWeightsSha256": manifest["files"]["weights.safetensors"]["sha256"],
               "parentCorpusSha256": metadata.get("corpusSha256"),
               "rootOrigin": "random_weights"}
    _validate_lineage(lineage)
    return lineage


def _extend_model(model, config, seed):
    import torch
    from ia_propia.conversation import artifacts
    from ia_propia.conversation.model import ConversationModel
    old_context = model.config["context_length"]
    inherited = model.state_dict()
    # A meta skeleton consumes no random numbers and allocates no second
    # backbone. The verified CPU snapshot supplies every inherited tensor.
    extended = ConversationModel._loading_skeleton(config)
    expected = extended.state_dict()
    if set(expected) != set(inherited):
        raise ValueError("Context extension must preserve the complete tensor-name set")
    positions = torch.empty(config["context_length"], config["d_model"], dtype=torch.float32)
    positions[:old_context].copy_(inherited[POSITION_KEY])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions[old_context:].normal_(mean=0.0, std=0.02, generator=generator)
    merged = {**inherited, POSITION_KEY: positions}
    for name, value in merged.items():
        if value.shape != expected[name].shape or not artifacts._finite_float32(value):
            raise ValueError("Extended tensors must have native shapes and finite float32 values")
    extended.load_state_dict(merged, strict=True, assign=True)
    actual = extended.state_dict()
    for name, before in inherited.items():
        after = actual[name][:old_context] if name == POSITION_KEY else actual[name]
        if not torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8)):
            raise ValueError(f"Inherited tensor values changed: {name}")
    extended.eval()
    return extended


def extend_context_artifact(source, output=None, *, expected_model_config,
                            context_length=1024, seed=None,
                            max_parameters=MAX_PARAMETERS):
    """Create a separate native artifact under a CPU-one-thread resource ceiling.

    The caller pins the complete source configuration before any weight payload
    is loaded. The seed defaults to that own model's recorded seed. Corpus and
    optimizer files are never opened. Returned metadata explicitly says that
    the extension has not been trained or promoted.
    """
    import torch
    from ia_propia.conversation import artifacts
    from ia_propia.conversation.configuration import (ConversationError,
        count_config_parameters, validate_config)

    started = time.monotonic()
    def deadline():
        if time.monotonic() - started > MAX_OPERATION_SECONDS:
            raise ConversationError("Context extension exceeded its 120-second operation budget")

    source = Path(source)
    admitted = validate_config(expected_model_config)
    _integer(admitted["context_length"], "source context_length", 2, MAX_SOURCE_CONTEXT)
    _integer(context_length, "context_length", admitted["context_length"] + 1, MAX_EXTENDED_CONTEXT)
    _integer(max_parameters, "max_parameters", 1, MAX_PARAMETERS)
    initialization_seed = admitted["seed"] if seed is None else _integer(seed, "seed", 0, 2**63 - 1)
    config = validate_config({**admitted, "context_length": context_length})
    old_count, new_count = count_config_parameters(admitted), count_config_parameters(config)
    if old_count > max_parameters or new_count > max_parameters:
        raise ConversationError("Context extension exceeds the explicitly admitted parameter ceiling")
    target = _new_destination(source, output, context_length)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model, tokenizer, parent = artifacts.load_artifact(source, expected_model_config=admitted)
        lineage = _own_lineage(parent)
        deadline()
        tokenizer_sha = parent["files"]["tokenizer.json"]["sha256"]
        if hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest() != tokenizer_sha:
            raise ConversationError("Native tokenizer round-trip must preserve its exact SHA-256")
        extended = _extend_model(model, config, initialization_seed)
        delta = (context_length - admitted["context_length"]) * admitted["d_model"]
        if extended.parameter_count != new_count or new_count - old_count != delta:
            raise ConversationError("Context extension must add only the declared positional parameters")
        proof = {"schemaVersion": 1, "status": "untrained",
                 "fromContextLength": admitted["context_length"], "toContextLength": context_length,
                 "inheritedPositionRange": [0, admitted["context_length"]],
                 "initializedPositionRange": [admitted["context_length"], context_length],
                 "rangeEndExclusive": True, "initialization": "normal_mean_0_std_0.02",
                 "initializationSeed": initialization_seed,
                 "parentModelId": parent["modelId"],
                 "parentWeightsSha256": lineage["parentWeightsSha256"],
                 "parentTokenizerSha256": tokenizer_sha,
                 "parentCorpusSha256": lineage["parentCorpusSha256"],
                 "parentTrainedFrom": parent["training"]["trainedFrom"],
                 "inheritedTensorNames": sorted(model.state_dict()),
                 "inheritedTensorValuesByteExact": True, "tokenizerUnchanged": True,
                 "oldParameterCount": old_count, "newParameterCount": new_count,
                 "addedParameterCount": delta, "torchThreads": 1,
                 "maxParameters": max_parameters, "maxContextLength": MAX_EXTENDED_CONTEXT,
                 "maxOperationSeconds": MAX_OPERATION_SECONDS}
        if "lineage" in parent["training"]:
            proof["parentLineage"] = deepcopy(parent["training"]["lineage"])
        training = {"trainedFrom": "own_checkpoint", "lineage": lineage,
                    "corpusSha256": lineage["parentCorpusSha256"],
                    "corpusSha256Meaning": "inherited_parent_corpus_not_extension_training",
                    "contextExtension": proof, "trainingPerformed": False,
                    "optimizerSteps": 0, "optimizerStateIncluded": False,
                    "trainingCorpusRead": False, "externalWeightsUsed": False,
                    "promoted": False}
        deadline()
        # Native save_artifact can replace exports. Reserve a new directory
        # exclusively first so this helper never invokes that replacement path.
        target.mkdir(parents=True, exist_ok=False, mode=0o700)
        manifest = artifacts.save_artifact(target, extended, tokenizer, training=training,
            evaluation={"contextExtensionTrained": False, "conversationAccepted": False})
        if manifest["files"]["tokenizer.json"]["sha256"] != tokenizer_sha:
            raise ConversationError("Exported tokenizer SHA-256 changed")
        deadline()
        return {"output": str(target.resolve()), "manifest": manifest}
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="verified own portable artifact")
    parser.add_argument("--expected-config", type=Path, required=True,
                        help="host-selected complete source configuration JSON")
    parser.add_argument("--output", type=Path, help="new nonexistent directory; default: unique sibling")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, help="own initialization seed; default: recorded model seed")
    parser.add_argument("--max-parameters", type=int, default=MAX_PARAMETERS)
    args = parser.parse_args(argv)
    from ia_propia.conversation import artifacts
    config = artifacts._parse_json(artifacts._bounded_read(args.expected_config, artifacts.MAX_JSON_BYTES))
    result = extend_context_artifact(args.source, args.output, expected_model_config=config,
        context_length=args.context_length, seed=args.seed, max_parameters=args.max_parameters)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
