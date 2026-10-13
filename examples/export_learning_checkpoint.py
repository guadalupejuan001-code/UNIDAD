"""Export an explicit own native checkpoint without evaluating any corpus.

This supports inspecting LAST alongside the validation-selected BEST. An
explicit checkpoint is never described as validation-selected or accepted.
"""

import argparse
import json
from pathlib import Path
import time

from ia_propia.conversation import training
from ia_propia.conversation.artifacts import save_artifact


def export(run_dir, checkpoint, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Use a new output directory to preserve earlier artifacts")
    started = time.monotonic()
    model, tokenizer, state, payload = training._load_checkpoint(Path(run_dir).resolve(), checkpoint)
    del payload
    metadata = {"corpusSha256": state["corpus_sha256"], "selectedStep": state["step"],
                **training._origin_metadata(state), **training._loss_metadata(state["training_config"]),
                **training._sampling_metadata(state), **training._parameter_metadata(state),
                **training._data_loading_metadata(state), "selectionSplit": "explicit_checkpoint",
                "selectedCheckpoint": checkpoint, "assistantTargets": state["assistant_targets"],
                "windowPolicy": state["window_policy"], "testEvaluated": False,
                "config": state["training_config"], "selectionReason": "Inspect actual endpoint; not selected by its replies",
                "bestValidationNllInRun": state["best_validation_nll"],
                "checkpointValidationNll": None}
    manifest = save_artifact(output, model, tokenizer, training=metadata, evaluation=None)
    return {"modelId": manifest["modelId"], "step": state["step"],
            "elapsedSeconds": time.monotonic() - started,
            "selectionSplit": "explicit_checkpoint", "conversationAccepted": False,
            "files": manifest["files"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(export(arguments.run_dir, arguments.checkpoint, arguments.output_dir)))
