"""Restore a published native checkpoint into a new run, without training."""

import argparse
import json
from pathlib import Path
import shutil

from ia_propia.conversation import training


def restore(package, run_dir, *, checkpoint="best", corpus=None):
    package, run_dir = Path(package).resolve(), Path(run_dir).resolve()
    if checkpoint not in ("best", "endpoint"):
        raise ValueError("Choose best or endpoint")
    if run_dir.exists():
        raise FileExistsError("Restore into a new run directory")
    receipt = training._json_read(package / "checkpoint-package.json")
    for name, identity in receipt["files"].items():
        member = (package / name).resolve()
        if not member.is_relative_to(package) or training._sha(member) != identity["sha256"]:
            raise ValueError("Published checkpoint package hash mismatch")
    root = Path(__file__).resolve().parents[3]
    corpus = Path(corpus).resolve() if corpus is not None else root / receipt["corpusRepositoryPath"]
    if training._sha(corpus) != receipt["corpusSha256"]:
        raise ValueError("Restore requires the exact published corpus")
    selected = receipt["checkpointPointers"][checkpoint]
    model, _, state, payload = training._load_checkpoint(package, selected)
    training._validate_parameter_policy(model, state)
    torch = training._torch()
    config = state["training_config"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    warmup = config["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
    training._validate_resume_payload(payload, optimizer, scheduler, model, state)
    if state["corpus_sha256"] != receipt["corpusSha256"]:
        raise ValueError("Checkpoint and corpus receipt disagree")
    # Both selected and endpoint checkpoints remain byte-exact. Only the new
    # run's status pointer changes when restoring the earlier selected state.
    run_dir.mkdir(parents=True, mode=0o700)
    shutil.copytree(package / "checkpoints", run_dir / "checkpoints")
    status = training._json_read(package / "status.json")
    status.update({"lastCheckpoint": selected, "bestCheckpoint": state["best_checkpoint"],
                   "step": state["step"], "epoch": state["epoch"],
                   "elapsedSeconds": state["elapsed_seconds"], "status": "stopped",
                   "reason": "restored_without_training"})
    training._atomic_json(run_dir / "status.json", status)
    restored = {"schemaVersion": 1, "package": str(package), "corpus": str(corpus),
                "corpusSha256": receipt["corpusSha256"], "checkpoint": selected,
                "step": state["step"], "weightsOptimizerAndRngValidated": True,
                "trainingExecuted": False, "mmapCacheMustBeRebuiltOnResume": True}
    training._atomic_json(run_dir / "restoration.json", restored)
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=("best", "endpoint"), default="best")
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.package, args.run_dir, checkpoint=args.checkpoint,
                             corpus=args.corpus), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
