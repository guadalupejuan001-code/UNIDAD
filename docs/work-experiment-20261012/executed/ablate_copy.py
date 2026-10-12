"""Diagnostic only: export C7's unchanged backbone without its copy head."""
import json
from pathlib import Path
import torch
from ia_propia.conversation.artifacts import load_artifact, save_artifact
from ia_propia.conversation.model import ConversationModel

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
original, tokenizer, manifest = load_artifact(root.parent / "UNIDAD/models/conversation-reference-c7")
config = original.config
config.pop("copy_head")
model = ConversationModel(config)
model.load_state_dict({k: v for k, v in original.state_dict().items() if not k.startswith("copy_")}, strict=True)
result = save_artifact(root / "generator-only", model, tokenizer,
    training={"trainedFrom": "own_checkpoint", "rootOrigin": "random_weights", "diagnosticOnly": True,
              "intervention": "Remove copy head; preserve every backbone weight exactly",
              "lineage": {"parentModelId": manifest["modelId"], "parentWeightsSha256": manifest["files"]["weights.safetensors"]["sha256"], "rootOrigin": "random_weights"}},
    evaluation={"productionAccepted": False})
print(json.dumps({"modelId": result["modelId"], "parameterCount": model.parameter_count}))
