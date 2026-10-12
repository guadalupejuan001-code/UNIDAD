"""Export existing own weights for a local ONNX adapter; never trains a model.

Run with the repository's conversation dependencies plus onnx==1.17.0 and
onnxruntime==1.20.1. The .unidad archive is NOT a renamed GGUF file.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F

from ia_propia.conversation import contract
from ia_propia.conversation.runtime import ConversationEngine
from ia_propia.conversation.tokenizer import encode_prompt


class MobileProjection(torch.nn.Module):
    """The native backbone, tied output and USER successor copy mixture.

    Explicit role tracking is supplied as a boolean input by the host. Padding
    the successor keys makes the one-token/no-source case portable as well.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, copy_allowed):
        hidden = self.model._hidden(input_ids)
        query = hidden[:, -1, :]
        vocabulary = F.linear(query, self.model.token_embedding.weight)
        if "copy_head" not in self.model.config:
            # Retain the protocol's second input for generator-only checkpoints.
            return vocabulary + copy_allowed.float().sum(dim=1, keepdim=True) * 0.0
        previous = torch.cat((torch.zeros_like(hidden[:, :1, :]), hidden[:, :-1, :]), dim=1)
        keys = self.model.copy_key(previous)
        projected = self.model.copy_query(query)
        scores = (keys * projected[:, None, :]).sum(dim=-1) / math.sqrt(self.model.config["copy_head"]["rank"])
        available = copy_allowed.any(dim=1, keepdim=True)
        scores = scores.masked_fill(~copy_allowed, -torch.inf)
        scores = torch.where(available, scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, dim=-1) * copy_allowed
        generator = torch.softmax(vocabulary, dim=-1)
        copied = torch.zeros_like(generator).scatter_add(1, input_ids, attention)
        gate = torch.sigmoid(self.model.copy_gate(query)) * available
        mixture = (1 - gate) * generator + gate * copied
        return torch.where(available, mixture.clamp_min(torch.finfo(torch.float32).tiny).log(), vocabulary)


def role_mask(ids):
    role = -1
    allowed = []
    for index, token in enumerate(ids):
        if token <= contract.ASSISTANT_ID:
            role = token
        allowed.append(index > 0 and role == contract.USER_ID and token > contract.ASSISTANT_ID)
    return allowed


def file_record(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def infer(session, ids):
    return session.run(["logits"], {"input_ids": np.asarray([ids], dtype=np.int64),
                                   "copy_allowed": np.asarray([role_mask(ids)], dtype=bool)})[0][0]


def greedy(session, tokenizer, messages, context_length, budget):
    ids, discarded = encode_prompt(messages, tokenizer, context_length)
    generated = []
    for _ in range(budget):
        scores = infer(session, ids[-context_length:])
        scores[[0, 1, 3, 4, 5]] = -np.inf
        token = int(scores.argmax())
        generated.append(token)
        if token == contract.EOS_ID:
            break
        ids.append(token)
    return {"text": tokenizer.decode(generated, skip_special_tokens=True),
            "generatedTokenIds": generated, "discardedPromptTokens": discarded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace an existing exported model")
    torch.set_num_threads(1)
    engine = ConversationEngine(args.model)
    model, tokenizer = engine._model.eval(), engine._tokenizer
    source = {name: file_record(args.model / name) for name in
              ("weights.safetensors", "tokenizer.json", "config.json", "manifest.json")}
    context = model.config["context_length"]
    example = [1, 3, 100, 2, 4, 101, 102, 2, 5]
    wrapper = MobileProjection(model).eval()
    with tempfile.TemporaryDirectory(prefix="unidad-export-") as temporary:
        root = Path(temporary)
        graph_path = root / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(wrapper, (torch.tensor([example]), torch.tensor([role_mask(example)])), graph_path,
                              input_names=["input_ids", "copy_allowed"], output_names=["logits"],
                              dynamic_axes={"input_ids": {1: "sequence"}, "copy_allowed": {1: "sequence"}},
                              opset_version=17, dynamo=False, external_data=False)
        graph = onnx.load(graph_path)
        onnx.checker.check_model(graph, full_check=True)
        assert all(t.data_location != onnx.TensorProto.EXTERNAL for t in graph.graph.initializer)
        assert all(node.domain in ("", "ai.onnx") for node in graph.graph.node)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(graph_path), options, providers=["CPUExecutionProvider"])
        tests = [[5], [100, 101, 5], example, [1, 4, 400, 400, 400, 2, 5],
                 [1, 4, 20, 21, 2, 5, 22, 2, 4, 23, 24, 2, 5],
                 [1, 3, 100, 2, 5, 101],
                 [4] + [262 + (i * 17) % (model.config["vocab_size"] - 262) for i in range(context - 2)] + [5]]
        max_error = 0.0
        with torch.inference_mode():
            for ids in tests:
                values = torch.tensor([ids])
                selected = torch.zeros_like(values, dtype=torch.bool)
                selected[:, -1] = True
                reference = model.selected_logits(values, selected)[0].numpy()
                portable = infer(session, ids)
                max_error = max(max_error, float(np.abs(reference - portable).max()))
                np.testing.assert_allclose(portable, reference, rtol=3e-4, atol=3e-4)
        messages = []
        replies = []
        for prompt in ["Hola, ¿cómo te llamas?", "Mi cita es el martes a las cuatro.",
                       "La cambiaron al jueves a las cinco.", "¿Qué día y a qué hora es mi cita?"]:
            messages.append({"role": "user", "content": prompt})
            native = engine.reply(messages, max_new_tokens=32, temperature=0, seed=42)
            mobile = greedy(session, tokenizer, messages, context, 32)
            assert mobile["generatedTokenIds"] == native["generatedTokenIds"], (native, mobile)
            assert mobile["text"] == native["text"]
            replies.append({"messages": list(messages), "expected": mobile})
            if native["text"].strip():
                messages.append({"role": "assistant", "content": native["text"]})
            else:
                messages = []
        texts = ["Hola, ¿cómo estás?", "  a\n\n b\t\r\n", "don't I'M I'd we'll", "🐜👩🏽‍💻中文العربية",
                 "á é í ó ú ñ a\u0301", "\u00a0\u2003\u2028\u0085", "", "a" * 128,
                 contract.DEFAULT_SYSTEM_PROMPT]
        golden = [{"text": t, "ids": tokenizer.encode(t, add_special_tokens=False).ids} for t in texts]
        for name in ("config.json", "tokenizer.json"):
            (root / name).write_bytes((args.model / name).read_bytes())
        manifest = {"format": "unidad-onnx", "formatVersion": 1, "modelType": "unidad-conversation-transformer",
                    "modelId": engine.model_id, "sourceFiles": source, "parameterCount": engine.parameter_count,
                    "contextLength": context, "vocabSize": model.config["vocab_size"],
                    "experimental": True, "productionAccepted": False, "temperature": 0,
                    "contract": {"specialTokens": list(contract.SPECIAL_TOKENS),
                                 "defaultSystemPrompt": contract.DEFAULT_SYSTEM_PROMPT,
                                 "maxMessages": contract.MAX_MESSAGES, "maxMessageBytes": contract.MAX_MESSAGE_BYTES,
                                 "maxDialogueBytes": contract.MAX_DIALOGUE_BYTES, "maxNewTokens": contract.MAX_NEW_TOKENS},
                    "files": {name: file_record(root / name) for name in ("model.onnx", "config.json", "tokenizer.json")}}
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("manifest.json", "model.onnx", "config.json", "tokenizer.json"):
                archive.write(root / name, name)
        evidence = {"modelId": engine.model_id, "sourceFiles": source, "export": file_record(args.output),
                    "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
                    "logitCases": len(tests), "maxAbsoluteLogitError": max_error,
                    "greedyTurnsIdentical": len(replies), "replies": replies, "tokenizerCases": golden,
                    "qualityImprovementClaimed": False, "androidDeviceTested": False}
        assert source == {name: file_record(args.model / name) for name in source}, "Source artifact changed"
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "sha256": evidence["export"]["sha256"],
                          "logitCases": len(tests), "maxAbsoluteLogitError": max_error, "greedyTurnsIdentical": len(replies)}))


if __name__ == "__main__":
    main()
