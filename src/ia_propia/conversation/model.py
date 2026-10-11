"""A small decoder Transformer built from random parameters, without pretrained weights."""

from copy import deepcopy
import math
import threading

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (ASSISTANT_ID, BOS_ID, DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED,
                       DEFAULT_TEMPERATURE, EOS_ID, IGNORE_INDEX, MAX_NEW_TOKENS, PAD_ID,
                       SYSTEM_ID, USER_ID)
from .tokenizer import (ConversationError, MAX_CONTEXT_LENGTH, MAX_VOCAB_SIZE,
                        encode_prompt, integer, number, validate_tokenizer)


MODEL_TYPE = "unidad-conversation-transformer"
DEFAULT_CONFIG = {"d_model": 128, "n_layers": 4, "n_heads": 4, "context_length": 256,
                  "dropout": 0.0, "ffn_multiplier": 4, "seed": 42}
MAX_MODEL_PARAMETERS = 20_000_000


def validate_config(config):
    if type(config) is not dict or "vocab_size" not in config or set(config) - (set(DEFAULT_CONFIG) | {"vocab_size"}):
        raise ConversationError("Model configuration requires vocabulary size and known Transformer fields")
    result = {**DEFAULT_CONFIG, **config}
    integer(result["vocab_size"], "vocab_size", 262, MAX_VOCAB_SIZE)
    d = integer(result["d_model"], "d_model", 8, 1024)
    layers = integer(result["n_layers"], "n_layers", 1, 24)
    heads = integer(result["n_heads"], "n_heads", 1, 16)
    context = integer(result["context_length"], "context_length", 2, MAX_CONTEXT_LENGTH)
    multiplier = integer(result["ffn_multiplier"], "ffn_multiplier", 1, 8)
    integer(result["seed"], "seed")
    result["dropout"] = number(result["dropout"], "dropout", 0, 0.5)
    if d % heads:
        raise ConversationError("d_model must be divisible by n_heads")
    # Count exactly before constructing tensors; tied output has no second copy.
    parameter_count = result["vocab_size"] * d + context * d + 2 * d
    parameter_count += layers * (4 * d * d + 2 * multiplier * d * d + (9 + multiplier) * d)
    if parameter_count > MAX_MODEL_PARAMETERS:
        raise ConversationError("Transformer exceeds the configured parameter budget")
    return result


class _CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["d_model"]
        self.n_heads = config["n_heads"]
        self.head_size = d // self.n_heads
        self.dropout = config["dropout"]
        self.qkv = nn.Linear(d, 3 * d, device="cpu", dtype=torch.float32)
        self.projection = nn.Linear(d, d, device="cpu", dtype=torch.float32)
        self.residual_dropout = nn.Dropout(self.dropout)

    def forward(self, value):
        batch, length, dimensions = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        q, k, v = (item.reshape(batch, length, self.n_heads, self.head_size).transpose(1, 2) for item in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        merged = attended.transpose(1, 2).contiguous().reshape(batch, length, dimensions)
        return self.residual_dropout(self.projection(merged))


class _DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["d_model"]
        self.attention_norm = nn.LayerNorm(d, device="cpu", dtype=torch.float32)
        self.attention = _CausalAttention(config)
        self.feedforward_norm = nn.LayerNorm(d, device="cpu", dtype=torch.float32)
        self.feedforward = nn.Sequential(nn.Linear(d, config["ffn_multiplier"] * d, device="cpu", dtype=torch.float32), nn.GELU(),
                                         nn.Linear(config["ffn_multiplier"] * d, d, device="cpu", dtype=torch.float32), nn.Dropout(config["dropout"]))

    def forward(self, value):
        value = value + self.attention(self.attention_norm(value))
        return value + self.feedforward(self.feedforward_norm(value))


class ConversationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = validate_config(config)
        config = self._config
        self._generation_lock = threading.RLock()
        # Preserve caller RNG state. The trainer owns its dropout/shuffle seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"], device="cpu", dtype=torch.float32)
            self.position_embedding = nn.Embedding(config["context_length"], config["d_model"], device="cpu", dtype=torch.float32)
            self.embedding_dropout = nn.Dropout(config["dropout"])
            self.blocks = nn.ModuleList(_DecoderBlock(config) for _ in range(config["n_layers"]))
            self.final_norm = nn.LayerNorm(config["d_model"], device="cpu", dtype=torch.float32)
            self.apply(self._initialize)
            residual_std = 0.02 / math.sqrt(2 * config["n_layers"])
            for block in self.blocks:
                nn.init.normal_(block.attention.projection.weight, std=residual_std)
                nn.init.normal_(block.feedforward[2].weight, std=residual_std)

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ConversationError("input_ids must be a two-dimensional long tensor")
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self._config["context_length"]:
            raise ConversationError("Input sequence must fit the model's nonempty context")
        if input_ids.device != self.token_embedding.weight.device:
            raise ConversationError("Input tensor and model must use the same device")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self._config["vocab_size"]):
            raise ConversationError("Input contains an out-of-vocabulary token")
        positions = torch.arange(length, device=input_ids.device)
        value = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            value = block(value)
        # Tied output projection is functional: only one embedding tensor is
        # serialized, so safetensors never has an ambiguous shared-weight copy.
        return F.linear(self.final_norm(value), self.token_embedding.weight)


def assistant_token_loss(logits, labels, loss_reduction="token_mean"):
    """Mean target CE, optionally giving each supervised window equal weight.

    A sequence is one encoded window, which may cover only part of a long reply.
    Rows without targets do not contribute; an entirely unsupervised batch fails.
    """
    if loss_reduction not in ("token_mean", "sequence_mean"):
        raise ConversationError("Unknown assistant loss reduction")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ConversationError("logits must be a batch/sequence/vocabulary tensor")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.shape != logits.shape[:2]:
        raise ConversationError("labels must be a matching two-dimensional long tensor")
    selected = labels != IGNORE_INDEX
    if not torch.any(selected):
        raise ConversationError("A supervised batch must contain assistant targets")
    if torch.any(labels[selected] < 0) or torch.any(labels[selected] >= logits.shape[-1]):
        raise ConversationError("Supervised labels contain out-of-vocabulary targets")
    if not torch.all(torch.isfinite(logits[selected])):
        raise ConversationError("Supervised logits must be finite")
    if loss_reduction == "token_mean":
        return F.cross_entropy(logits[selected], labels[selected])
    token_losses = F.cross_entropy(logits[selected], labels[selected], reduction="none")
    rows = selected.nonzero(as_tuple=False)[:, 0]
    sums = token_losses.new_zeros(labels.shape[0]).scatter_add(0, rows, token_losses)
    counts = selected.sum(dim=1)
    valid = counts > 0
    return (sums[valid] / counts[valid]).mean()


def chat(model, tokenizer, messages, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
         temperature=DEFAULT_TEMPERATURE, seed=DEFAULT_SEED):
    """Generate an assistant continuation with local RNG and an explicit budget.

    This first implementation recomputes a sliding causal context. It makes no
    claims of instruction-following competence or calibrated confidence.
    """
    if not isinstance(model, ConversationModel):
        raise ConversationError("Expected a configured ConversationModel")
    validate_tokenizer(tokenizer)
    if tokenizer.get_vocab_size() != model.config["vocab_size"]:
        raise ConversationError("Tokenizer vocabulary differs from the model")
    integer(max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS)
    temperature = number(temperature, "temperature", 0, 2)
    integer(seed, "seed")
    context, discarded = encode_prompt(messages, tokenizer, model.config["context_length"])
    device = model.token_embedding.weight.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generated = []
    stop_reason = "token_budget"
    forbidden = [PAD_ID, BOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID]
    with model._generation_lock:
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    inputs = torch.tensor([context[-model.config["context_length"]:]], dtype=torch.long, device=device)
                    logits = model(inputs)[0, -1].float()
                    if not torch.all(torch.isfinite(logits)):
                        raise ConversationError("Generation produced nonfinite logits")
                    logits[forbidden] = -torch.inf
                    if temperature == 0:
                        token = int(torch.argmax(logits).item())
                    else:
                        probabilities = torch.softmax(logits / temperature, dim=-1)
                        if not torch.all(torch.isfinite(probabilities)):
                            raise ConversationError("Generation produced nonfinite probabilities")
                        token = int(torch.multinomial(probabilities, 1, generator=generator).item())
                    generated.append(token)
                    if token == EOS_ID:
                        stop_reason = "eos"
                        break
                    context.append(token)
        finally:
            model.train(was_training)
    return {"text": tokenizer.decode(generated, skip_special_tokens=True), "generatedTokenIds": generated,
            "generatedTokens": len(generated), "stopReason": stop_reason,
            "promptTokens": len(context) - len([token for token in generated if token != EOS_ID]),
            "discardedPromptTokens": discarded, "seed": seed, "temperature": temperature,
            "modelType": MODEL_TYPE, "experimental": True,
            "instructionFollowingValidated": False, "generalConversationValidated": False}
