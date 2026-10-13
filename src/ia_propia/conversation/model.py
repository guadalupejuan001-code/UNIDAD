"""A small decoder Transformer built from random parameters, without pretrained weights."""

from copy import deepcopy
import math
import threading

import torch
from torch import nn
from torch.nn import functional as F

from .configuration import (DEFAULT_CONFIG, MAX_MODEL_PARAMETERS, MAX_PARAMETER_BUDGET,
                            MAX_SCALE_PARAMETERS, count_config_parameters, validate_config)
from .contract import (ASSISTANT_ID, BOS_ID, DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED,
                       DEFAULT_TEMPERATURE, EOS_ID, IGNORE_INDEX, MAX_NEW_TOKENS, PAD_ID,
                       SYSTEM_ID, USER_ID)
from .tokenizer import (ConversationError, MAX_CONTEXT_LENGTH, MAX_VOCAB_SIZE,
                        encode_prompt, integer, number, validate_tokenizer)


MODEL_TYPE = "unidad-conversation-transformer"


class _CausalAttention(nn.Module):
    def __init__(self, config, *, device="cpu"):
        super().__init__()
        d = config["d_model"]
        self.n_heads = config["n_heads"]
        self.head_size = d // self.n_heads
        self.dropout = config["dropout"]
        self.qkv = nn.Linear(d, 3 * d, device=device, dtype=torch.float32)
        self.projection = nn.Linear(d, d, device=device, dtype=torch.float32)
        self.residual_dropout = nn.Dropout(self.dropout)

    def forward(self, value):
        batch, length, dimensions = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        q, k, v = (item.reshape(batch, length, self.n_heads, self.head_size).transpose(1, 2) for item in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        merged = attended.transpose(1, 2).contiguous().reshape(batch, length, dimensions)
        return self.residual_dropout(self.projection(merged))


class _DecoderBlock(nn.Module):
    def __init__(self, config, *, device="cpu"):
        super().__init__()
        d = config["d_model"]
        self.attention_norm = nn.LayerNorm(d, device=device, dtype=torch.float32)
        self.attention = _CausalAttention(config, device=device)
        self.feedforward_norm = nn.LayerNorm(d, device=device, dtype=torch.float32)
        self.feedforward = nn.Sequential(nn.Linear(d, config["ffn_multiplier"] * d, device=device, dtype=torch.float32), nn.GELU(),
                                         nn.Linear(config["ffn_multiplier"] * d, d, device=device, dtype=torch.float32), nn.Dropout(config["dropout"]))

    def forward(self, value):
        value = value + self.attention(self.attention_norm(value))
        return value + self.feedforward(self.feedforward_norm(value))


class ConversationModel(nn.Module):
    def __init__(self, config, *, _device="cpu"):
        super().__init__()
        if _device not in ("cpu", "meta"):
            raise ConversationError("Model construction requires CPU parameters or a private loading skeleton")
        self._config = validate_config(config)
        config = self._config
        self._generation_lock = threading.RLock()
        # Preserve caller RNG state. The trainer owns its dropout/shuffle seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"], device=_device, dtype=torch.float32)
            self.position_embedding = nn.Embedding(config["context_length"], config["d_model"], device=_device, dtype=torch.float32)
            self.embedding_dropout = nn.Dropout(config["dropout"])
            self.blocks = nn.ModuleList(_DecoderBlock(config, device=_device) for _ in range(config["n_layers"]))
            self.final_norm = nn.LayerNorm(config["d_model"], device=_device, dtype=torch.float32)
            if _device == "cpu":
                self.apply(self._initialize)
                residual_std = 0.02 / math.sqrt(2 * config["n_layers"])
                for block in self.blocks:
                    nn.init.normal_(block.attention.projection.weight, std=residual_std)
                    nn.init.normal_(block.feedforward[2].weight, std=residual_std)
        # Optional random head uses its own seed; the existing backbone and
        # caller RNG remain untouched. Absent config adds no modules or fields.
        if "copy_head" in config:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["copy_head"]["seed"])
                rank, dimensions = config["copy_head"]["rank"], config["d_model"]
                self.copy_query = nn.Linear(dimensions, rank, bias=False, device=_device, dtype=torch.float32)
                self.copy_key = nn.Linear(dimensions, rank, bias=False, device=_device, dtype=torch.float32)
                self.copy_gate = nn.Linear(dimensions, 1, device=_device, dtype=torch.float32)
                if _device == "cpu":
                    nn.init.xavier_uniform_(self.copy_query.weight)
                    nn.init.xavier_uniform_(self.copy_key.weight)
                    nn.init.normal_(self.copy_gate.weight, std=0.02)
                    nn.init.constant_(self.copy_gate.bias, -4.6)

    @classmethod
    def _loading_skeleton(cls, config):
        """Validate tensor structure on meta without a second full random model."""
        return cls(config, _device="meta")

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

    def _hidden(self, input_ids):
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
        return self.final_norm(value)

    def forward(self, input_ids):
        # Tied output projection is functional: only one embedding tensor is
        # serialized, so safetensors never has an ambiguous shared-weight copy.
        if "copy_head" in self._config:
            values, _ = self._copy_project(input_ids)
            return values.reshape(*input_ids.shape, -1)
        return F.linear(self._hidden(input_ids), self.token_embedding.weight)

    def selected_logits(self, input_ids, mask):
        """Project selected causal positions in row-major order, retaining all context.

        Selection occurs after every Transformer block; unselected input tokens
        still condition later targets and receive their normal context gradients.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise ConversationError("input_ids must be a two-dimensional long tensor")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != input_ids.shape:
            raise ConversationError("Projection mask must be a matching boolean tensor")
        if mask.device != input_ids.device:
            raise ConversationError("Projection mask and inputs must use the same device")
        if not torch.any(mask):
            raise ConversationError("Projection mask must select at least one position")
        if "copy_head" in self._config:
            return self._copy_project(input_ids, mask)[0]
        return F.linear(self._hidden(input_ids)[mask], self.token_embedding.weight)


    @staticmethod
    def _copy_roles(input_ids):
        """Read explicit format markers, never names, values or natural language.

        A left-cropped fragment without a visible marker has unknown role and
        is excluded as a copy source. EOS/PAD and every role marker reset scope.
        """
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].expand_as(input_ids)
        boundaries = torch.where(input_ids <= ASSISTANT_ID, positions, -1).cummax(dim=1).values
        roles = input_ids.gather(1, boundaries.clamp_min(0))
        return roles, boundaries >= 0

    def _copy_project(self, input_ids, mask=None, *, details=False):
        """One hidden pass: generator plus learned causal USER successor pointer.

        Returned scores are logits-compatible log mixture probabilities wherever
        a USER source exists, and the original generator logits otherwise.
        All probability arithmetic is F32; the result retains projection dtype
        (BF16 under the trainer's existing autocast, F32 at native inference).
        No labels are read here and no intermediate tensor is stored on self.
        """
        if "copy_head" not in self._config:
            raise ConversationError("Copy projection requires an explicit copy_head")
        if mask is not None:
            if not isinstance(input_ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise ConversationError("Projection requires tensor inputs and mask")
            if mask.dtype != torch.bool or mask.shape != input_ids.shape or mask.device != input_ids.device or not torch.any(mask):
                raise ConversationError("Projection requires a nonempty matching boolean mask")
        hidden = self._hidden(input_ids)
        if mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        selected = mask.nonzero(as_tuple=False)
        rows, times = selected[:, 0], selected[:, 1]
        queries = hidden[mask]
        vocabulary_logits = F.linear(queries, self.token_embedding.weight)
        if not torch.all(torch.isfinite(vocabulary_logits)):
            raise ConversationError("Copy generator scores must be finite")
        length = input_ids.shape[1]
        roles, known = self._copy_roles(input_ids)
        user_sources = known & (roles == USER_ID) & (input_ids > ASSISTANT_ID)
        source_positions = torch.arange(1, length, device=input_ids.device)
        source_ids = input_ids[rows, 1:]
        allowed = user_sources[rows, 1:] & (source_positions[None, :] <= times[:, None])
        available = allowed.any(dim=1)
        # Small neural head and probability arithmetic stay F32, independent of
        # the backbone's autocast. Only final output returns to projection dtype.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            gate_logits = self.copy_gate(queries.float()).flatten()
            gate = torch.sigmoid(gate_logits)
            if not torch.all(torch.isfinite(gate_logits)):
                raise ConversationError("Copy gate scores must be finite")
            if length > 1:
                keys = self.copy_key(hidden[:, :-1].float())
                projected_queries = self.copy_query(queries.float())
                scores = (keys[rows] * projected_queries[:, None, :]).sum(dim=-1) / math.sqrt(self._config["copy_head"]["rank"])
                if not torch.all(torch.isfinite(scores)):
                    raise ConversationError("Copy attention scores must be finite")
                scores = scores.masked_fill(~allowed, -torch.inf)
                scores = torch.where(available[:, None], scores, torch.zeros_like(scores))
                attention = torch.softmax(scores, dim=-1) * allowed
            else:
                attention = gate_logits.new_zeros((len(rows), 0))
            generator = torch.softmax(vocabulary_logits.float(), dim=-1)
            copy_probabilities = generator.new_zeros(generator.shape).scatter_add(1, source_ids, attention)
            effective_gate = gate * available
            mixture = (1 - effective_gate[:, None]) * generator + effective_gate[:, None] * copy_probabilities
            # The floor is solely for log(0) caused by finite-precision underflow,
            # not a semantic fallback or an invented confidence estimate.
            scores = mixture.clamp_min(torch.finfo(torch.float32).tiny).log()
            scores = torch.where(available[:, None], scores, vocabulary_logits.float())
            scores = scores.to(vocabulary_logits.dtype)
        if not torch.all(torch.isfinite(scores)):
            raise ConversationError("Copy projection must produce finite scores")
        info = None
        if details:
            info = {"attention": attention, "allowed": allowed, "sourceIds": source_ids,
                    "sourcePositions": source_positions, "rows": rows, "times": times,
                    "gateLogits": gate_logits, "gate": gate, "effectiveGate": effective_gate}
        return scores, info

    def copy_training_outputs(self, input_ids, labels):
        """TRAIN-only selected scores plus two unweighted auxiliary objectives.

        Caller supplies the native last-reply label mask; this initial opt-in
        additionally requires the last ASSISTANT marker to remain visible in
        every window. Labels from older replies, USER text, PAD/role targets or
        markerless cropped fragments fail. Inference never calls this method.
        Each whole-context window supports 1..128 last-reply targets including
        EOS, within the existing maximum of 512 context tokens.

        Returns (selected_scores, alignment_loss, gate_loss, detached_stats).
        Alignment averages eligible tokens per window, then eligible windows.
        Gate BCE averages ALL targets per window, then windows; EOS is negative.
        Auxiliary coefficients belong to the trainer, not model inference.
        """
        if "copy_head" not in self._config or not self.training:
            raise ConversationError("Copy auxiliaries require the explicit head in TRAIN mode")
        if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long or input_ids.ndim != 2
                or not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
                or labels.shape != input_ids.shape or labels.device != input_ids.device):
            raise ConversationError("Copy training requires matching long inputs and labels")
        if input_ids.shape[0] < 1 or not 1 <= input_ids.shape[1] <= min(self._config["context_length"], 512):
            raise ConversationError("Copy auxiliary supervision requires a nonempty whole context of at most 512 tokens")
        selected = labels != IGNORE_INDEX
        target_counts = selected.sum(dim=1)
        if not torch.all((target_counts > 0) & (target_counts <= 128)):
            raise ConversationError("Each copy training window requires 1..128 last-reply targets, including EOS")
        targets = labels[selected]
        if torch.any((targets < 0) | (targets >= self._config["vocab_size"])) or torch.any((targets <= ASSISTANT_ID) & (targets != EOS_ID)):
            raise ConversationError("Copy training targets must be payload tokens or EOS")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].expand_as(input_ids)
        last_assistant = torch.where(input_ids == ASSISTANT_ID, positions, -1).max(dim=1).values
        if torch.any(last_assistant < 0) or torch.any(selected & (positions < last_assistant[:, None])):
            raise ConversationError("Copy supervision requires only the last visible ASSISTANT reply")
        last_target = torch.where(selected, positions, -1).max(dim=1).values
        ending = labels.gather(1, last_target[:, None]).flatten()
        complete_span = (positions >= last_assistant[:, None]) & (positions <= last_target[:, None])
        if torch.any(ending != EOS_ID) or not torch.equal(selected, complete_span):
            raise ConversationError("Copy supervision requires a complete contiguous last reply ending in EOS")
        if torch.any((selected & (labels == EOS_ID)).sum(dim=1) != 1):
            raise ConversationError("Copy supervision requires exactly one final EOS target per reply")
        shifted = selected[:, :-1] & (positions[:, :-1] < last_target[:, None])
        if torch.any(shifted & (labels[:, :-1] != input_ids[:, 1:])):
            raise ConversationError("Copy labels must preserve the native next-token shift")
        if torch.any((positions > last_target[:, None]) & (input_ids != PAD_ID)):
            raise ConversationError("Only padding may follow the final supervised reply position")
        # _hidden later checks vocabulary/device/length. Validate role lookup
        # bounds first so malformed inputs cannot fail through an indexing error.
        if torch.any(input_ids < 0) or torch.any(input_ids >= self._config["vocab_size"]):
            raise ConversationError("Copy inputs contain an out-of-vocabulary token")
        roles, known = self._copy_roles(input_ids)
        if torch.any(selected & (~known | (roles != ASSISTANT_ID))):
            raise ConversationError("Copy supervised queries must belong to the last ASSISTANT payload")
        values, info = self._copy_project(input_ids, selected, details=True)
        rows = info["rows"]
        matching = info["allowed"] & (info["sourceIds"] == targets[:, None]) & (targets[:, None] > ASSISTANT_ID)
        eligible = matching.any(dim=1)
        attention = info["attention"]
        with torch.autocast(device_type=values.device.type, enabled=False):
            if torch.any(eligible):
                masses = (attention * matching).sum(dim=1)
                losses = -masses[eligible].clamp_min(torch.finfo(torch.float32).tiny).log()
                sums = losses.new_zeros(input_ids.shape[0]).scatter_add(0, rows[eligible], losses)
                counts = losses.new_zeros(input_ids.shape[0]).scatter_add(0, rows[eligible], torch.ones_like(losses))
                eligible_windows = counts > 0
                alignment = (sums[eligible_windows] / counts[eligible_windows]).mean()
            else:
                alignment = (info["gateLogits"].sum() + attention.sum()) * 0
                eligible_windows = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            token_gate_losses = F.binary_cross_entropy_with_logits(info["gateLogits"], eligible.float(), reduction="none")
            sums = token_gate_losses.new_zeros(input_ids.shape[0]).scatter_add(0, rows, token_gate_losses)
            counts = selected.sum(dim=1)
            gate_loss = (sums / counts).mean()
        if not torch.isfinite(alignment) or not torch.isfinite(gate_loss):
            raise ConversationError("Copy auxiliary objectives must be finite")
        statistics = {"supervisedQueries": int(selected.sum()), "eligibleQueries": int(eligible.sum()),
                      "eligibleWindows": int(eligible_windows.sum()), "batchWindows": input_ids.shape[0],
                      "gatePositiveQueries": int(eligible.sum()), "gateNegativeQueries": int((~eligible).sum()),
                      "matchingPositions": int(matching.sum()), "meanGate": float(info["gate"].detach().mean()),
                      "meanEffectiveGate": float(info["effectiveGate"].detach().mean()),
                      "queryCoverage": int(eligible.sum()) / targets.numel(),
                      "windowCoverage": int(eligible_windows.sum()) / input_ids.shape[0],
                      "supervisionScope": "last_visible_assistant_full_context_only",
                      "maxWholeContextTokens": 512, "maxReplyTargetTokensIncludingEos": 128}
        return values, alignment, gate_loss, statistics


def _assistant_targets(logits, labels, projection_mode):
    if projection_mode not in ("full", "supervised"):
        raise ConversationError("Unknown vocabulary projection mode")
    dimensions = 3 if projection_mode == "full" else 2
    if not isinstance(logits, torch.Tensor) or logits.ndim != dimensions:
        raise ConversationError("logits must match the vocabulary projection mode")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.ndim != 2:
        raise ConversationError("labels must be a two-dimensional long tensor")
    if labels.device != logits.device:
        raise ConversationError("Labels and logits must use the same device")
    if projection_mode == "full" and labels.shape != logits.shape[:2]:
        raise ConversationError("labels must match batch/sequence logits")
    selected = labels != IGNORE_INDEX
    count = int(selected.sum())
    if count == 0:
        raise ConversationError("A supervised batch must contain assistant targets")
    if projection_mode == "supervised" and logits.shape[0] != count:
        raise ConversationError("Projected logits must match the number of assistant targets")
    targets = labels[selected]
    if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        raise ConversationError("Supervised labels contain out-of-vocabulary targets")
    projected = logits[selected] if projection_mode == "full" else logits
    if not torch.all(torch.isfinite(projected)):
        raise ConversationError("Supervised logits must be finite")
    return projected, targets, selected


def assistant_token_losses(logits, labels, *, projection_mode="full"):
    """One CE per supervised position, in row-major order, without repetition."""
    projected, targets, _ = _assistant_targets(logits, labels, projection_mode)
    return F.cross_entropy(projected, targets, reduction="none")


def assistant_token_loss(logits, labels, loss_reduction="token_mean", *, projection_mode="full"):
    """Mean target CE, optionally giving each supervised window equal weight.

    A sequence is one encoded window, which may cover only part of a long reply.
    Rows without targets do not contribute; an entirely unsupervised batch fails.
    Supervised projection accepts only the selected [targets, vocabulary] logits;
    labels retain their window positions so both reductions have the same meaning.
    """
    if loss_reduction not in ("token_mean", "sequence_mean"):
        raise ConversationError("Unknown assistant loss reduction")
    projected, targets, selected = _assistant_targets(logits, labels, projection_mode)
    if loss_reduction == "token_mean":
        return F.cross_entropy(projected, targets)
    token_losses = F.cross_entropy(projected, targets, reduction="none")
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
