"""Request-local KV cache; no parameters, cross-request state or training changes."""

import torch
from torch.nn import functional as F

from .tokenizer import ConversationError


class GenerationCache:
    def __init__(self, model):
        self.model = model
        self.ids = None
        self.layers = None
        self.hidden = None
        self.prefills = 0
        self.incremental_steps = 0

    def last_logits(self, input_ids):
        model = self.model
        if model.training or torch.is_grad_enabled():
            raise ConversationError("Generation cache requires evaluation and disabled gradients")
        if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long
                or not 1 <= input_ids.shape[1] <= model._config["context_length"]
                or input_ids.device != model.token_embedding.weight.device
                or torch.any(input_ids < 0) or torch.any(input_ids >= model._config["vocab_size"])):
            raise ConversationError("Generation cache requires one valid context on the model device")
        incremental = (self.ids is not None and input_ids.shape[1] == self.ids.shape[1] + 1
                       and torch.equal(input_ids[:, :-1], self.ids))
        # Absolute learned positions restart when the sliding window moves. Old
        # hidden states then have different positions/context and MUST be rebuilt.
        if incremental:
            ids = input_ids[:, -1:]
            positions = torch.tensor([input_ids.shape[1] - 1], device=ids.device)
            past = self.layers
            self.incremental_steps += 1
        else:
            ids = input_ids
            positions = torch.arange(ids.shape[1], device=ids.device)
            past = [None] * len(model.blocks)
            self.prefills += 1
        value = model.embedding_dropout(model.token_embedding(ids) + model.position_embedding(positions)[None, :, :])
        layers = []
        for block, previous in zip(model.blocks, past):
            attended, current = block.attention._attend(block.attention_norm(value), previous)
            value = value + attended
            value = value + block.feedforward(block.feedforward_norm(value))
            layers.append(current)
        hidden = model.final_norm(value)
        self.ids = input_ids.clone()
        self.layers = layers
        if "copy_head" in model._config:
            self.hidden = torch.cat((self.hidden, hidden), dim=1) if incremental else hidden
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            mask[:, -1] = True
            return model._copy_project_hidden(input_ids, self.hidden, mask)[0][0]
        return F.linear(hidden[:, -1], model.token_embedding.weight)[0]
