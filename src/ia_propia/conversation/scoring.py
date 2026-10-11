"""Teacher-forced reply likelihood with the same context as local generation.

Likelihood is a diagnostic, not a measure of conversational competence. Replies
are explicit development candidates, never replacements for generated output.
"""

import math

from .tokenizer import ConversationError, EOS_ID, encode_prompt, validate_messages, validate_tokenizer


def score_reply(model, tokenizer, messages, text):
    """Score every reply token and EOS under the actual sliding chat context.

    Short examples use one causal forward pass. Long examples move the context
    exactly as chat does, rather than resetting it at a disjoint chunk boundary.
    The raw vocabulary distribution matches training loss; generation additionally
    masks role-control tokens. Neither candidates nor parameters are modified.
    """
    import torch
    from torch.nn import functional as F

    validate_tokenizer(tokenizer)
    validate_messages([{"role": "assistant", "content": text}])
    if tokenizer.get_vocab_size() != model.config["vocab_size"]:
        raise ConversationError("Tokenizer vocabulary differs from the model")
    limit = model.config["context_length"]
    prompt, discarded = encode_prompt(messages, tokenizer, limit)
    targets = [*tokenizer.encode(text, add_special_tokens=False).ids, EOS_ID]
    if len(targets) > 1025:
        raise ConversationError("Development candidate exceeds 1024 text tokens")
    device = model.token_embedding.weight.device
    log_probabilities = []
    with model._generation_lock:
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                sequence = [*prompt, *targets]
                if len(sequence) - 1 <= limit:
                    inputs = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
                    logits = model(inputs)[0, len(prompt) - 1:]
                    if not torch.all(torch.isfinite(logits)):
                        raise ConversationError("Reply scoring produced nonfinite logits")
                    labels = torch.tensor(targets, dtype=torch.long, device=device)
                    selected = F.log_softmax(logits.float(), dim=-1).gather(1, labels[:, None])[:, 0]
                    log_probabilities = selected.double().tolist()
                else:
                    context = list(prompt)
                    for token in targets:
                        inputs = torch.tensor([context[-limit:]], dtype=torch.long, device=device)
                        logits = model(inputs)[0, -1].float()
                        if not torch.all(torch.isfinite(logits)):
                            raise ConversationError("Reply scoring produced nonfinite logits")
                        log_probabilities.append(float(F.log_softmax(logits, dim=-1)[token]))
                        context.append(token)
        finally:
            model.train(was_training)
    total = math.fsum(log_probabilities)
    if not math.isfinite(total):
        raise ConversationError("Reply scoring produced a nonfinite probability")
    return {"totalLogProbability": total, "meanLogProbability": total / len(targets),
            "meanNll": -total / len(targets), "targetTokens": len(targets), "includesEos": True,
            "promptTokens": len(prompt), "discardedPromptTokens": discarded,
            "targetsWithSlidingPrompt": max(0, len(prompt) + len(targets) - 1 - limit),
            "teacherForced": True, "distribution": "unmasked_training_vocabulary"}
