"""Neutral-by-default generation controls, separate from learned model weights."""

from .tokenizer import integer, number


def validate_controls(top_k=0, top_p=1.0, repetition_penalty=1.0):
    return {"top_k": integer(top_k, "top_k", 0, 65536),
            "top_p": number(top_p, "top_p", 0.000001, 1),
            "repetition_penalty": number(repetition_penalty, "repetition_penalty", 1, 2)}


def torch_token(logits, forbidden, generated, *, temperature, generator,
                top_k=0, top_p=1.0, repetition_penalty=1.0):
    """Keep the original Torch RNG/sampling path when controls are neutral."""
    import torch
    scores = logits.clone()
    if repetition_penalty != 1 and generated:
        ids = torch.tensor(sorted(set(generated)), device=scores.device)
        values = scores[ids]
        scores[ids] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    scores[forbidden] = -torch.inf
    if temperature == 0:
        return int(torch.argmax(scores).item())
    if top_k or top_p < 1:
        # Stable ties: the smaller token ID comes first on both backends.
        order = torch.argsort(scores, descending=True, stable=True)
        if top_k:
            scores[order[top_k:]] = -torch.inf
        if top_p < 1:
            probabilities = torch.softmax(scores[order] / temperature, dim=-1)
            # Include the token that crosses the nucleus threshold.
            remove = probabilities.cumsum(0) - probabilities >= top_p
            scores[order[remove]] = -torch.inf
    probabilities = torch.softmax(scores / temperature, dim=-1)
    if not torch.all(torch.isfinite(probabilities)):
        from .tokenizer import ConversationError
        raise ConversationError("Generation produced nonfinite probabilities")
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
