"""Bounded, teacher-forced character likelihood under the existing own RNN.

This diagnostic neither generates nor edits candidate text. Its training-domain
likelihood is not a check of grammar, facts, intent or conversational quality.
"""

import numpy as np

from ..conversation.contract import MAX_MESSAGE_BYTES
from .model import LanguageError, WEIGHT_NAMES, encode, loss_and_gradients, validate_artifact


MAX_SCORING_CHARACTERS = 512


def score_text(artifact, text, *, max_chars=MAX_SCORING_CHARACTERS):
    """Score an original text prefix with the RNN's next-character objective.

    The first character initializes the recurrent state; characters 2..N are
    teacher-forced targets, without an invented BOS or EOS. A one-character
    candidate has no target and explicitly abstains. Unknown coverage includes
    the entire scored prefix, including its first input character.
    """
    if type(max_chars) is not int or not 1 <= max_chars <= MAX_SCORING_CHARACTERS:
        raise LanguageError("max_chars must be an integer from 1 to 512")
    if type(text) is not str or not 0 < len(text) <= MAX_MESSAGE_BYTES:
        raise LanguageError("Candidate text must be nonempty bounded text")
    try:
        size = len(text.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise LanguageError("Candidate text must be valid UTF-8") from error
    if size > MAX_MESSAGE_BYTES:
        raise LanguageError("Candidate text exceeds the conversation byte limit")
    validate_artifact(artifact)
    original_length = len(text)
    prefix = text[:max_chars]
    encoded = encode(prefix, artifact["tokenizer"])
    targets = len(encoded) - 1
    result = {
        "modelId": artifact["modelId"], "modelType": artifact["modelType"],
        "artifactSha256": artifact["artifactSha256"], "meanNll": None,
        "targetChars": targets, "unknownRatio": float(np.count_nonzero(encoded == 0) / len(encoded)),
        "charLength": original_length, "scoredCharacters": len(encoded),
        "truncated": original_length > len(encoded), "scoreable": targets > 0,
        "teacherForced": True, "includesEos": False, "executionBackend": "numpy",
        "automaticLearning": False, "parametersUpdated": False,
        "provenance": {"input": "explicit original text candidate", "selection": "first max_chars characters",
            "objective": "next-character negative log likelihood in nats",
            "initialState": "zero", "firstCharacterScored": False,
            "unknownRatioBasis": "all scored prefix characters including initial input",
            "factualAccuracyValidated": False, "grammarQualityValidated": False,
            "generalConversationValidated": False},
    }
    if targets == 0:
        result["reason"] = "no_next_character_target"
        return result
    parameters = {name: np.asarray(artifact["weights"][name], dtype=np.float64)
                  for name in WEIGHT_NAMES}
    result["meanNll"], _ = loss_and_gradients(parameters, encoded[:-1], encoded[1:], gradients=False)
    return result
