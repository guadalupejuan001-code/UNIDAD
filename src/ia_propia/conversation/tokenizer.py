"""Train-only local byte-level BPE and explicit dialogue supervision."""

from copy import deepcopy
import math

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .contract import (ASSISTANT_ID, BOS_ID, DEFAULT_SYSTEM_PROMPT, EOS_ID, IGNORE_INDEX,
                       MAX_DIALOGUE_BYTES, MAX_MESSAGE_BYTES, MAX_MESSAGES, PAD_ID, ROLE_IDS,
                       SPECIAL_TOKENS, SYSTEM_ID, USER_ID)

MAX_CONTEXT_LENGTH = 4096
MAX_VOCAB_SIZE = 16_384


class ConversationError(ValueError):
    """Invalid conversational configuration, input, or portable artifact."""


def integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConversationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def number(value, name, minimum=0.0, maximum=2.0):
    if type(value) not in (int, float):
        raise ConversationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ConversationError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ConversationError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    return result


def validate_messages(messages):
    """Accept text and explicit roles; text cannot introduce control boundaries.

    A supplied initial system message is conditioning text, never tool authority.
    The returned snapshot does not alias the caller's messages.
    """
    if type(messages) is not list or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ConversationError(f"messages must contain 1 to {MAX_MESSAGES} explicit turns")
    total = 0
    for index, message in enumerate(messages):
        if type(message) is not dict or set(message) != {"role", "content"}:
            raise ConversationError("Each message must contain only role and content")
        role, text = message["role"], message["content"]
        if type(role) is not str or role not in ROLE_IDS:
            raise ConversationError("Message role must be system, user or assistant")
        if role == "system" and index != 0:
            raise ConversationError("A system message may appear only at the start")
        if type(text) is not str or not text.strip():
            raise ConversationError("Message content must be nonempty explicit text")
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ConversationError("Message content must be valid UTF-8") from error
        if size > MAX_MESSAGE_BYTES:
            raise ConversationError("Message content exceeds the UTF-8 byte limit")
        total += size
        if total > MAX_DIALOGUE_BYTES:
            raise ConversationError("Dialogue exceeds the UTF-8 byte limit")
        if any(token in text for token in SPECIAL_TOKENS):
            raise ConversationError("Message text cannot contain reserved dialogue control tokens")
    if messages[0]["role"] != "system":
        if len(messages) >= MAX_MESSAGES or total + len(DEFAULT_SYSTEM_PROMPT.encode("utf-8")) > MAX_DIALOGUE_BYTES:
            raise ConversationError("Dialogue must leave room for its default system message")
    return deepcopy(messages)


def prepare_messages(messages):
    result = validate_messages(messages)
    if result[0]["role"] != "system":
        result.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    return validate_messages(result)


def validate_tokenizer(tokenizer, *, strict=False):
    if not isinstance(tokenizer, Tokenizer):
        raise ConversationError("Expected a local tokenizers.Tokenizer")
    if (not isinstance(tokenizer.model, models.BPE) or not isinstance(tokenizer.pre_tokenizer, pre_tokenizers.ByteLevel)
            or not isinstance(tokenizer.decoder, decoders.ByteLevel) or tokenizer.normalizer is not None
            or tokenizer.post_processor is not None or tokenizer.padding is not None or tokenizer.truncation is not None
            or tokenizer.model.dropout not in (None, 0.0) or tokenizer.model.unk_token is not None):
        raise ConversationError("Tokenizer must use deterministic unpadded byte-level BPE without hidden processing")
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(token) != expected_id:
            raise ConversationError("Tokenizer special-token IDs do not match the dialogue contract")
    integer(tokenizer.get_vocab_size(), "vocabulary size", 262, MAX_VOCAB_SIZE)
    if strict:
        vocabulary = tokenizer.get_vocab()
        if set(vocabulary.values()) != set(range(len(vocabulary))):
            raise ConversationError("Tokenizer vocabulary IDs must be contiguous")
        if any(character not in vocabulary for character in pre_tokenizers.ByteLevel.alphabet()):
            raise ConversationError("Tokenizer must retain the complete byte alphabet")
    return tokenizer


def _record_messages(record):
    if type(record) is dict and "messages" in record:
        return record["messages"]
    if type(record) is list:
        return record
    raise ConversationError("Training records must contain explicit messages")


def fit_tokenizer(train_records, vocab_size=2048):
    """Fit BPE using only the caller's training records; no downloads or defaults.

    The default system prompt is included once per training dialogue, exactly as
    it is at inference time when the caller omits an explicit system turn.
    """
    integer(vocab_size, "vocab_size", 262, MAX_VOCAB_SIZE)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=list(SPECIAL_TOKENS),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    record_count = 0

    def texts():
        nonlocal record_count
        for record in train_records:
            messages = prepare_messages(_record_messages(record))
            record_count += 1
            for message in messages:
                yield message["content"]

    tokenizer.train_from_iterator(texts(), trainer=trainer)
    if not record_count:
        raise ConversationError("Cannot fit a tokenizer without training records")
    return validate_tokenizer(tokenizer, strict=True)


def fit_text_tokenizer(train_records, vocab_size=8192):
    """Fit own BPE on TRAIN document text only, without role/system wrappers."""
    from .language_data import validate_text_record
    integer(vocab_size, "vocab_size", 262, MAX_VOCAB_SIZE)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=list(SPECIAL_TOKENS),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    count = 0

    def texts():
        nonlocal count
        for record in train_records:
            validate_text_record(record)
            if record["split"] != "train":
                raise ConversationError("Text tokenizer fitting accepts TRAIN documents only")
            count += 1
            yield record["text"]

    tokenizer.train_from_iterator(texts(), trainer=trainer)
    if not count:
        raise ConversationError("Cannot fit a tokenizer without training documents")
    return validate_tokenizer(tokenizer, strict=True)


def text_tokens(text, tokenizer):
    """Document BOS + byte-preserving content + EOS, without system/role text."""
    # Corpus validation handles document limits; this helper also bounds direct
    # callers and prevents content from injecting special-token boundaries.
    from .language_data import validate_text_record
    validate_text_record({"id": "encode", "groupId": "encode", "split": "train",
                          "source": {"dataset": "explicit-text", "kind": "internal-validation"}, "text": text})
    validate_tokenizer(tokenizer)
    return [BOS_ID, *tokenizer.encode(text, add_special_tokens=False).ids, EOS_ID]


def _prepared_tokens(messages, tokenizer):
    validate_tokenizer(tokenizer)
    ids, supervised = [BOS_ID], [False]
    for message in messages:
        role = message["role"]
        content = tokenizer.encode(message["content"], add_special_tokens=False).ids
        ids.extend([ROLE_IDS[role], *content, EOS_ID])
        supervised.extend([False, *([role == "assistant"] * len(content)), role == "assistant"])
    return ids, supervised


def dialogue_tokens(messages, tokenizer):
    """Full sequence and same-position assistant mask, before shifting/cropping."""
    return _prepared_tokens(prepare_messages(messages), tokenizer)


def encode_dialogue(messages, tokenizer, context_length):
    """Next-token inputs/labels, supervising assistant text and its EOS only.

    If long, retain the latest context window. For multi-window training the
    trainer can use dialogue_tokens and shift each contiguous chunk explicitly.
    """
    if context_length is not None:
        integer(context_length, "context_length", 2, MAX_CONTEXT_LENGTH)
    ids, supervised = dialogue_tokens(messages, tokenizer)
    inputs = ids[:-1]
    labels = [token if teach else IGNORE_INDEX for token, teach in zip(ids[1:], supervised[1:])]
    if context_length is None:
        return inputs, labels
    return inputs[-context_length:], labels[-context_length:]


def encode_prompt(messages, tokenizer, context_length):
    integer(context_length, "context_length", 2, MAX_CONTEXT_LENGTH)
    prepared = prepare_messages(messages)
    if prepared[-1]["role"] != "user":
        raise ConversationError("Chat messages must end with an explicit user turn")
    ids, _ = _prepared_tokens(prepared, tokenizer)
    ids.append(ASSISTANT_ID)
    return ids[-context_length:], max(0, len(ids) - context_length)
