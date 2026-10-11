"""Pure conversation constants shared by ingestion, inference and adapters."""

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>")
PAD_ID, BOS_ID, EOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID = range(6)
ROLE_IDS = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}
IGNORE_INDEX = -100
MAX_MESSAGES = 64
MAX_MESSAGE_BYTES = 16_384
MAX_DIALOGUE_BYTES = 65_536
DEFAULT_SYSTEM_PROMPT = (
    "Eres Unidad. Habla en español de forma cercana y de confianza. Relaciona lo que la persona ya ha compartido; "
    "distingue tus inferencias de los hechos y no inventes recuerdos."
)
DEFAULT_MAX_NEW_TOKENS = 96
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 42
MAX_NEW_TOKENS = 1024
