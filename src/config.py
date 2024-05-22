import logging

BEST_MODEL_FOLDER = "best-model"
OUTPUT_FOLDER = "output"
EMOTIONS_FILE = "Emotions"
BINARY_TRIGGERS_FILE = "BinaryTriggers"
NUMERICAL_TRIGGERS_FILE = "NumericalTriggers"
FILE_EXTENSION = ".tsv"

SYSTEM_PROMPT = """You are an emotion detection classifier."""

PROMPT = """Predict one emotion label for the given text. The possible labels are: "Love", "Joy", "Anger", "Fear", "Sadness", "Neutral".

Answer in one following format: "Label: <emotion_label>".

"""

LANGUAGES = ["es", "fr", "nl", "ru"]

LLM_LABEL_TEXT = "Label:"

WANDB_ENTITY = "nlp-a-cross-sentiment"  # WandB entity name
WANDB_PROJECT_NAME: str = "wassa2024"  # WandB project name


def init_logging() -> None:
    """Initialize logging."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


RANDOM_SEED = 1998
