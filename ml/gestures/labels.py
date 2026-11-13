"""Single source of truth for the five static Kannada gesture classes."""

GESTURE_LABELS = (
    "ನಮಸ್ಕಾರ",
    "ಹೌದು",
    "ಇಲ್ಲ",
    "ಸರಿ",
    "ಮುಗಿದಿದೆ",
)

GESTURE_NAMES = ("namaskara", "haudu", "illa", "sari", "mugidide")
GESTURE_ENGLISH = ("Namaskara (Open Palm)", "Haudu (Thumbs Up)", "Illa (Thumbs Down)", "Sari (OK Sign)", "Mugidide (Finished)")

LABEL_TO_INDEX = {label: index for index, label in enumerate(GESTURE_LABELS)}
NAME_TO_INDEX = {name: index for index, name in enumerate(GESTURE_NAMES)}


def label_for_class(class_index: int) -> str:
    if not 0 <= class_index < len(GESTURE_LABELS):
        raise ValueError(f"Unknown five-gesture class index: {class_index}")
    return GESTURE_LABELS[class_index]


def english_label_for_class(class_index: int) -> str:
    if not 0 <= class_index < len(GESTURE_ENGLISH):
        raise ValueError(f"Unknown five-gesture class index: {class_index}")
    return GESTURE_ENGLISH[class_index]
