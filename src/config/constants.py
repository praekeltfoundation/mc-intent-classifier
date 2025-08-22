import re
from enum import Enum


class IntentEnum(str, Enum):
    SERVICE_FEEDBACK = "service_feedback"
    SENSITIVE_EXIT = "sensitive_exit"
    OTHER = "other"
    NOISE = "noise"


# Label display names
LABEL_MAP: dict[IntentEnum, str] = {
    IntentEnum.SERVICE_FEEDBACK: "Service Feedback",
    IntentEnum.SENSITIVE_EXIT: "Baby Loss / Opt-out",
    IntentEnum.OTHER: "Other",
    IntentEnum.NOISE: "Noise/Spam",
}


# Abbreviations for normalisation
ABBREV: dict[str, str] = {
    r"\bpls\b": "please",
    r"\bplz\b": "please",
    r"\bthnx\b": "thanks",
    r"\bthx\b": "thanks",
    r"\bmsg(s)?\b": "message",
    r"\bmscarriage\b": "miscarriage",
    r"\bmiskraam\b": "miscarriage",
    r"\bstill\s?birth\b": "stillbirth",
    r"\bstill\s?born\b": "stillborn",
    r"\bu\b": "you",
    r"\br\b": "are",
    r"\bdnt\b": "do not",
}


# Regex for bereavement detection
BEREAVEMENT_PATTERNS: re.Pattern[str] = re.compile(
    r"(lost my baby|miscarriage|stillbirth|stillborn|didn't make it|passed away|angel baby|ngishonelwe|ndiphunyelwe|swelekile|swelekelwe|bhubhile|file|akasekho|ayisekho)",
    re.IGNORECASE,
)
