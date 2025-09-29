# src/config/constants.py
"""Single source of truth for labels, patterns, and normalisation settings."""

from __future__ import annotations

import re
from typing import Final

# ===== Label space (parents + subintents) =====

PARENTS: Final[list[str]] = ["FEEDBACK", "SENSITIVE_EXIT", "NOISE_SPAM", "OTHER"]

FEEDBACK_SUBTYPES: Final[list[str]] = ["COMPLIMENT", "COMPLAINT"]
SENSITIVE_EXIT_SUBTYPES: Final[list[str]] = ["BABY_LOSS", "OPTOUT"]
OTHER_SUBTYPES: Final[list[str]] = [
    "ACCOUNT_UPDATE",
    "INFORMATION_QUERY",
    "CONFIRMATION",
]

FAMILY_SUBTYPES: Final[dict[str, list[str]]] = {
    "FEEDBACK": FEEDBACK_SUBTYPES,
    "SENSITIVE_EXIT": SENSITIVE_EXIT_SUBTYPES,
    "OTHER": OTHER_SUBTYPES,
    "NOISE_SPAM": [],
}

# ===== Metadata =====

# Propagated into metadata for traceability
MAPPING_VERSION: Final[str] = "2025-09-29-OTHER-v2"

# ===== Normalisation config =====
# Let src/utils/normalise.py read these so all components behave the same.

NORMALISE_LOWERCASE: Final[bool] = True
NORMALISE_STRIP_WHITESPACE: Final[bool] = True
NORMALISE_UNICODE_NFKC: Final[bool] = True
ABBREV_EXPANSIONS: Final[dict[str, str]] = {
    "im": "i am",
    "ive": "i have",
    "ur": "your",
    "u": "you",
    "r": "are",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "tnx": "thanks",
}

# ===== Heuristic patterns (pre-routing / data QA) =====
# These **do not** set labels; they're hints for routing, QA, or active learning.

# Bereavement (Baby Loss) — broad but conservative.
BEREAVEMENT_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(stillbirth|stillborn)\b", re.I),
    re.compile(r"\b(miscarriag(e|ed|ing))\b", re.I),
    re.compile(r"\b(lost|lose|losing)\s+(my|our)\s+(baby|child)\b", re.I),
    re.compile(r"\b(my|our)\s+baby\s+(passed|is\s+gone|died)\b", re.I),
    re.compile(r"\b(baby|child)\s+(didn[']?t\s+make it|no longer with us)\b", re.I),
]

# Opt-out — common verbs and short commands
OPTOUT_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(stop|cancel|end)\s+(msgs|messages|sms|texts)\b", re.I),
    re.compile(r"\b(unsub(scribe)?|opt\s?out)\b", re.I),
    re.compile(r"\b(don[']?t)\s+send\b", re.I),
]
