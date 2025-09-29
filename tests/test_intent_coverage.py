# tests/test_intent_coverage.py
from __future__ import annotations

from pathlib import Path

import yaml

# 🔒 The canonical legacy taxonomy we expect in src/data/nlu.yaml
LEGACY_16 = {
    # Sensitive Exit
    "Baby Loss",
    "Opt out",
    # Feedback
    "Facility Compliment",
    "Facility Complaint",
    "Chatbot Compliment",
    "Chatbot Complaint",
    # Noise/Spam
    "Spam",
    # Other (collapse to OTHER)
    "Affirm",
    "Baby Development Enquiry",
    "General Pregnancy Enquiry",
    "Language",
    "Channel Switch",
    "Personal Data Update",
    "PMTCT",
    "Switch to Postbirth",
    "Clinic Appointment Enquiry",
}


def test_no_unexpected_intents_in_source_nlu_yaml() -> None:
    """
    Guardrail: if any *new* intent appears in src/data/nlu.yaml,
    this test fails until you explicitly add it to LEGACY_16 (and mapping).
    """
    src = Path("src/data/nlu.yaml")
    assert src.exists(), "src/data/nlu.yaml not found"

    doc = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    intents_in_file = {
        str(item.get("intent", "")).strip()
        for item in (doc.get("nlu") or [])
        if isinstance(item, dict)
    }

    # Allow missing ones (e.g., partial files in branches), but forbid *extra* ones.
    unexpected = intents_in_file - LEGACY_16
    assert not unexpected, (
        "New or unexpected intents found in src/data/nlu.yaml:\n"
        f"  {sorted(unexpected)}\n\n"
        "➡️  Add them to LEGACY_16 here AND update the mapping in src/data/build_datasets.py "
        "(REROUTE_LOWER) so they map into one of the four parents."
    )
