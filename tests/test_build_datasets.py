"""Test the dataset annotation and intent mapping functions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.data.build_datasets import annotate_file, map_intent

LEGACY_16 = [
    "Baby Loss",
    "Opt out",
    "Facility Compliment",
    "Facility Complaint",
    "Chatbot Compliment",
    "Chatbot Complaint",
    "Spam",
    "Affirm",
    "Baby Development Enquiry",
    "General Pregnancy Enquiry",
    "Language",
    "Channel Switch",
    "Personal Data Update",
    "PMTCT",
    "Switch to Postbirth",
    "Clinic Appointment Enquiry",
]

OTHER_SUBS = {"ACCOUNT_UPDATE", "INFORMATION_QUERY", "CONFIRMATION"}


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Provide a temporary, isolated data directory for tests."""
    d = tmp_path / "src" / "data"
    d.mkdir(parents=True, exist_ok=True)

    # Clean up after test completes
    def cleanup() -> None:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    request.addfinalizer(cleanup)
    return d


def _seed_yaml(intent: str, example: str) -> str:
    """Create a minimal Rasa NLU YAML string for testing."""
    return yaml.safe_dump(
        {
            "version": "3.1",
            "nlu": [{"intent": intent, "examples": f"|\n  - {example}"}],
        },
        sort_keys=False,
        allow_unicode=True,
    )


@pytest.mark.parametrize("intent", LEGACY_16)
def test_parent_as_intent_subintent_order_legacy_and_jsonl(
    tmp_data_dir: Path, intent: str
) -> None:
    """Verify `annotate_file` correctly maps a legacy intent in-place.

    Checks for correct parent/sub-intent structure, `legacy_intent`
    preservation, correct JSONL output, and that the process is
    **stable if run again**.
    """

    src = tmp_data_dir / "nlu.yaml"
    example = f"{intent} example"
    src.write_text(_seed_yaml(intent, example), encoding="utf-8")

    annotate_file(src, emit_jsonl=True)  # in-place

    out: dict[str, Any] = yaml.safe_load(src.read_text(encoding="utf-8"))
    rec: dict[str, Any] = out["nlu"][0]

    parent, sub = map_intent(intent)
    assert rec["intent"] == parent
    assert "parent" not in rec

    # subintent directly after intent when applicable
    keys = list(rec.keys())
    if parent in {"FEEDBACK", "SENSITIVE_EXIT"} and sub:
        assert keys[:2] == ["intent", "subintent"]
        assert rec["subintent"] == sub
    elif parent == "OTHER":
        if sub:
            assert sub in OTHER_SUBS
            assert keys[:2] == ["intent", "subintent"]
            assert rec["subintent"] == sub
        else:
            assert "subintent" not in rec
    else:
        assert "subintent" not in rec

    # legacy_intent preserved
    assert rec.get("legacy_intent") == intent

    # examples literal block with "- " lines
    examples_block = rec["examples"]
    assert isinstance(examples_block, str)
    assert examples_block.strip().startswith("- ")
    assert "\n\n" not in examples_block

    # JSONL exists + contains correct label and subtype when applicable
    jsonl = tmp_data_dir / "samples.train.jsonl"
    rows = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(r["label"] == parent for r in rows)
    if parent == "FEEDBACK" and sub:
        assert any(r.get("feedback_subtype") == sub for r in rows)
    if parent == "SENSITIVE_EXIT" and sub:
        assert any(r.get("sensitive_exit_subtype") == sub for r in rows)
    if parent == "OTHER" and sub:
        assert any(r.get("other_subtype") == sub for r in rows)

    # Idempotent
    before = src.read_text(encoding="utf-8")
    annotate_file(src, emit_jsonl=False)
    after = src.read_text(encoding="utf-8")
    assert before == after


def test_non_destructive_suffix_outputs(tmp_data_dir: Path) -> None:
    """
    Verify using `out_suffix` creates new files and preserves the original.
    """

    src = tmp_data_dir / "nlu.yaml"
    src.write_text(_seed_yaml("Opt out", "STOP all messages"), encoding="utf-8")

    annotate_file(src, emit_jsonl=True, out_suffix=".mapped")

    # source unchanged
    orig: dict[str, Any] = yaml.safe_load(src.read_text(encoding="utf-8"))
    assert orig["nlu"][0]["intent"] == "Opt out"

    mapped_yaml = tmp_data_dir / "nlu.mapped.yaml"
    mapped: dict[str, Any] = yaml.safe_load(mapped_yaml.read_text(encoding="utf-8"))
    rec = mapped["nlu"][0]
    assert rec["intent"] in {"SENSITIVE_EXIT", "FEEDBACK", "NOISE_SPAM", "OTHER"}
    assert rec.get("legacy_intent") == "Opt out"
    assert (tmp_data_dir / "samples.train.mapped.jsonl").exists()
    assert (tmp_data_dir / "samples.train.mapped.meta.json").exists()
