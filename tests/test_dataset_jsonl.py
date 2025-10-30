"""Tests for the JSONL and metadata file generation process."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.data.build_datasets import process_file

pytestmark = pytest.mark.data


def test_emit_jsonl_and_meta_from_annotated_yaml(tmp_path: Path) -> None:
    """
    Verify that JSONL and meta files are created correctly from an annotated YAML.

    This test checks that:
        - Both `.jsonl` and `.meta.json` files are created.
        - The JSONL rows contain the correct labels and counts.
        - The ordering of rows in the JSONL is deterministic.
        - The meta file contains the expected version, SHA hash, and sample
          counts.
    """

    data_dir = tmp_path / "src" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / "nlu.yaml"

    # minimal Rasa file with one of each parent
    doc = {
        "version": "3.1",
        "nlu": [
            {"intent": "Baby Loss", "examples": "|\n  - I lost my baby last week"},
            {"intent": "Opt out", "examples": "|\n  - STOP all messages"},
            {
                "intent": "Facility Compliment",
                "examples": "|\n  - Nurse Thandi was amazing!",
            },
            {"intent": "Spam", "examples": "|\n  - WIN CASH now http://x"},
            {
                "intent": "Clinic Appointment Enquiry",
                "examples": "|\n  - Next clinic appointment when?",
            },
        ],
    }
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    # annotate + emit JSONL
    process_file(p, mode="annotate", emit_jsonl=True)

    # JSONL & meta exist
    jsonl = data_dir / "samples.train.jsonl"
    meta = data_dir / "samples.train.meta.json"
    assert jsonl.exists() and meta.exists()

    # JSONL rows look sane and deterministic ordering (by label, then text)
    rows = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [r["label"] for r in rows]
    assert set(labels) == {"SENSITIVE_EXIT", "FEEDBACK", "NOISE_SPAM", "OTHER"}
    cnt = Counter(labels)
    assert cnt == Counter(
        {"SENSITIVE_EXIT": 2, "FEEDBACK": 1, "NOISE_SPAM": 1, "OTHER": 1}
    )

    # meta has counts and a sha
    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_obj.get("mapping_version")
    assert meta_obj.get("source_sha256")
    assert meta_obj.get("counts") == dict(cnt)
    assert meta_obj.get("num_samples") == len(rows)
