"""
Tests for data consistency between annotated YAML and generated JSONL files.

Ensures that the YAML source of truth and the derived JSONL artifacts contain
the exact same number of samples, labels, and text examples.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

# Search both locations
BASE_DIRS = [Path("src/data"), Path("src/mapped_data")]


def _pick_existing_pairs() -> list[tuple[Path, str, str]]:
    """
    Scan project directories to find all existing YAML/JSONL file pairs to
    test.
    """

    candidates = [
        # in-place
        ("nlu.yaml", "samples.train.jsonl"),
        ("validation.yaml", "samples.validation.jsonl"),
        ("test.yaml", "samples.test.jsonl"),
        # same-dir suffix
        ("nlu.mapped.yaml", "samples.train.mapped.jsonl"),
        ("validation.mapped.yaml", "samples.validation.mapped.jsonl"),
        ("test.mapped.yaml", "samples.test.mapped.jsonl"),
        # out-dir (no suffix, when using --out-dir)
        ("nlu.yaml", "samples.train.jsonl"),
        ("validation.yaml", "samples.validation.jsonl"),
        ("test.yaml", "samples.test.jsonl"),
    ]
    pairs: list[tuple[Path, str, str]] = []
    for base in BASE_DIRS:
        for y, j in candidates:
            ypath, jpath = base / y, base / j
            if ypath.exists() and jpath.exists():
                pairs.append((base, ypath.name, jpath.name))
    return pairs


def _flatten_yaml(yaml_path: Path) -> list[dict[str, Any]]:
    """
    Parse a Rasa NLU YAML file and flatten it into a list of example records.
    """

    doc: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    rows: list[dict[str, Any]] = []
    for item in doc.get("nlu") or []:
        if not isinstance(item, dict):
            continue
        parent: str = str(item.get("intent") or "OTHER")
        sub: str | None = (
            item.get("subintent") if isinstance(item.get("subintent"), str) else None
        )
        examples_block: str = str(item.get("examples") or "")
        examples = [
            ln[2:].strip()
            for ln in examples_block.splitlines()
            if ln.strip().startswith("- ")
        ]
        for t in examples:
            rec: dict[str, Any] = {"text": t, "label": parent}
            if parent == "FEEDBACK" and sub in {"COMPLIMENT", "COMPLAINT"}:
                rec["feedback_subtype"] = sub
            if parent == "SENSITIVE_EXIT" and sub in {"BABY_LOSS", "OPTOUT"}:
                rec["sensitive_exit_subtype"] = sub
            if parent == "OTHER" and sub in {
                "ACCOUNT_UPDATE",
                "INFORMATION_QUERY",
                "CONFIRMATION",
            }:
                rec["other_subtype"] = sub
            rows.append(rec)
    rows.sort(key=lambda r: (str(r["label"]), str(r["text"]).casefold()))
    return rows


def _load_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file into a list."""
    return [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_yaml_and_jsonl_counts_and_labels_match() -> None:
    """
    Verify that YAML and JSONL files are consistent in content.

    Checks for every pair of YAML/JSONL files found that:
        - The total number of examples is identical.
        - The counts for each parent label (i.e. intent) are identical.
        - The set of all example texts is identical.
    """

    pairs = _pick_existing_pairs()
    if not pairs:
        pytest.skip(
            "No YAML/JSONL artifacts present; run `make datasets` (with suffix or out-dir)."
        )

    for base, yml_name, jsn_name in pairs:
        yml_path = base / yml_name
        jsn_path = base / jsn_name

        y_rows = _flatten_yaml(yml_path)
        j_rows = _load_jsonl(jsn_path)

        assert len(y_rows) == len(j_rows), f"count mismatch: {yml_path} vs {jsn_path}"
        assert Counter(r["label"] for r in y_rows) == Counter(
            r["label"] for r in j_rows
        ), f"label mix mismatch: {yml_path} vs {jsn_path}"
        assert {r["text"] for r in y_rows} == {
            r["text"] for r in j_rows
        }, f"text set mismatch: {yml_path} vs {jsn_path}"
