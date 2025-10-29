from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import Node

# ===== Config =====
DEFAULT_FILES = ["nlu.yaml", "test.yaml", "validation.yaml"]
PARENTS: set[str] = {"FEEDBACK", "SENSITIVE_EXIT", "NOISE_SPAM", "OTHER"}
MAPPING_VERSION = "2025-09-29-OTHER-v2"

# Legacy intent -> (parent, subintent|None), case-insensitive keys
REROUTE_LOWER: dict[str, tuple[str, str | None]] = {
    # SENSITIVE_EXIT
    "baby loss": ("SENSITIVE_EXIT", "BABY_LOSS"),
    "opt out": ("SENSITIVE_EXIT", "OPTOUT"),
    # FEEDBACK
    "facility complaint": ("FEEDBACK", "COMPLAINT"),
    "facility compliment": ("FEEDBACK", "COMPLIMENT"),
    "chatbot complaint": ("FEEDBACK", "COMPLAINT"),
    "chatbot compliment": ("FEEDBACK", "COMPLIMENT"),
    # NOISE/SPAM
    "spam": ("NOISE_SPAM", None),
    # OTHER → ACCOUNT_UPDATE
    "language": ("OTHER", "ACCOUNT_UPDATE"),
    "personal data update": ("OTHER", "ACCOUNT_UPDATE"),
    "channel switch": ("OTHER", "ACCOUNT_UPDATE"),
    "switch to postbirth": ("OTHER", "ACCOUNT_UPDATE"),
    # OTHER → INFORMATION_QUERY
    "clinic appointment enquiry": ("OTHER", "INFORMATION_QUERY"),
    "general pregnancy enquiry": ("OTHER", "INFORMATION_QUERY"),
    "baby development enquiry": ("OTHER", "INFORMATION_QUERY"),
    "pmtct": ("OTHER", "INFORMATION_QUERY"),
    # OTHER → CONFIRMATION
    "affirm": ("OTHER", "CONFIRMATION"),
}

# Allowed subintents per parent (NOISE_SPAM intentionally empty)
ALLOWED_SUBINTENTS: dict[str, set[str]] = {
    "FEEDBACK": {"COMPLIMENT", "COMPLAINT"},
    "SENSITIVE_EXIT": {"BABY_LOSS", "OPTOUT"},
    "OTHER": {"ACCOUNT_UPDATE", "INFORMATION_QUERY", "CONFIRMATION"},
    "NOISE_SPAM": set(),
}

# JSONL subtype field per parent
SUBTYPE_FIELD: dict[str, str | None] = {
    "FEEDBACK": "feedback_subtype",
    "SENSITIVE_EXIT": "sensitive_exit_subtype",
    "OTHER": "other_subtype",
    "NOISE_SPAM": None,
}


# ===== YAML dumper forcing literal blocks =====
class _LiteralDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, data: str) -> Node:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _represent_str)


# ===== utils =====
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def backup_then_write(path: Path, text: str) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    print(f"[ok] wrote {path.name} (backup {bak.name})")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    print(f"[ok] wrote {path}")


def map_intent(legacy_intent: str) -> tuple[str, str | None]:
    return REROUTE_LOWER.get(legacy_intent.casefold(), ("OTHER", None))


def _examples_from_block(block: str) -> list[str]:
    return [
        ln[2:].strip()
        for ln in (block or "").splitlines()
        if ln.strip().startswith("- ")
    ]


def _examples_block_from_list(examples: list[str]) -> str:
    # Trailing newline prevents PyYAML from emitting "|-"
    return "\n".join(f"- {t}" for t in examples) + ("\n" if examples else "")


def make_dest_paths(
    src: Path, out_dir: Path | None, out_suffix: str
) -> tuple[Path, Path, Path]:
    base_dir = out_dir if out_dir else src.parent
    stem = src.stem

    dest_yaml = base_dir / f"{stem}{out_suffix}{src.suffix}"

    if "nlu" in stem:
        jsonl_base = "samples.train"
    elif "validation" in stem:
        jsonl_base = "samples.validation"
    elif "test" in stem:
        jsonl_base = "samples.test"
    else:
        jsonl_base = stem

    dest_jsonl = base_dir / f"{jsonl_base}{out_suffix}.jsonl"
    dest_meta = base_dir / f"{jsonl_base}{out_suffix}.meta.json"
    return dest_yaml, dest_jsonl, dest_meta


# ===== main workers =====


def yaml_to_jsonl(path: Path, out_dir: Path | None, out_suffix: str) -> None:
    """Non-destructively read a mapped YAML and write only JSONL and meta files."""
    print(f"Generating JSONL for {path.name}...")
    raw = path.read_text(encoding="utf-8")
    doc: dict[str, Any] = yaml.safe_load(raw) or {}
    nlu_items = doc.get("nlu", [])
    samples: list[dict[str, Any]] = []

    for item in nlu_items:
        if not isinstance(item, dict):
            continue
        parent = item.get("intent", "OTHER")
        sub = item.get("subintent")
        examples = _examples_from_block(str(item.get("examples", "")))
        subtype_field = SUBTYPE_FIELD.get(parent)
        for ex in examples:
            row: dict[str, Any] = {"text": ex, "label": parent}
            if subtype_field and sub:
                row[subtype_field] = sub
            samples.append(row)

    _, dest_jsonl, dest_meta = make_dest_paths(path, out_dir, out_suffix)
    jsonl_text = "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + (
        "\n" if samples else ""
    )
    atomic_write_text(dest_jsonl, jsonl_text)

    counts = {lbl: sum(1 for s in samples if s["label"] == lbl) for lbl in PARENTS}
    meta: dict[str, Any] = {
        "source_file": path.name,
        "source_sha256": sha256_text(raw),
        "mapping_version": MAPPING_VERSION,
        "counts": counts,
        "num_samples": len(samples),
    }
    atomic_write_text(dest_meta, json.dumps(meta, indent=2))


def annotate_file(
    path: Path,
    emit_jsonl: bool,
    out_dir: Path | None = None,
    out_suffix: str = "",
) -> None:
    """Map legacy intents and write YAML, JSONL, and meta files."""
    raw = path.read_text(encoding="utf-8")
    doc: dict[str, Any] = yaml.safe_load(raw) or {}
    if "nlu" not in doc:
        print(f"[skip] {path.name} not rasa format")
        return
    nlu_items = doc.get("nlu", [])

    updated: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for item in nlu_items:
        if not isinstance(item, dict):
            continue
        legacy_intent = str(
            item.get("legacy_intent") or item.get("intent") or ""
        ).strip()
        parent, sub = map_intent(legacy_intent)
        examples = _examples_from_block(str(item.get("examples", "")))
        examples_block = _examples_block_from_list(examples)

        annotated: dict[str, Any] = {"intent": parent}
        if sub and sub in ALLOWED_SUBINTENTS.get(parent, set()):
            annotated["subintent"] = sub
        annotated["legacy_intent"] = legacy_intent
        annotated["examples"] = examples_block
        updated.append(annotated)

        if emit_jsonl:
            subtype_field = SUBTYPE_FIELD.get(parent)
            for ex in examples:
                row: dict[str, Any] = {"text": ex, "label": parent}
                if subtype_field and sub:
                    row[subtype_field] = sub
                samples.append(row)

    dest_yaml, dest_jsonl, dest_meta = make_dest_paths(path, out_dir, out_suffix)

    body = yaml.dump(
        {"nlu": updated}, sort_keys=False, allow_unicode=True, Dumper=_LiteralDumper
    )
    out_text = 'version: "3.1"\n' + body
    if dest_yaml.resolve() == path.resolve() and not out_suffix:
        backup_then_write(path, out_text)
    else:
        atomic_write_text(dest_yaml, out_text)

    if emit_jsonl:
        jsonl_text = "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + (
            "\n" if samples else ""
        )
        atomic_write_text(dest_jsonl, jsonl_text)
        counts = {lbl: sum(1 for s in samples if s["label"] == lbl) for lbl in PARENTS}
        meta: dict[str, Any] = {
            "source_file": path.name,
            "source_sha256": sha256_text(raw),
            "mapping_version": MAPPING_VERSION,
            "counts": counts,
            "num_samples": len(samples),
        }
        atomic_write_text(dest_meta, json.dumps(meta, indent=2))


# Back-compat shim for older tests/tools
def process_file(
    path: Path,
    mode: str = "annotate",
    emit_jsonl: bool = True,
    out_dir: Path | None = None,
    out_suffix: str = "",
) -> None:
    if mode != "annotate":
        raise ValueError(f"unsupported mode: {mode!r}")
    annotate_file(path, emit_jsonl=emit_jsonl, out_dir=out_dir, out_suffix=out_suffix)


# ===== CLI =====
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("src/data"))
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--emit-jsonl", action="store_true")
    ap.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Only generate JSONL from mapped YAMLs.",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="Suffix for output files, e.g. .mapped (default: in-place).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write outputs (default: alongside source).",
    )
    args = ap.parse_args()

    for name in args.files:
        src = args.data_dir / name
        if not src.exists():
            print(f"[skip] {name} not found")
            continue

        if args.jsonl_only:
            yaml_to_jsonl(src, out_dir=args.out_dir, out_suffix=args.out_suffix)
        else:
            annotate_file(
                src, args.emit_jsonl, out_dir=args.out_dir, out_suffix=args.out_suffix
            )


if __name__ == "__main__":
    main()
