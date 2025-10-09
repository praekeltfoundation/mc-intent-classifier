# src/train_model.py
"""Trains the parent classifier and any specialized sub-intent classifiers."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config.constants import PARENTS

# --- Configuration ---
ENCODER_NAME = "BAAI/bge-m3"
MODEL_VERSION = f"{time.strftime('%Y-%m-%d')}-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DEFAULT_DATA_PATH = Path(__file__).parent / "mapped_data/samples.train.jsonl"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_parent_classifier(
    df: pd.DataFrame, encoder: SentenceTransformer, artifacts_dir: Path
) -> None:
    """Trains and serializes the main parent intent classification model."""
    logging.info("Training parent classifier...")
    le = LabelEncoder().fit(PARENTS)
    y = le.transform(df["label"])
    X = df["text"].tolist()

    logging.info("Encoding text data for parent model...")
    X_embeddings = encoder.encode(X, show_progress_bar=True)

    clf = LogisticRegression(
        class_weight="balanced", random_state=42, C=1.0, solver="liblinear"
    )
    clf.fit(X_embeddings, y)

    joblib.dump(clf, artifacts_dir / "clf_parent.pkl")
    joblib.dump(le, artifacts_dir / "le_parent.pkl")
    logging.info("Parent classifier and label encoder saved.")


def train_sub_intent_classifier(
    df: pd.DataFrame,
    encoder: SentenceTransformer,
    artifacts_dir: Path,
    parent_intent: str,
    sub_intent_col: str,
) -> list[str]:
    """Trains a specialized classifier for a specific parent's sub-intents."""
    logging.info(f"Training specialized sub-intent classifier for '{parent_intent}'...")
    sub_df = df[df["label"] == parent_intent].dropna(subset=[sub_intent_col])

    if len(sub_df[sub_intent_col].unique()) < 2:
        logging.warning(
            f"Skipping sub-intent model for '{parent_intent}': not enough unique data points."
        )
        return []

    le_sub = LabelEncoder().fit(sub_df[sub_intent_col])
    y_sub = le_sub.transform(sub_df[sub_intent_col])
    X_sub = sub_df["text"].tolist()

    logging.info(f"Encoding text data for '{parent_intent}' sub-model...")
    X_sub_embeddings = encoder.encode(X_sub, show_progress_bar=True)

    clf_sub = LogisticRegression(class_weight="balanced", random_state=42)
    clf_sub.fit(X_sub_embeddings, y_sub)

    joblib.dump(clf_sub, artifacts_dir / f"clf_{parent_intent.lower()}.pkl")
    joblib.dump(le_sub, artifacts_dir / f"le_{parent_intent.lower()}.pkl")
    logging.info(f"Sub-intent model for '{parent_intent}' saved.")
    return cast(list[str], le_sub.classes_.tolist())


def main(data_path: Path, artifacts_dir: Path) -> None:
    """Main function to run the complete training pipeline."""
    logging.info(f"Starting model training with version: {MODEL_VERSION}")
    artifacts_dir.mkdir(exist_ok=True)

    # 1. Load data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    df = pd.read_json(data_path, lines=True)
    df = df.dropna(subset=["text", "label"])
    logging.info(f"Loaded {len(df)} samples from {data_path}.")

    # 2. Load encoder (the most resource-intensive part)
    logging.info(f"Loading sentence encoder: {ENCODER_NAME}")
    encoder = SentenceTransformer(ENCODER_NAME)

    # 3. Train models
    train_parent_classifier(df, encoder, artifacts_dir)
    sensitive_exit_labels = train_sub_intent_classifier(
        df, encoder, artifacts_dir, "SENSITIVE_EXIT", "sensitive_exit_subtype"
    )

    # 4. Create and save manifest
    manifest: dict[str, Any] = {
        "version": MODEL_VERSION,
        "encoder": ENCODER_NAME,
        "parent_labels": PARENTS,
        "sub_intent_labels": {"SENSITIVE_EXIT": sensitive_exit_labels},
        "training_data": str(data_path.name),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info(f"Manifest saved to {manifest_path}")

    logging.info("✅ Model training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the intent classifier model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()
    main(data_path=args.data_path, artifacts_dir=args.artifacts_dir)
