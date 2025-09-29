# src/train_model.py
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config.constants import PARENTS

# --- Configuration ---
# Match the model used in the ensemble classifier.
ENCODER_NAME = "BAAI/bge-m3"
MODEL_VERSION = f"{time.strftime('%Y-%m-%d')}-v1"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DEFAULT_DATA_PATH = Path(__file__).parent / "mapped_data/samples.train.jsonl"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_classifier(
    data_path: Path = DEFAULT_DATA_PATH, artifacts_dir: Path = ARTIFACTS_DIR
) -> None:
    """Trains and serializes the intent classification model."""
    logging.info(f"Starting model training with version: {MODEL_VERSION}")

    # 1. Load and prepare data
    if not data_path.exists():
        logging.error(f"Data file not found at: {data_path}")
        raise FileNotFoundError(f"Required data file is missing: {data_path}")

    df = pd.read_json(data_path, lines=True)
    df = df[["text", "label"]].dropna()

    if len(df) == 0:
        logging.error("No data available to train the model.")
        return

    logging.info(f"Loaded {len(df)} samples from {data_path}.")

    # Encode labels
    le = LabelEncoder().fit(PARENTS)
    y = le.transform(df["label"])
    X = df["text"].tolist()

    # 2. Encode text into embeddings
    logging.info(f"Loading sentence encoder: {ENCODER_NAME}")
    encoder = SentenceTransformer(ENCODER_NAME)
    logging.info("Encoding text data into embeddings... (this may take a moment)")
    X_embeddings = encoder.encode(X, show_progress_bar=True)

    # 3. Train the classifier
    # Using class_weight='balanced' to handle potential label imbalance.
    clf = LogisticRegression(
        class_weight="balanced", random_state=42, C=1.0, solver="liblinear"
    )
    logging.info("Training logistic regression classifier...")
    clf.fit(X_embeddings, y)

    # 4. Save artifacts
    artifacts_dir.mkdir(exist_ok=True)

    # Save classifier
    classifier_path = artifacts_dir / "clf.pkl"
    joblib.dump(clf, classifier_path)
    logging.info(f"Classifier saved to {classifier_path}")

    # Save label encoder
    le_path = artifacts_dir / "label_encoder.pkl"
    joblib.dump(le, le_path)
    logging.info(f"Label encoder saved to {le_path}")

    # Create and save manifest
    manifest: dict[str, Any] = {
        "version": MODEL_VERSION,
        "encoder": ENCODER_NAME,
        "parent_labels": le.classes_.tolist(),
        "training_data": str(data_path.name),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info(f"Manifest saved to {manifest_path}")

    logging.info("✅ Model training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the intent classifier model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the training data JSONL file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory to save model artifacts.",
    )
    args = parser.parse_args()
    train_classifier(data_path=args.data_path, artifacts_dir=args.artifacts_dir)
