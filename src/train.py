import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.config.constants import IntentEnum
from src.config.thresholds import Thresholds
from src.utils.normalise import normalise_text

LABEL_ORDER = list(IntentEnum)


def load_data(yaml_paths: list[Path]) -> tuple[list[str], list[int]]:
    """Loads and prepares data from YAML files for training."""
    texts, labels = [], []
    for path in yaml_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            for entry in data.get("nlu", []):
                intent = entry.get("intent")
                if not intent:
                    continue
                try:
                    intent_enum = IntentEnum(intent)
                except ValueError:
                    continue
                examples = entry.get("examples", "").splitlines()
                for ex in examples:
                    text = ex.lstrip("-").strip()
                    if text:
                        texts.append(normalise_text(text))
                        labels.append(LABEL_ORDER.index(intent_enum))
    return texts, labels


def main(
    data_dir: Path, artifacts_dir: Path, encoder_name: str = "BAAI/bge-m3"
) -> None:
    """
    Main training function to generate and save the classification model and artifacts.
    """
    # Load & encode data
    texts, labels = load_data(list(data_dir.glob("*.yaml")))
    encoder = SentenceTransformer(encoder_name)
    X = encoder.encode(texts, show_progress_bar=True)
    y = np.array(labels)

    # Train model
    base_clf = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42
    )
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    clf.fit(X, y)

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = artifacts_dir / f"mcic-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save classifier
    joblib.dump(clf, out_dir / "clf.pkl")

    # Save manifest
    manifest = {
        "version": f"mcic-{timestamp}",
        "encoder": encoder_name,
        "labels": [lbl.value for lbl in LABEL_ORDER],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save default thresholds
    thresholds = Thresholds()
    (out_dir / "thresholds.json").write_text(thresholds.model_dump_json(indent=2))

    print(f"✅ Model saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("src/data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("src/artifacts"))
    parser.add_argument("--encoder", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()

    main(args.data_dir, args.artifacts_dir, args.encoder)
