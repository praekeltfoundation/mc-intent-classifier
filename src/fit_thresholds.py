import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_fscore_support

from src.classifiers.ensemble import EnsembleClassifier
from src.config.constants import IntentEnum
from src.config.thresholds import Thresholds
from src.utils.normalise import normalise_text


def load_yaml_examples(validation_path: Path) -> tuple[list[str], list[str]]:
    """Parse Rasa-style YAML and flatten into list of (text, intent)."""
    with open(validation_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    examples, intents = [], []
    for item in data.get("nlu", []):
        intent = item["intent"]
        raw_examples = item["examples"].split("\n")
        for ex in raw_examples:
            ex = ex.strip().lstrip("-").strip()
            if ex:  # skip blanks
                examples.append(normalise_text(ex))
                intents.append(intent)
    return examples, intents


def evaluate_thresholds(y_true: list[int], y_scores: list[float], policy: str) -> float:
    """Sweep thresholds for one intent, return best cutoff by policy bias."""
    best_t, best_metric = 0.5, -1.0
    for t in np.arange(0.2, 0.91, 0.05):
        y_pred = [1 if score >= t else 0 for score in y_scores]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        metric = {
            "precision": precision,
            "recall": recall,
            "balance": f1,
        }[policy]

        if metric > best_metric:
            best_metric, best_t = metric, float(t)

    return round(float(best_t), 2)


def main(model_dir: Path, nlu_path: Path, validation_path: Path) -> None:
    """
    Tune thresholds + review band from validation data.

    Policies:
    - sensitive_exit → recall priority
    - service_feedback → precision priority
    - noise → precision priority
    - other → balance
    - review_band → derived dynamically from margin distribution
    """
    clf = EnsembleClassifier(artifacts_dir=model_dir)

    texts, labels = load_yaml_examples(validation_path)

    # Initialise collections
    scores_by_intent: dict[str, list[float]] = {label.value: [] for label in IntentEnum}
    true_by_intent: dict[str, list[int]] = {label.value: [] for label in IntentEnum}

    # For review band derivation
    margins: list[float] = []
    review_candidates: list[dict[str, str | float]] = []

    # Collect classifier scores
    for text, label in zip(texts, labels):
        norm_text = normalise_text(text)
        emb = clf.encoder.encode([norm_text], show_progress_bar=False)
        clf_probs = clf.clf.predict_proba(emb)[0]
        scores = {
            intent.value: prob for intent, prob in zip(list(IntentEnum), clf_probs)
        }

        # Populate per-intent evaluation buckets
        for intent, score in scores.items():
            scores_by_intent[intent].append(score)
            true_by_intent[intent].append(1 if label == intent else 0)

        # Top-2 margin for review band
        sorted_probs = sorted(clf_probs, reverse=True)
        top, second = sorted_probs[0], sorted_probs[1]
        pred = list(IntentEnum)[int(np.argmax(clf_probs))].value

        if pred != label:
            margins.append(top - second)

        # Record borderline examples
        review_candidates.append(
            {
                "text": text,
                "true_label": label,
                "predicted": pred,
                "top_prob": round(float(top), 3),
                "second_prob": round(float(second), 3),
                "margin": round(float(top - second), 3),
            }
        )

    # Derive review band
    if margins:
        review_margin = float(np.percentile(margins, 75))
        print(f"📊 Derived review margin = {review_margin:.2f}")
    else:
        review_margin = 0.10  # fallback if no errors

    # Sweep thresholds with policy bias
    thresholds = Thresholds(
        service_feedback=evaluate_thresholds(
            true_by_intent["service_feedback"],
            scores_by_intent["service_feedback"],
            policy="precision",
        ),
        sensitive_exit=evaluate_thresholds(
            true_by_intent["sensitive_exit"],
            scores_by_intent["sensitive_exit"],
            policy="recall",
        ),
        noise=evaluate_thresholds(
            true_by_intent["noise"],
            scores_by_intent["noise"],
            policy="precision",
        ),
        other=evaluate_thresholds(
            true_by_intent["other"],
            scores_by_intent["other"],
            policy="balance",
        ),
        review_band=review_margin,
    )

    # Save thresholds JSON
    out_path = model_dir / "thresholds.json"
    out_path.write_text(thresholds.model_dump_json(indent=2))
    print(f"✅ Thresholds tuned and saved at {out_path}")

    # Save review candidates CSV
    df = pd.DataFrame(review_candidates)
    csv_path = model_dir / "review_candidates.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Review candidates saved at {csv_path}")

    # Print summary
    print("\n=== Thresholds Summary ===")
    print(json.dumps(thresholds.model_dump(), indent=2))
    print("==========================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nlu-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    args = parser.parse_args()

    main(args.model_dir, args.nlu_path, args.validation_path)
