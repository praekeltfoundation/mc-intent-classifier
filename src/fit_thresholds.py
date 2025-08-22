import argparse
from pathlib import Path

import numpy as np
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
    Initialise or update threshold config for intent classification.

    Policy (DoH + SxD agreed):
    - Baby Loss / Opt-out (sensitive_exit): lower threshold (0.30)
      → prioritise recall, accept more false positives to avoid missing real cases.
    - Service Feedback (complaints/compliments): higher threshold (0.60)
      → prioritise precision, avoid overloading staff with false positives.
    - Noise/Spam: strict (0.75)
      → only accept if highly confident, don't discard real messages too easily.
    - Other: balanced (0.50)
      → fallback catch-all.
    - Review band: 0.40
      → messages in 0.30-0.50 range can be flagged for human review.
    """
    clf = EnsembleClassifier(artifacts_dir=model_dir)

    texts, labels = load_yaml_examples(validation_path)

    # Initialise collections
    scores_by_intent: dict[str, list[float]] = {label.value: [] for label in IntentEnum}
    true_by_intent: dict[str, list[int]] = {label.value: [] for label in IntentEnum}

    # Collect classifier scores
    for text, label in zip(texts, labels):
        # Replicate scoring logic from EnsembleClassifier.predict
        norm_text = normalise_text(text)
        emb = clf.encoder.encode([norm_text], show_progress_bar=False)
        clf_probs = clf.clf.predict_proba(emb)[0]
        scores = {
            intent.value: prob for intent, prob in zip(list(IntentEnum), clf_probs)
        }

        for intent, score in scores.items():
            scores_by_intent[intent].append(score)
            true_by_intent[intent].append(1 if label == intent else 0)

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
        review_band=0.40,  # fixed for now
    )

    # Save thresholds
    out_path = model_dir / "thresholds.json"
    out_path.write_text(thresholds.model_dump_json(indent=2))
    print(f"✅ Thresholds tuned and saved at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--nlu-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    args = parser.parse_args()

    main(args.model_dir, args.nlu_path, args.validation_path)
