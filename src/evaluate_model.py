"""Evaluate the model and find optimal confidence thresholds.

This script provides two modes:
1. `tune`: Finds optimal confidence thresholds using a validation set.
2. `report`: Gives a final, unbiased performance report on a test set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from transformers import AutoTokenizer, TextClassificationPipeline, pipeline

from src.config.thresholds import Thresholds

# --- Configuration ---
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EVAL_DIR = Path(__file__).parent / "evaluations"
DEFAULT_DATA_PATH = Path(__file__).parent / "mapped_data/samples.validation.jsonl"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
EVAL_DIR.mkdir(exist_ok=True)


class MockEnsemble:
    """A mock of the EnsembleClassifier for evaluation purposes."""

    def __init__(self, artifacts_dir: Path):
        self.encoder: SentenceTransformer
        self.clf_sensitive_exit: ClassifierMixin | None
        self.le_sensitive_exit: LabelEncoder | None
        self.sentiment_pipeline: TextClassificationPipeline
        (
            self.encoder,
            self.clf_sensitive_exit,
            self.le_sensitive_exit,
            self.sentiment_pipeline,
        ) = self._load_sub_models(artifacts_dir)

    def _load_sub_models(
        self, artifacts_dir: Path
    ) -> tuple[
        SentenceTransformer,
        ClassifierMixin | None,
        LabelEncoder | None,
        TextClassificationPipeline,
    ]:
        manifest = json.loads((artifacts_dir / "manifest.json").read_text())
        encoder = SentenceTransformer(manifest["encoder"])

        clf_se_path = artifacts_dir / "clf_sensitive_exit.pkl"
        le_se_path = artifacts_dir / "le_sensitive_exit.pkl"

        clf_se = joblib.load(clf_se_path) if clf_se_path.exists() else None
        le_se = joblib.load(le_se_path) if le_se_path.exists() else None

        sentiment_pipe = pipeline(
            task="text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer=AutoTokenizer.from_pretrained(  # type: ignore
                "cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False
            ),
        )
        return encoder, clf_se, le_se, sentiment_pipe

    def enrich(self, text: str, parent_label: str) -> str | None:
        """Simulates the enrichment process to get a sub-intent prediction."""
        if parent_label == "FEEDBACK":
            sentiment = self.sentiment_pipeline(text, top_k=1)[0]
            return "COMPLIMENT" if sentiment["label"] == "positive" else "COMPLAINT"
        if (
            parent_label == "SENSITIVE_EXIT"
            and self.clf_sensitive_exit
            and self.le_sensitive_exit
        ):
            embedding = self.encoder.encode([text])
            pred_idx = self.clf_sensitive_exit.predict(embedding)[0]
            return cast(str, self.le_sensitive_exit.inverse_transform([pred_idx])[0])
        return None


def _load_main_artifacts(
    artifacts_dir: Path,
) -> tuple[SentenceTransformer, ClassifierMixin, LabelEncoder]:
    """Load the main parent classifier artifacts."""
    logging.info("Loading parent model artifacts from %s...", artifacts_dir)
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    encoder = SentenceTransformer(manifest["encoder"])
    clf = joblib.load(artifacts_dir / "clf_parent.pkl")
    le = joblib.load(artifacts_dir / "le_parent.pkl")
    return encoder, clf, le


def _get_predictions(
    df: pd.DataFrame, encoder: SentenceTransformer, clf: ClassifierMixin
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Get model probability scores and return true labels in a DataFrame."""
    logging.info("Encoding text data and getting predictions...")
    X = df["text"].tolist()
    y_probs = clf.predict_proba(encoder.encode(X, show_progress_bar=True))
    return df.copy(), y_probs


def _find_optimal_thresholds(
    y_true: list[str], y_probs: NDArray[np.float64], le: LabelEncoder
) -> tuple[dict[str, float], str]:
    """Find the best threshold for each actionable parent class."""
    logging.info("Finding optimal thresholds for primary intents...")
    thresholds: dict[str, float] = {}
    report_data = []

    for i, class_name in enumerate(le.classes_):
        if class_name == "OTHER":
            thresholds[class_name] = 1.0
            continue

        precision, recall, class_thresholds = precision_recall_curve(
            y_true, y_probs[:, i], pos_label=class_name
        )
        if len(class_thresholds) == 0:
            thresholds[class_name] = 0.5
            report_data.append([class_name, "N/A", "N/A", 0.5, "Default (No Samples)"])
            continue

        f1_scores = np.nan_to_num((2 * precision * recall) / (precision + recall))
        if class_name == "SENSITIVE_EXIT":
            target_recall = 0.99
            high_recall_indices = np.where(recall >= target_recall)[0]
            best_idx = (
                high_recall_indices[0]
                if len(high_recall_indices) > 0
                else np.argmax(f1_scores)
            )
            strategy = (
                f"Recall >= {target_recall}"
                if len(high_recall_indices) > 0
                else "F1-Max (Fallback)"
            )
        else:
            best_idx = np.argmax(f1_scores)
            strategy = "F1-Max"

        optimal_threshold = class_thresholds[best_idx]
        thresholds[class_name] = round(float(optimal_threshold), 4)
        report_data.append(
            [
                class_name,
                f"{precision[best_idx]:.4f}",
                f"{recall[best_idx]:.4f}",
                f"{thresholds[class_name]:.4f}",
                strategy,
            ]
        )

    headers = ["Intent", "Precision", "Recall", "Optimal Threshold", "Strategy"]
    return thresholds, tabulate(report_data, headers=headers, tablefmt="grid")


def _get_predicted_parent_labels(
    y_probs: NDArray[np.float64], le: LabelEncoder, thresholds: Thresholds
) -> list[str]:
    """Generate final predicted parent labels using the 'default-to-OTHER' logic."""
    parent_pred_labels = []
    for prob_vector in y_probs:
        confident_preds = [
            (le.classes_[i], prob_vector[i])
            for i, class_name in enumerate(le.classes_)
            if class_name != "OTHER"
            and prob_vector[i] >= thresholds.for_parent(class_name)
        ]
        if confident_preds:
            parent_pred_labels.append(max(confident_preds, key=lambda item: item[1])[0])
        else:
            parent_pred_labels.append("OTHER")
    return parent_pred_labels


def _generate_hierarchical_report(
    true_df: pd.DataFrame,
    y_pred_parent: list[str],
    mock_ensemble: MockEnsemble,
) -> str:
    """Generate a two-level report for parent and sub-intent performance."""
    parent_report = classification_report(
        true_df["label"], y_pred_parent, zero_division=0
    )

    sub_intent_reports = []
    true_df["predicted_parent"] = y_pred_parent

    for parent_name, sub_col in [
        ("FEEDBACK", "feedback_subtype"),
        ("SENSITIVE_EXIT", "sensitive_exit_subtype"),
    ]:
        sub_df = true_df[true_df["label"] == parent_name].copy()
        if not sub_df.empty:
            sub_df["predicted_sub"] = [
                mock_ensemble.enrich(text, parent)
                for text, parent in zip(sub_df["text"], sub_df["predicted_parent"])
            ]
            y_true_sub = sub_df[sub_col].fillna("None").tolist()
            y_pred_sub = sub_df["predicted_sub"].fillna("None").tolist()
            sub_report = classification_report(y_true_sub, y_pred_sub, zero_division=0)
            sub_intent_reports.append(
                f"--- Sub-Intent Performance: {parent_name} ---\n\n{sub_report}"
            )

    full_report = f"--- Parent-Level Performance ---\n\n{parent_report}\n\n"
    full_report += "\n".join(sub_intent_reports)
    return full_report


def _save_error_analysis_artifacts(
    y_true_parent: list[str],
    y_pred_parent: list[str],
    texts: list[str],
    out_dir: Path,
    report_suffix: str,
) -> None:
    """Generate and save detailed error analysis artifacts for parent labels."""
    logging.info("Generating error analysis artifacts for parent-level predictions...")
    all_labels = sorted(set(y_true_parent) | set(y_pred_parent))

    cm = confusion_matrix(y_true_parent, y_pred_parent, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap="Blues", xticks_rotation="vertical", ax=ax)
    plt.tight_layout()
    cm_path = out_dir / f"confusion_matrix{report_suffix}.png"
    plt.savefig(cm_path)
    plt.close()
    logging.info("Saved confusion matrix to %s", cm_path)

    misclassified_samples = [
        {"text": text, "true_label": true, "predicted_label": pred}
        for text, true, pred in zip(texts, y_true_parent, y_pred_parent)
        if true != pred
    ]
    if misclassified_samples:
        errors_path = out_dir / f"misclassified_samples{report_suffix}.json"
        errors_path.write_text(json.dumps(misclassified_samples, indent=2))
        logging.info(
            "Saved %d misclassified samples to %s",
            len(misclassified_samples),
            errors_path,
        )


def plot_precision_recall_curves(
    y_true: list[str],
    y_probs: NDArray[np.float64],
    le: LabelEncoder,
    out_dir: Path,
) -> None:
    """Generate and save precision-recall curve plots for each class."""
    logging.info("Generating precision-recall curve plots for each class...")
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(le.classes_):
        if class_name == "OTHER":
            continue
        precision, recall, _ = precision_recall_curve(
            y_true, y_probs[:, i], pos_label=class_name
        )
        plt.plot(recall, precision, lw=2, label=f"PR curve for {class_name}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for each Intent")
    plt.legend(loc="best")
    plt.grid(True)
    plot_path = out_dir / "precision_recall_curves.png"
    plt.savefig(plot_path)
    logging.info("Saved PR curve plot to %s", plot_path)
    plt.close()


def tune_and_evaluate(artifacts_dir: Path, data_path: Path, save_plots: bool) -> None:
    """Full pipeline to tune thresholds and evaluate the model."""
    encoder, clf, le = _load_main_artifacts(artifacts_dir)
    df = pd.read_json(data_path, lines=True)
    true_df, y_probs = _get_predictions(df, encoder, clf)

    threshold_values, threshold_report = _find_optimal_thresholds(
        true_df["label"].tolist(), y_probs, le
    )
    final_thresholds = {
        "review_band": round(
            np.mean([v for k, v in threshold_values.items() if k != "OTHER"]), 4
        ),
        "per_parent": threshold_values,
    }
    thresholds_path = artifacts_dir / "thresholds.json"
    thresholds_path.write_text(json.dumps(final_thresholds, indent=2))
    logging.info("Saved optimal thresholds to %s", thresholds_path)

    thresholds = Thresholds.model_validate(final_thresholds)
    y_pred_parent = _get_predicted_parent_labels(y_probs, le, thresholds)

    mock_ensemble = MockEnsemble(artifacts_dir)
    final_report = _generate_hierarchical_report(true_df, y_pred_parent, mock_ensemble)

    full_report = (
        f"--- Optimal Threshold Report ---\n\n{threshold_report}\n\n"
        f"--- Hierarchical Performance Report ---\n\n{final_report}"
    )
    print(full_report)

    report_path = EVAL_DIR / "evaluation_report.txt"
    report_path.write_text(full_report)
    logging.info("Saved full evaluation report to %s", report_path)

    _save_error_analysis_artifacts(
        true_df["label"].tolist(),
        y_pred_parent,
        true_df["text"].tolist(),
        EVAL_DIR,
        "_tuning",
    )

    if save_plots:
        plot_precision_recall_curves(true_df["label"].tolist(), y_probs, le, EVAL_DIR)

    logging.info("All evaluation artifacts saved under %s", EVAL_DIR)


def report_only(artifacts_dir: Path, data_path: Path) -> None:
    """Runs the model against a test set with pre-tuned thresholds."""
    thresholds_path = artifacts_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(
            "Thresholds file not found. Run 'tune-thresholds' first."
        )

    thresholds = Thresholds.model_validate(json.loads(thresholds_path.read_text()))
    encoder, clf, le = _load_main_artifacts(artifacts_dir)
    df = pd.read_json(data_path, lines=True)
    true_df, y_probs = _get_predictions(df, encoder, clf)

    y_pred_parent = _get_predicted_parent_labels(y_probs, le, thresholds)
    mock_ensemble = MockEnsemble(artifacts_dir)
    final_report = _generate_hierarchical_report(true_df, y_pred_parent, mock_ensemble)

    full_report = (
        f"--- Final Hold-Out Test Set Performance Report ---\n\n{final_report}"
    )
    print(full_report)

    report_path = EVAL_DIR / "final_performance_report.txt"
    report_path.write_text(full_report)
    logging.info("Saved final performance report to %s", report_path)

    _save_error_analysis_artifacts(
        true_df["label"].tolist(),
        y_pred_parent,
        true_df["text"].tolist(),
        EVAL_DIR,
        "_final",
    )


def main() -> None:
    """Main function to parse args and run the correct mode."""
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    tune_parser = subparsers.add_parser(
        "tune", help="Tune thresholds on a validation set."
    )
    tune_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    tune_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    tune_parser.add_argument("--save-plots", action="store_true")

    report_parser = subparsers.add_parser(
        "report", help="Report performance on a test set."
    )
    report_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    report_parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).parent / "mapped_data/samples.test.jsonl",
    )

    args = parser.parse_args()

    if args.mode == "tune":
        tune_and_evaluate(args.artifacts_dir, args.data_path, args.save_plots)
    elif args.mode == "report":
        report_only(args.artifacts_dir, args.data_path)


if __name__ == "__main__":
    main()
