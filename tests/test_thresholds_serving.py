# tests/test_thresholds_serving.py
"""Tests for the Thresholds class and its logic."""

from __future__ import annotations

from src.config.constants import PARENTS
from src.config.thresholds import Thresholds


def test_for_parent_map() -> None:
    """Ensures for_parent() returns the correct threshold for each parent label."""
    th = Thresholds.defaults()
    assert th.for_parent("FEEDBACK") == th.service_feedback
    assert th.for_parent("SENSITIVE_EXIT") == th.sensitive_exit
    assert th.for_parent("NOISE_SPAM") == th.noise
    assert th.for_parent("OTHER") == th.other


def test_accept_and_review_logic() -> None:
    """Verifies the logic for acceptance and review band checks."""
    # Mypy requires all fields to be explicitly set when not using defaults.
    th = Thresholds(
        service_feedback=0.6,
        sensitive_exit=0.45,  # Using a default value
        other=0.5,  # Using a default value
        noise=0.7,  # Using a default value
        review_band=0.4,
        sentiment_review_band=0.75,
    )

    # Case 1: Score is above acceptance and review band
    probs = [0.65, 0.1, 0.1, 0.15]  # FEEDBACK strongest
    j = probs.index(max(probs))
    label = PARENTS[j]
    score = probs[j]
    assert label == "FEEDBACK"
    assert score >= th.for_parent(label)
    assert not (score < th.review_band)

    # Case 2: Score is below acceptance but above review band
    probs2 = [0.41, 0.40, 0.10, 0.09]
    j2 = probs2.index(max(probs2))
    label2 = PARENTS[j2]
    score2 = probs2[j2]
    assert label2 == "FEEDBACK"
    assert score2 < th.for_parent(label2)
    assert not (score2 < th.review_band)

    # Case 3: Score is below acceptance and review band
    probs3 = [0.35, 0.30, 0.20, 0.15]
    score3 = max(probs3)
    assert score3 < th.review_band
