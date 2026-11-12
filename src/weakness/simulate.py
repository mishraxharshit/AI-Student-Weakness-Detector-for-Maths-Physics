"""Synthetic quiz generator with KNOWN ground truth (so detection quality can be measured exactly).

Generative model (a 2-parameter-logistic style IRT model with a hierarchical student ability):
    theta[s,t] = g[s] + u[s,t]          g ~ N(0,1) general ability,  u ~ N(0, sigma_topic^2) topic-specific
    P(correct) = sigmoid( a * (theta[s,t] - b[q]) )     b[q] ~ N(0,1) question difficulty
A (student, topic) pair is TRULY WEAK when theta[s,t] < WEAK_THRESHOLD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOPICS = {
    "Algebra": "Maths", "Calculus": "Maths", "Probability": "Maths",
    "Mechanics": "Physics", "Electricity": "Physics", "Waves & Optics": "Physics",
    "Python Basics": "Python", "Data Structures": "Python",
}
WEAK_THRESHOLD = -0.7
DISCRIMINATION = 1.3


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(n_students: int = 200, questions_per_topic: int = 25, attempts_per_topic: int = 10,
             sigma_topic: float = 0.8, seed: int = 0):
    """Return (attempts DataFrame, truth DataFrame, question difficulty DataFrame)."""
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    g = rng.normal(0, 1, n_students)
    theta = g[:, None] + rng.normal(0, sigma_topic, (n_students, len(topics)))
    b = rng.normal(0, 1, (len(topics), questions_per_topic))

    rows = []
    for s in range(n_students):
        for ti, topic in enumerate(topics):
            qs = rng.choice(questions_per_topic, attempts_per_topic, replace=False)
            for step, q in enumerate(qs):
                p = _sigmoid(DISCRIMINATION * (theta[s, ti] - b[ti, q]))
                rows.append((f"S{s:03d}", topic, f"{topic[:3].upper()}-{q:02d}", step, int(rng.random() < p)))
    attempts = pd.DataFrame(rows, columns=["student_id", "topic", "question_id", "attempt_order", "correct"])
    truth = pd.DataFrame([(f"S{s:03d}", t, theta[s, ti], theta[s, ti] < WEAK_THRESHOLD)
                          for s in range(n_students) for ti, t in enumerate(topics)],
                         columns=["student_id", "topic", "true_ability", "is_weak"])
    qdiff = pd.DataFrame([(f"{t[:3].upper()}-{q:02d}", t, b[ti, q]) for ti, t in enumerate(topics)
                          for q in range(questions_per_topic)], columns=["question_id", "topic", "true_difficulty"])
    return attempts, truth, qdiff


def split_observed_heldout(attempts: pd.DataFrame, n_observed: int):
    """Per (student, topic): the first n_observed attempts are 'observed', the rest are held out for evaluation."""
    obs = attempts[attempts.attempt_order < n_observed].reset_index(drop=True)
    held = attempts[attempts.attempt_order >= 5].reset_index(drop=True)  # fixed held-out block (attempts 5-9)
    return obs, held
