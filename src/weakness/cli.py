"""Command line: python -m weakness.cli --data data/sample_quiz_synthetic.csv --student S007

CSV columns required: student_id, topic, question_id, correct (0/1).
"""
from __future__ import annotations

import argparse

import pandas as pd

from .recommend import Advisor


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weak-topic report and practice plan for one student.")
    ap.add_argument("--data", required=True, help="CSV of quiz attempts")
    ap.add_argument("--student", required=True)
    ap.add_argument("--topics", type=int, default=3, help="how many weakest topics to report")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.data)
    missing = {"student_id", "topic", "question_id", "correct"} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")
    if a.student not in set(df.student_id):
        raise SystemExit(f"student '{a.student}' not found in data")
    adv = Advisor(df)
    print(f"\n=== Weakest topics for {a.student} ===")
    print(adv.weakest_topics(a.student, a.topics).round(2).to_string(index=False))
    print("\n=== Suggested practice questions (aimed at ~70% success) ===")
    print(adv.recommend_questions(a.student, k_topics=a.topics).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
