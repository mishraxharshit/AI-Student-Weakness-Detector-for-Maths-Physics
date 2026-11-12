"""Turn fitted estimates into a practice plan for one student."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import HierarchicalIRT
from .simulate import TOPICS

TARGET_SUCCESS = 0.70   # practise questions the student gets right ~70 % of the time ("desirable difficulty")


class Advisor:
    def __init__(self, attempts: pd.DataFrame, **irt_kwargs):
        self.attempts = attempts
        self.model = HierarchicalIRT(**irt_kwargs).fit(attempts)
        self.scores = self.model.scores()
        qt = attempts.drop_duplicates("question_id").set_index("question_id").topic
        self.question_topic = qt

    def weakest_topics(self, student_id: str, k: int = 3) -> pd.DataFrame:
        s = self.scores[self.scores.student_id == student_id].copy()
        s["weakness_rank_score"] = -s.score
        # how much evidence: number of attempts per topic
        n = self.attempts[self.attempts.student_id == student_id].groupby("topic").size()
        s["n_attempts"] = s.topic.map(n).fillna(0).astype(int)
        acc = self.attempts[self.attempts.student_id == student_id].groupby("topic").correct.mean()
        s["raw_accuracy"] = s.topic.map(acc)
        return s.sort_values("score").head(k)[["topic", "score", "raw_accuracy", "n_attempts"]].reset_index(drop=True)

    def recommend_questions(self, student_id: str, n_per_topic: int = 3, k_topics: int = 3) -> pd.DataFrame:
        done = set(self.attempts[self.attempts.student_id == student_id].question_id)
        diff = self.model.difficulty()
        out = []
        for _, row in self.weakest_topics(student_id, k_topics).iterrows():
            qids = [q for q in diff.index if self.question_topic.get(q) == row.topic and q not in done]
            if not qids:
                continue
            ability = self.scores.query("student_id == @student_id and topic == @row.topic").score.iloc[0]
            # model: logit p = ability - difficulty -> pick questions whose predicted p is closest to TARGET_SUCCESS
            p = 1 / (1 + np.exp(-(ability - diff[qids].to_numpy())))
            order = np.argsort(np.abs(p - TARGET_SUCCESS))[:n_per_topic]
            out += [(row.topic, TOPICS.get(row.topic, ""), qids[i], float(p[i])) for i in order]
        return pd.DataFrame(out, columns=["topic", "subject", "question_id", "predicted_success"])
