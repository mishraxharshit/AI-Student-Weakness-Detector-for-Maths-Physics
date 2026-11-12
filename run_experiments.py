"""Benchmark the three estimators on synthetic data with known ground truth:  python run_experiments.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from weakness import simulate as sim  # noqa: E402
from weakness.estimators import BetaShrinkage, HierarchicalIRT, RawAccuracy  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
N_SIM = 20
OBSERVED = [3, 5, 8]          # quiz questions observed per (student, topic) before judging weakness
FLAG_FRACTION = 0.25          # flag the 25% lowest-scoring (student, topic) pairs as "weak"


def precision_recall_at_flag(truth_weak, score):
    k = int(len(score) * FLAG_FRACTION)
    flagged = np.zeros(len(score), bool)
    flagged[np.argsort(score)[:k]] = True
    tp = (flagged & truth_weak).sum()
    return tp / max(flagged.sum(), 1), tp / max(truth_weak.sum(), 1)


def one_run(seed, n_obs):
    attempts, truth, _ = sim.simulate(seed=seed, attempts_per_topic=n_obs + 5)
    obs = attempts[attempts.attempt_order < n_obs].reset_index(drop=True)
    held = attempts[attempts.attempt_order >= n_obs].reset_index(drop=True)   # unseen questions
    res = {}
    for est in (RawAccuracy(), BetaShrinkage(), HierarchicalIRT()):
        est.fit(obs)
        sc = truth.merge(est.scores(), on=["student_id", "topic"], how="left")
        sc["score"] = sc.score.fillna(sc.score.mean())
        prec, rec = precision_recall_at_flag(sc.is_weak.to_numpy(), sc.score.to_numpy())
        r = {"weak_auc": roc_auc_score(sc.is_weak, -sc.score), "precision": prec, "recall": rec,
             "ability_corr": float(np.corrcoef(sc.score, sc.true_ability)[0, 1])}
        if isinstance(est, HierarchicalIRT):
            p = np.clip(est.predict_proba(held), 1e-6, 1 - 1e-6)
            r["heldout_auc"], r["heldout_logloss"] = roc_auc_score(held.correct, p), log_loss(held.correct, p)
        res[est.name] = r
    # baseline for held-out answer prediction: raw accuracy of the (student, topic) pair
    raw = obs.groupby(["student_id", "topic"]).correct.mean().rename("p").reset_index()
    h = held.merge(raw, on=["student_id", "topic"], how="left")
    pr = np.clip(h.p.fillna(obs.correct.mean()).to_numpy(), 0.05, 0.95)
    res["Raw accuracy"]["heldout_auc"], res["Raw accuracy"]["heldout_logloss"] = roc_auc_score(h.correct, pr), log_loss(h.correct, pr)
    return res


if __name__ == "__main__":
    all_res = {}
    for n_obs in OBSERVED:
        runs = [one_run(seed, n_obs) for seed in range(N_SIM)]
        agg = {}
        for m in runs[0]:
            agg[m] = {k: {"mean": float(np.mean([r[m][k] for r in runs])), "std": float(np.std([r[m][k] for r in runs]))}
                      for k in runs[0][m]}
        all_res[str(n_obs)] = agg
        print(f"n_observed={n_obs}", {m: round(agg[m]["weak_auc"]["mean"], 3) for m in agg}, flush=True)
    (OUT / "results.json").write_text(json.dumps({"n_simulations": N_SIM, "students": 200, "topics": len(sim.TOPICS),
                                                  "flag_fraction": FLAG_FRACTION, "results_by_n_observed": all_res}, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, mk in [("Raw accuracy", "s"), ("Beta shrinkage", "^"), ("Hierarchical IRT", "o")]:
        for ax, metric, ttl in [(axs[0], "weak_auc", "Weak-topic detection (ROC-AUC vs ground truth)"),
                                (axs[1], "precision", "Precision when flagging the lowest 25% of pairs")]:
            ys = [all_res[str(n)][m][metric]["mean"] for n in OBSERVED]
            es = [all_res[str(n)][m][metric]["std"] for n in OBSERVED]
            ax.errorbar(OBSERVED, ys, yerr=es, marker=mk, capsize=3, label=m); ax.set_title(ttl, fontsize=10)
    for ax in axs:
        ax.set(xlabel="questions observed per student per topic", xticks=OBSERVED); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "detection_quality.png", dpi=150); plt.close(fig)

    attempts, truth, _ = sim.simulate(seed=42, attempts_per_topic=10)
    attempts.to_csv(Path(__file__).parent / "data" / "sample_quiz_synthetic.csv", index=False)
    print("saved results + sample data")
