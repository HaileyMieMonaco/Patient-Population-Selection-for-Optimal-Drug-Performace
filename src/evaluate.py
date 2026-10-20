"""
Grade the pipeline.

1. Validation against ground truth -- only possible because the trial is
   simulated. Checks whether the discovered subgroup actually lines up
   with the patients who truly benefit (precision/recall) and whether the
   estimated CATE tracks the true CATE (calibration). This is how you'd
   convince yourself the METHOD works before ever trusting it on real
   data, where the ground truth is never available.

2. Analyses you *could* run on a real trial -- the observed average
   treatment effect (ATE) inside vs. outside the discovered subgroup,
   computed the old-fashioned way (difference in arm means, with a
   confidence interval), using only outcomes and treatment assignment,
   never the ground-truth CATE. This is the evidence you'd actually put
   in front of a clinical team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def observed_ate(df: pd.DataFrame, mask: np.ndarray | None = None, outcome_col: str = "tumor_change_pct",
                  treatment_col: str = "treatment", n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference-in-means ATE (treated - control) with a bootstrap 95% CI,
    computed only from observed outcomes and arm assignment -- exactly
    what you'd compute on a real trial. `mask` restricts to a subgroup
    (e.g. the discovered responders); None uses everyone."""
    sub = df if mask is None else df.loc[mask]
    treated = sub.loc[sub[treatment_col] == 1, outcome_col].to_numpy()
    control = sub.loc[sub[treatment_col] == 0, outcome_col].to_numpy()

    point = treated.mean() - control.mean()

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        t_bs = rng.choice(treated, size=len(treated), replace=True)
        c_bs = rng.choice(control, size=len(control), replace=True)
        boots[i] = t_bs.mean() - c_bs.mean()

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(sub),
        "n_treated": len(treated),
        "n_control": len(control),
        "ate": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def subgroup_vs_ground_truth(in_subgroup: np.ndarray, true_responder: np.ndarray) -> dict:
    """How well does the discovered subgroup match the (otherwise hidden)
    true responder population? Only computable because this is simulated
    data with a known answer key."""
    return {
        "precision": float(precision_score(true_responder, in_subgroup, zero_division=0)),
        "recall": float(recall_score(true_responder, in_subgroup, zero_division=0)),
        "f1": float(f1_score(true_responder, in_subgroup, zero_division=0)),
        "n_discovered": int(in_subgroup.sum()),
        "n_true_responders": int(true_responder.sum()),
        "n_overlap": int((in_subgroup & true_responder).sum()),
    }


def cate_calibration(cate_hat: np.ndarray, true_cate: np.ndarray) -> dict:
    """Pearson correlation and mean absolute error between the estimated
    and true individual treatment effects."""
    corr = float(np.corrcoef(cate_hat, true_cate)[0, 1])
    mae = float(np.mean(np.abs(cate_hat - true_cate)))
    return {"pearson_r": corr, "mae": mae}
