"""
Estimate individual-level treatment effects (CATE = Conditional Average
Treatment Effect) from the trial data using a cross-fitted T-learner.

What a T-learner is
--------------------
Train two separate outcome models -- one on the treated arm, one on the
control arm -- each predicting the outcome from baseline covariates. The
estimated treatment effect for a patient is simply the difference between
what the treated-arm model and the control-arm model would have predicted
for that patient:

    CATE_hat(x) = E[Y | X=x, treated](x) - E[Y | X=x, control](x)

This is the simplest member of the "meta-learner" family used for
heterogeneous treatment effect estimation (see Kunzel et al., 2019,
"Metalearners for estimating heterogeneous treatment effects using
machine learning"). It's a reasonable first step for a trial this size;
more sophisticated methods (causal forests, X-learners, R-learners) trade
a bit of transparency for statistical efficiency.

Why cross-fit ("honest" estimation)
------------------------------------
If we estimate CATE for a patient using a model that was trained on that
same patient's data, the estimate is optimistic -- the model has partly
memorized the patient. That inflates apparent heterogeneity and can make
subgroup discovery "find" a subgroup that's really just overfitting noise.
Cross-fitting splits patients into folds, predicts each fold using a model
trained only on the *other* folds, and stitches the out-of-fold
predictions back together. This is the same idea as "honest" trees in
causal forests: keep effect *estimation* out-of-sample from wherever the
subgroup *discovery* happens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

DEFAULT_RF_KWARGS = dict(n_estimators=500, min_samples_leaf=15, random_state=0, n_jobs=-1)


def _fit_arm_model(X: np.ndarray, y: np.ndarray, **rf_kwargs) -> RandomForestRegressor:
    model = RandomForestRegressor(**{**DEFAULT_RF_KWARGS, **rf_kwargs})
    model.fit(X, y)
    return model


def estimate_cate_t_learner(
    df: pd.DataFrame,
    feature_columns: list[str],
    outcome_col: str = "tumor_change_pct",
    treatment_col: str = "treatment",
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Cross-fitted T-learner CATE estimation.

    Returns a dict with:
      - cate_hat: out-of-fold CATE estimate per patient (what subgroup
        discovery should be run on)
      - model_treated / model_control: models refit on the FULL data,
        useful for scoring new/future patients but NOT used to produce
        cate_hat (that would defeat the point of cross-fitting)
    """
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=float)
    t = df[treatment_col].to_numpy(dtype=int)
    n = len(df)

    cate_hat = np.full(n, np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        train_treated = train_idx[t[train_idx] == 1]
        train_control = train_idx[t[train_idx] == 0]

        m1 = _fit_arm_model(X[train_treated], y[train_treated], random_state=seed)
        m0 = _fit_arm_model(X[train_control], y[train_control], random_state=seed)

        cate_hat[test_idx] = m1.predict(X[test_idx]) - m0.predict(X[test_idx])

    assert not np.isnan(cate_hat).any(), "every patient should get an out-of-fold estimate"

    # Refit on the full dataset too -- this is what you'd actually deploy
    # to score a brand-new patient walking in the door.
    model_treated = _fit_arm_model(X[t == 1], y[t == 1], random_state=seed)
    model_control = _fit_arm_model(X[t == 0], y[t == 0], random_state=seed)

    return {
        "cate_hat": cate_hat,
        "model_treated": model_treated,
        "model_control": model_control,
        "feature_columns": feature_columns,
    }
