"""
Simulate a randomized controlled trial (RCT) of a fictional targeted therapy
in advanced non-small-cell lung cancer (NSCLC), with a KNOWN ground-truth
responder subgroup built into the data-generating process.

Clinical framing
-----------------
Drug: a fictional EGFR/immune-pathway-targeted agent, "ONCB-317", tested
head-to-head against standard-of-care chemotherapy in advanced NSCLC.
Primary endpoint: percent change in tumor burden (sum of target lesion
diameters) at 12 weeks, a standard RECIST-style efficacy readout. Negative
values mean the tumor shrank.

Two kinds of baseline variables are simulated on purpose, because
distinguishing them is the whole point of a good stratification analysis:

* PROGNOSTIC variables affect the outcome regardless of which arm a
  patient is in (e.g. sicker patients do worse whether or not they get the
  drug). Age, ECOG performance status, baseline tumor burden, comorbidity
  count and tumor mutational burden play this role here.

* PREDICTIVE (moderator) variables change the SIZE of the treatment
  effect itself. Here, only two variables are predictive: EGFR mutation
  status and PD-L1 expression. Patients who are EGFR-mutant AND have high
  PD-L1 expression get a large extra benefit from the drug; everyone else
  gets, at most, a small background benefit. This is deliberately modeled
  as a smooth function of PD-L1 (a sigmoid), not a hard cutoff, so that
  "discovering" the subgroup requires approximating a threshold rather
  than just recovering a literal rule we wrote down.

Run this file directly to regenerate data/trial_data.csv (the "observed"
trial dataset, as an analyst would actually see it) and
data/ground_truth.json (the hidden truth, used only for validating the
pipeline in evaluate.py -- never used as a model input).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20260917  # fixed for reproducibility but an arbitrary choice

FEATURE_COLUMNS = [
    "age",
    "sex_male",
    "ecog_ps",
    "smoking_current",
    "smoking_former",
    "histology_squamous",
    "baseline_tumor_burden_mm",
    "comorbidity_index",
    "prior_lines_of_therapy",
    "tumor_mutational_burden",
    "egfr_mutation",
    "pdl1_expression_pct",
]

# Clinically meaningful "extra benefit" threshold (percentage points of
# additional tumor shrinkage from the drug) used to define a "true
# responder" for evaluation purposes only.
RESPONDER_CATE_THRESHOLD = -20.0


def simulate(n_patients: int = 1200, seed: int = RNG_SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = n_patients

    # ---- Baseline covariates (measured before randomization) ----
    age = rng.normal(64, 10, n).clip(35, 89)
    sex_male = rng.binomial(1, 0.52, n)
    ecog_ps = rng.choice([0, 1, 2], size=n, p=[0.45, 0.42, 0.13])
    smoking = rng.choice(["never", "former", "current"], size=n, p=[0.15, 0.55, 0.30])
    smoking_current = (smoking == "current").astype(int)
    smoking_former = (smoking == "former").astype(int)
    histology_squamous = rng.binomial(1, 0.30, n)
    baseline_tumor_burden_mm = rng.gamma(shape=6.0, scale=12.0, size=n)
    comorbidity_index = rng.poisson(1.4, n)
    prior_lines_of_therapy = rng.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])
    tumor_mutational_burden = rng.gamma(shape=3.0, scale=3.0, size=n)

    # Predictive biomarkers (treatment-effect modifiers)
    egfr_mutation = rng.binomial(1, 0.22, n)
    pdl1_expression_pct = (rng.beta(1.4, 3.0, n) * 100)

    # ---- Randomization: independent of every covariate (this is an RCT) ----
    treatment = rng.binomial(1, 0.5, n)

    # ---- Ground-truth outcome model (never seen by the modeling code) ----
    prognostic = (
        0.06 * baseline_tumor_burden_mm
        + 6.0 * ecog_ps
        + 2.0 * comorbidity_index
        + 0.15 * (age - 64)
        - 1.5 * tumor_mutational_burden
    )

    base_treatment_effect = -6.0  # small benefit for essentially everyone
    pdl1_signal = 1.0 / (1.0 + np.exp(-(pdl1_expression_pct - 50.0) / 8.0))
    predictive_effect = -42.0 * egfr_mutation * pdl1_signal
    true_cate = base_treatment_effect + predictive_effect

    noise = rng.normal(0, 9.0, n)
    tumor_change_pct = 5.0 + prognostic + treatment * true_cate + noise
    tumor_change_pct = tumor_change_pct.clip(-90, 60)
    objective_response = (tumor_change_pct <= -30).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(n)],
            "age": age,
            "sex_male": sex_male,
            "ecog_ps": ecog_ps,
            "smoking_current": smoking_current,
            "smoking_former": smoking_former,
            "histology_squamous": histology_squamous,
            "baseline_tumor_burden_mm": baseline_tumor_burden_mm,
            "comorbidity_index": comorbidity_index,
            "prior_lines_of_therapy": prior_lines_of_therapy,
            "tumor_mutational_burden": tumor_mutational_burden,
            "egfr_mutation": egfr_mutation,
            "pdl1_expression_pct": pdl1_expression_pct,
            "treatment": treatment,
            "tumor_change_pct": tumor_change_pct,
            "objective_response": objective_response,
        }
    )

    true_responder = true_cate <= RESPONDER_CATE_THRESHOLD
    ground_truth = {
        "patient_id": df["patient_id"].tolist(),
        "true_cate": true_cate.tolist(),
        "true_responder": true_responder.tolist(),
        "responder_cate_threshold": RESPONDER_CATE_THRESHOLD,
        "responder_rule_description": (
            "EGFR-mutant AND PD-L1 expression high (smooth threshold around "
            "~50%), not a hard box rule"
        ),
        "responder_prevalence": float(true_responder.mean()),
    }
    return df, ground_truth


def save(df: pd.DataFrame, ground_truth: dict, out_dir: str | Path = "data") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "trial_data.csv", index=False)
    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)


if __name__ == "__main__":
    trial_df, truth = simulate()
    save(trial_df, truth)
    print(f"Simulated {len(trial_df)} patients -> data/trial_data.csv")
    print(f"True responder prevalence: {truth['responder_prevalence']:.1%}")
    print(f"Arms: {trial_df['treatment'].value_counts().to_dict()}")
