"""
Turn a per-patient CATE estimate into a small number of human-readable
eligibility rules -- the actual deliverable a clinical team can act on.

What a trial team needs is a short rule like "EGFR-mutant AND
PD-L1 >= 52%". We get there by fitting a shallow decision tree to predict
the estimated CATE from baseline covariates (a "policy tree" /
interpretable surrogate, in the spirit of Athey & Wager's policy trees and
the "Virtual Twins" method of Foster, Taylor & Ruberg, 2011). The tree
isn't the causal model -- the T-learner is -- it's a compression step that
turns a continuous, high-dimensional effect surface into a rule simple
enough for a protocol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree


def fit_policy_tree(
    X: pd.DataFrame,
    cate_hat: np.ndarray,
    max_depth: int = 3,
    min_samples_leaf_frac: float = 0.03,
    seed: int = 0,
) -> DecisionTreeRegressor:
    """Fit a shallow regression tree that predicts estimated CATE from
    baseline covariates. Shallow on purpose: depth 2-3 is about the limit
    of what fits cleanly into a trial eligibility criterion."""
    min_leaf = max(10, int(min_samples_leaf_frac * len(X)))
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed)
    tree.fit(X, cate_hat)
    return tree


def _leaf_rules(tree: DecisionTreeRegressor, feature_names: list[str]) -> dict[int, list[str]]:
    """Walk the fitted tree and collect the chain of split conditions that
    defines each leaf, as human-readable strings."""
    t = tree.tree_
    rules: dict[int, list[str]] = {}

    def recurse(node: int, conditions: list[str]) -> None:
        if t.feature[node] == _tree.TREE_UNDEFINED:
            rules[node] = conditions
            return
        name = feature_names[t.feature[node]]
        thresh = t.threshold[node]
        recurse(t.children_left[node], conditions + [f"{name} <= {thresh:.2f}"])
        recurse(t.children_right[node], conditions + [f"{name} > {thresh:.2f}"])

    recurse(0, [])
    return rules


def summarize_leaves(
    tree: DecisionTreeRegressor,
    X: pd.DataFrame,
    cate_hat: np.ndarray,
) -> pd.DataFrame:
    """One row per leaf: its rule, size, and mean estimated CATE, sorted
    from most-benefit to least-benefit."""
    feature_names = list(X.columns)
    leaf_id = tree.apply(X)
    rules = _leaf_rules(tree, feature_names)

    rows = []
    for node, conditions in rules.items():
        mask = leaf_id == node
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "leaf_id": node,
                "rule": " AND ".join(conditions) if conditions else "(everyone)",
                "n_patients": int(mask.sum()),
                "pct_of_population": float(mask.mean()),
                "mean_cate_hat": float(cate_hat[mask].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_cate_hat").reset_index(drop=True)
    return summary


def discover_subgroup(
    X: pd.DataFrame,
    cate_hat: np.ndarray,
    benefit_threshold: float = -20.0,
    max_depth: int = 3,
    seed: int = 0,
) -> dict:
    """End-to-end: fit the policy tree, then define the discovered
    subgroup as the union of leaves whose mean estimated CATE clears the
    benefit threshold (more negative = more tumor shrinkage = more
    benefit here). Returns the tree, the leaf summary table, a boolean
    membership mask, and the plain-language rule(s)."""
    tree = fit_policy_tree(X, cate_hat, max_depth=max_depth, seed=seed)
    leaves = summarize_leaves(tree, X, cate_hat)

    responder_leaves = leaves[leaves["mean_cate_hat"] <= benefit_threshold]
    leaf_id = tree.apply(X)
    in_subgroup = np.isin(leaf_id, responder_leaves["leaf_id"].to_numpy())

    return {
        "tree": tree,
        "leaf_summary": leaves,
        "in_subgroup": in_subgroup,
        "responder_leaves": responder_leaves,
        "rules": responder_leaves["rule"].tolist(),
    }
