#!/usr/bin/env python3
"""
Logistic-regression feature-subset search for homodimer prediction.

- Uses sklearn's roc_curve + interpolation to get TPR at an exact FPR (threshold-free).
- Keeps thresholded metrics (precision/F1/MCC/etc.) at a concrete threshold unchanged.
- Enforces an FPR band using the *nearest realized ROC point* to the target FPR.
- Supports splitting the combination space across runs: --split-combos, --n-splits-combos, --current-combo
"""

import re, itertools, math, numpy as np, pandas as pd, argparse
from typing import Tuple, Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tqdm


# ----------------------------
# sklearn-based TPR@exact FPR (threshold-free), plus nearest realized FPR
# ----------------------------
def tpr_at_exact_fpr_sklearn(y_true, y_score, target_fpr=0.05, sample_weight=None):
    """
    Returns:
      tpr_exact  : TPR at exactly target_fpr via linear interpolation on ROC
      fpr_near   : nearest realized FPR on the ROC (for tolerance checks)
      tpr_near   : TPR at that nearest realized FPR
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    fpr, tpr, _ = roc_curve(
        y_true, y_score, sample_weight=sample_weight, drop_intermediate=False
    )

    # Exact TPR by interpolation (threshold-free)
    tpr_exact = float(np.interp(target_fpr, fpr, tpr))

    # Nearest realized ROC step (for band enforcement/reporting)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    fpr_near = float(fpr[idx])
    tpr_near = float(tpr[idx])

    return tpr_exact, fpr_near, tpr_near


# ----------------------------
# Metric helpers
# ----------------------------
def _confusion(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
    TP = float(np.sum(sample_weight[(y_true == 1) & (y_pred == 1)]))
    TN = float(np.sum(sample_weight[(y_true == 0) & (y_pred == 0)]))
    FP = float(np.sum(sample_weight[(y_true == 0) & (y_pred == 1)]))
    FN = float(np.sum(sample_weight[(y_true == 1) & (y_pred == 0)]))
    return TP, FP, TN, FN


def _metrics_at_threshold(y_true, y_score, thr: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thr).astype(int)

    TP, FP, TN, FN = _confusion(y_true, y_pred)

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    tpr = rec
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        "precision_at_thr": float(prec),
        "recall_at_thr": float(rec),
        "f1_at_thr": float(f1),
        "mcc_at_thr": float(mcc),
        "tpr_at_thr": float(tpr),
        "fpr_at_thr": float(fpr),
        "accuracy_at_thr": float(acc),
    }


def _best_threshold_by_metric(y_true, y_score, metric: str) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    uniq = np.unique(y_score)
    if len(uniq) == 1:
        thr_candidates = [0.5]
    else:
        mids = (uniq[1:] + uniq[:-1]) / 2.0
        thr_candidates = np.concatenate(([uniq[0] - 1e-12], mids, [uniq[-1] + 1e-12]))
    best_v, best_thr = -np.inf, 0.5
    for thr in thr_candidates:
        m = _metrics_at_threshold(y_true, y_score, thr)
        v = m["f1_at_thr"] if metric == "f1" else m["mcc_at_thr"]
        if v > best_v:
            best_v, best_thr = v, thr
    return float(best_thr), float(best_v)


def _threshold_for_fpr(y_true, y_score, target_fpr=0.05, weights=None, n_grid=1000):
    """
    Manual sweep to find the threshold whose FPR is closest to target_fpr.
    Returns (chosen_threshold, fpr_at_thr, tpr_at_thr).
    Kept for selecting a concrete threshold to produce thresholded metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    w = (
        np.ones_like(y_true, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )

    smin, smax = float(np.min(y_score)), float(np.max(y_score))
    if np.isclose(smin, smax):
        thresholds = np.array([smin])
    else:
        thresholds = np.linspace(smin, smax, int(n_grid))

    fprs, tprs = [], []
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        TP = np.sum(w[(y_true == 1) & (pred == 1)])
        FN = np.sum(w[(y_true == 1) & (pred == 0)])
        FP = np.sum(w[(y_true == 0) & (pred == 1)])
        TN = np.sum(w[(y_true == 0) & (pred == 0)])

        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        fprs.append(fpr)
        tprs.append(tpr)

    fprs = np.asarray(fprs, dtype=float)
    tprs = np.asarray(tprs, dtype=float)

    idx = int(np.argmin(np.abs(fprs - target_fpr)))
    chosen_thr = float(thresholds[idx])
    return chosen_thr, float(fprs[idx]), float(tprs[idx])


def _tpr_at_fixed_threshold(y_true, y_score, thr: float) -> float:
    return _metrics_at_threshold(y_true, y_score, thr)["tpr_at_thr"]


def choose_threshold_and_score(
    y_true, y_score, optimize_for: str, target_fpr: float = 0.05
) -> Tuple[float, float]:
    """
    Returns (chosen_threshold, optimized_value).
    optimize_for ∈ {"f1","mcc","tpr_at_fpr5","tpr_at_xfpr","tpr_at_<float>"}.
    - For tpr_at_fpr5 / tpr_at_xfpr we use sklearn interpolation (threshold-free) and
      return (np.nan, tpr_exact) because there's no single hard threshold at that exact point.
    - "tpr_at_<float>": use that float as the fixed decision threshold and optimize TPR there.
    """
    optimize_for = optimize_for.lower()
    if optimize_for == "f1":
        return _best_threshold_by_metric(y_true, y_score, metric="f1")
    if optimize_for == "mcc":
        return _best_threshold_by_metric(y_true, y_score, metric="mcc")
    if optimize_for == "tpr_at_fpr5":
        tpr_exact, _, _ = tpr_at_exact_fpr_sklearn(y_true, y_score, target_fpr=0.05)
        return float("nan"), float(tpr_exact)
    if optimize_for == "tpr_at_xfpr":
        tpr_exact, _, _ = tpr_at_exact_fpr_sklearn(
            y_true, y_score, target_fpr=target_fpr
        )
        return float("nan"), float(tpr_exact)
    m = re.fullmatch(r"tpr_at_([0-9]*\.?[0-9]+)", optimize_for)
    if m:
        thr = float(m.group(1))
        return thr, _tpr_at_fixed_threshold(y_true, y_score, thr)
    raise ValueError(f"Unknown optimize_for: {optimize_for}")


def evaluate_full(y_true, y_score, thr_for_metrics: float, target_fpr=0.05) -> dict:
    """Thresholded metrics at thr + generic PR/ROC. Exact TPR@FPR computed outside."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
    out = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "auc_pr": auc(rec_curve, prec_curve),
    }
    out.update(_metrics_at_threshold(y_true, y_score, thr_for_metrics))
    return out


# ----------------------------
# Argparse config
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="LogReg feature-subset search with flexible thresholding."
    )
    p.add_argument(
        "--data",
        type=str,
        default="../tsvs/logreg_features/homodimers_logreg_features.tsv",
        help="Path to TSV with features + 'correct_result' and 'ID'.",
    )
    p.add_argument(
        "--optimize-metric",
        type=str,
        default="mcc",
        help="One of: f1, mcc, tpr_at_fpr5, tpr_at_xfpr, tpr_at_<float> (e.g., tpr_at_0.8).",
    )
    p.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target FPR for tpr_at_fpr5 / tpr_at_xfpr (default 0.05).",
    )
    p.add_argument(
        "--fpr-tol",
        type=float,
        default=0.01,
        help="Allowed deviation for mean FPR across folds when optimizing tpr_at_xfpr/fpr5.",
    )
    p.add_argument("--use-tqdm", action="store_true", help="Show tqdm progress bar.")
    p.add_argument("--split-combos", action="store_true", help="Split the combo space.")
    p.add_argument(
        "--n-splits-combos", type=int, default=10, help="Number of combo splits."
    )
    p.add_argument(
        "--current-combo", type=int, default=1, help="Which split (1-based) to run."
    )
    p.add_argument(
        "--cv-folds", type=int, default=5, help="Number of CV folds (default 5)."
    )
    p.add_argument("--random-state", type=int, default=42, help="Random seed for CV.")
    p.add_argument(
        "--min-k", type=int, default=1, help="Minimum number of features per cluster."
    )
    p.add_argument(
        "--max-k", type=int, default=1, help="Maximum number of features per cluster."
    )
    p.add_argument(
        "--no-fseek",
        action="store_true",
        help="Exclude Foldseek-derived (homology) features from the search space.",
    )
    p.add_argument(
        "--each-feature-1-cluster",
        "--each-feature-one-cluster",
        dest="each_feature_1_cluster",
        action="store_true",
        help="Treat each remaining feature as its own cluster (try all combinations).",
    )
    return p.parse_args()


# ----------------------------
# Feature/cluster setup utilities
# ----------------------------
FSEEK_FEATURES: List[str] = [
    "homomultimer_fraction_stoich_e0.6",
    "homomultimer_fraction_stoich_e0.8",
    "multimer_fraction_stoich_e0.8",
    "homomultimer_fraction_stoich_e0.9",
    "multimer_fraction_stoich_e0.4",
    "highest_evalue_all_hits",
    "highest_evalue_homomultimers",
    "multimer_fraction_stoich_e0.6",
]


def build_clusters(args) -> List[List[str]]:
    """
    Returns the list of clusters after applying --no-fseek and --each-feature-1-cluster.
    """
    clusters_base = [
        # non-fseek clusters
        ["buried_apolar_area", "buried_polar_area", "total_interaction_area"],
        ["fraction_buried_apolar_area", "fraction_buried_polar_area"],
        [
            "min_iptm",
            "min_rc",
            "structural_consensus",
            "min_contacts_across_predictions",
        ],
        [
            "max_iptm",
            "avg_iptm",
            "max_rc",
            "avg_rc",
            "num_contacts_with_max_n_models",
            "num_unique_contacts",
            "mean_contacts_across_predictions",
            "best_num_residue_contacts",
            "best_if_residues",
            "best_plddt_max",
            "best_contact_score_max",
        ],
        ["best_pae_min"],
        # fseek clusters
        ["hm_frac_tm0.6"],
        ["hm_frac_tm0.8"],
        ["multimer_frac_tm0.8"],
        ["hm_frac_tm0.9", "highest_tm_homomultimers"],
        ["multimer_frac_tm0.4", "multimer_frac_tm0.6"],
        ["highest_tm_all_hits"],
    ]

    if args.no_fseek:
        filtered = []
        for cl in clusters_base:
            kept = [f for f in cl if f not in FSEEK_FEATURES]
            if kept:
                filtered.append(kept)
        clusters = filtered
    else:
        clusters = clusters_base

    if args.each_feature_1_cluster:
        all_feats = []
        for cl in clusters:
            all_feats.extend(cl)
        seen = set()
        uniq_feats = []
        for f in all_feats:
            if f not in seen:
                seen.add(f)
                uniq_feats.append(f)
        clusters = [[f] for f in uniq_feats]

    return clusters


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    merged_df = pd.read_table(args.data, keep_default_na=False)
    df = merged_df.copy()
    labels = df["correct_result"].to_numpy()

    skf = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
    )
    DEFAULT_MIN_K = args.min_k
    DEFAULT_MAX_K = args.max_k

    clusters = build_clusters(args)
    folds = list(skf.split(np.zeros(len(labels)), labels))

    def selections_for_cluster(features, min_k, max_k):
        max_k = min(max_k, len(features))
        min_k = max(0, min_k)
        out = []
        for k in range(min_k, max_k + 1):
            out.extend(itertools.combinations(features, k))
        return out

    cluster_selections = [
        selections_for_cluster(feats, DEFAULT_MIN_K, DEFAULT_MAX_K)
        for feats in clusters
    ]

    def count_cluster_selections(n_feats, min_k, max_k):
        max_k = min(max_k, n_feats)
        min_k = max(0, min_k)
        return sum(math.comb(n_feats, k) for k in range(min_k, max_k + 1))

    # total number of combo tuples across all clusters
    n_combos_total = 1
    for feats in clusters:
        n_combos_total *= count_cluster_selections(
            len(feats), DEFAULT_MIN_K, DEFAULT_MAX_K
        )

    if n_combos_total == 0:
        raise RuntimeError(
            "No combinations to evaluate (check clusters / min_k / max_k)."
        )

    # Build iterable over combos (possibly sliced)
    def combos_slice(start, stop):
        return itertools.islice(itertools.product(*cluster_selections), start, stop)

    if args.split_combos:
        if args.n_splits_combos <= 0:
            raise ValueError("--n-splits-combos must be >= 1")
        if not (1 <= args.current_combo <= args.n_splits_combos):
            raise ValueError("--current-combo must be in [1, n_splits_combos]")

        chunk_size = (n_combos_total + args.n_splits_combos - 1) // args.n_splits_combos
        start = (args.current_combo - 1) * chunk_size
        stop = min(start + chunk_size, n_combos_total)
        n_this = max(0, stop - start)
        combo_iter = combos_slice(start, stop)
        desc = f"Searching combos [{args.current_combo}/{args.n_splits_combos}]"
        total_for_bar = n_this
    else:
        combo_iter = itertools.product(*cluster_selections)
        desc = "Searching feature combos"
        total_for_bar = n_combos_total

    # CV scorer for a subset with optional FPR band enforcement (sklearn interpolation + nearest realized FPR)
    def cv_score_for_subset(subset_cols, optimize_for: str) -> Tuple[float, float]:
        if len(subset_cols) == 0:
            return float("-inf"), float("nan")
        X = df[list(subset_cols)].values
        fold_vals = []
        fold_fprs_near = []

        enforce_fpr = optimize_for.lower() in ("tpr_at_xfpr", "tpr_at_fpr5")

        for tr, te in folds:
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    n_jobs=1,
                    random_state=args.random_state,
                ),
            )
            pipe.fit(X[tr], labels[tr])
            score = pipe.predict_proba(X[te])[:, 1]

            if enforce_fpr:
                tf = args.target_fpr if optimize_for.lower() == "tpr_at_xfpr" else 0.05
                tpr_exact, fpr_near, _ = tpr_at_exact_fpr_sklearn(
                    labels[te], score, target_fpr=tf
                )
                fold_vals.append(tpr_exact)
                fold_fprs_near.append(fpr_near)  # use realized FPR step for tolerance
            else:
                _, opt_val = choose_threshold_and_score(
                    labels[te],
                    score,
                    optimize_for=optimize_for,
                    target_fpr=args.target_fpr,
                )
                fold_vals.append(opt_val)

        mean_val, std_val = float(np.mean(fold_vals)), float(np.std(fold_vals))

        if enforce_fpr and len(fold_fprs_near) > 0:
            mean_fpr_near = float(np.mean(fold_fprs_near))
            tf = args.target_fpr if optimize_for.lower() == "tpr_at_xfpr" else 0.05
            if not (tf - args.fpr_tol <= mean_fpr_near <= tf + args.fpr_tol):
                return float("-inf"), float("nan")

        return mean_val, std_val

    # Search
    best_score = None
    best_subset = None
    best_var = None

    iterable = (
        tqdm.tqdm(combo_iter, total=total_for_bar, desc=desc)
        if args.use_tqdm
        else combo_iter
    )

    for combo in iterable:
        subset = [f for tpl in combo for f in tpl]
        subset = list(dict.fromkeys(subset))  # dedupe while preserving order
        if len(subset) == 0:
            continue
        mean_val, spread_val = cv_score_for_subset(
            subset, optimize_for=args.optimize_metric
        )
        if (best_score is None) or (mean_val > best_score):
            best_score, best_subset, best_var = mean_val, subset, spread_val

    if best_subset is None or best_score in (None, float("-inf")):
        raise RuntimeError(
            "No valid subsets were evaluated (likely all violated the FPR tolerance). "
            "Try increasing --fpr-tol or check your settings."
        )

    print(
        f"\n✅ Best subset by {args.optimize_metric}: {', '.join(best_subset) if best_subset else '(no features)'}"
    )
    print(f"   Mean {args.optimize_metric} over CV: {best_score:.4f} ± {best_var:.4f}")

    # Final CV with best subset
    X_best = df[best_subset].values if len(best_subset) > 0 else None
    final_metrics = []
    fold_thresholds = []
    fold_fprs_final = []

    for tr, te in folds:
        y_tr, y_te = labels[tr], labels[te]
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, solver="lbfgs", n_jobs=1, random_state=args.random_state
            ),
        )

        if X_best is None or X_best.shape[1] == 0:
            proba = np.full_like(y_te, fill_value=0.5, dtype=float)
        else:
            X_tr, X_te = X_best[tr], X_best[te]
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]

        # Keep a concrete threshold for thresholded metrics (unchanged behavior)
        if args.optimize_metric.lower() in ("tpr_at_xfpr", "tpr_at_fpr5"):
            tf = (
                args.target_fpr
                if args.optimize_metric.lower() == "tpr_at_xfpr"
                else 0.05
            )
            thr, fpr_at_thr, _ = _threshold_for_fpr(y_te, proba, target_fpr=tf)
        else:
            thr, _ = choose_threshold_and_score(
                y_te,
                proba,
                optimize_for=args.optimize_metric,
                target_fpr=args.target_fpr,
            )
            fpr_at_thr = _metrics_at_threshold(y_te, proba, thr)["fpr_at_thr"]

        fold_thresholds.append(thr)
        fold_fprs_final.append(fpr_at_thr)

        # exact, threshold-free TPR@5% FPR (independent of the threshold)
        tpr5_exact, _, _ = tpr_at_exact_fpr_sklearn(y_te, proba, target_fpr=0.05)

        fm = evaluate_full(y_te, proba, thr_for_metrics=thr, target_fpr=args.target_fpr)
        fm["tpr_at_fpr5_exact"] = float(tpr5_exact)
        final_metrics.append(fm)

    def _mean_std(key):
        vals = [m[key] for m in final_metrics]
        return np.mean(vals), np.std(vals)

    print("\n📈 Final Logistic-Regression performance (best subset):")
    for k in [
        "roc_auc",
        "auc_pr",
        "f1_at_thr",
        "mcc_at_thr",
        "precision_at_thr",
        "tpr_at_thr",
        "fpr_at_thr",
        "tpr_at_fpr5_exact",
    ]:
        mu, sd = _mean_std(k)
        print(f"  {k:<18}: {mu:.4f} ± {sd:.4f}")

    thr_mean = np.mean(fold_thresholds)
    thr_std = np.std(fold_thresholds)
    print(f"\n🔧 Thresholds used across folds: {thr_mean:.4f} ± {thr_std:.4f}")

    # Check final hard-threshold FPR band adherence (separate from threshold-free metric)
    if args.optimize_metric.lower() in ("tpr_at_xfpr", "tpr_at_fpr5"):
        tf = args.target_fpr if args.optimize_metric.lower() == "tpr_at_xfpr" else 0.05
        mean_final_fpr = float(np.mean(fold_fprs_final))
        if not (tf - args.fpr_tol <= mean_final_fpr <= tf + args.fpr_tol):
            print(
                f"\n⚠ Note: mean fpr_at_thr across folds = {mean_final_fpr:.4f}, "
                f"outside target {tf:.4f} ± {args.fpr_tol:.4f}. "
                f"Consider increasing --fpr-tol or revisiting features."
            )


if __name__ == "__main__":
    main()
