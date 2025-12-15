import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
    confusion_matrix
)

from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import matplotlib.pyplot as plt


FEATURE_COLS = [
    "QoA_VLCbitrate",
    "QoA_BUFFERINGtime",
    "QoA_VLCaudiorate",
    "QoA_VLCframerate",
    "QoD_model",      # categorical
    "QoF_audio",
    "QoF_begin",
    "QoF_video",
]
TARGET_COL = "MOS"
CATEGORICAL_FEATURES = ["QoD_model"]


# -------------------------
# Preprocess
# -------------------------
def _build_preprocess(feature_cols, categorical_features, scale_numeric=False):
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    if scale_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )
    return preprocess


# -------------------------
# Metrics
# -------------------------
def _evaluate_regression_and_exact_accuracy(y_true, y_pred, mos_min=1, mos_max=5):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    y_pred_round = np.rint(y_pred)
    y_pred_round = np.clip(y_pred_round, mos_min, mos_max).astype(int)

    y_true_int = np.rint(np.asarray(y_true)).astype(int)
    y_true_int = np.clip(y_true_int, mos_min, mos_max)

    acc = accuracy_score(y_true_int, y_pred_round)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "accuracy_exact": acc,
        "y_true_int": y_true_int,
        "y_pred_raw": y_pred,
        "y_pred_rounded": y_pred_round,
    }


# -------------------------
# Confusion matrices (pair)
# -------------------------
def _plot_confusion_matrices_pair(cm_items, labels, save_path, normalize="true", suptitle=None):
    """
    Plot 2 confusion matrices into a single 1x2 figure and save it.
    cm_items: list of exactly 2 tuples -> (title, y_true_int, y_pred_rounded)
    normalize: None | 'true' | 'pred' | 'all'
    """
    assert len(cm_items) == 2, "cm_items must contain exactly 2 items."
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = np.asarray(axes).ravel()

    cms = []
    for _, y_true, y_pred in cm_items:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
        cms.append(cm)

    vmin = 0.0
    vmax = max(float(cm.max()) for cm in cms) if cms else 1.0

    mappables = []
    for ax, (title, _, _), cm in zip(axes, cm_items, cms):
        im = ax.imshow(cm, vmin=vmin, vmax=vmax)
        mappables.append(im)

        ax.set_title(title)
        ax.set_xlabel("Predicted MOS")
        ax.set_ylabel("True MOS")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        fmt = ".2f" if normalize else "d"
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center")

    fig.subplots_adjust(right=0.86, wspace=0.25)
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.70])
    fig.colorbar(mappables[0], cax=cax)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.show()
    print(f"Saved figure: {save_path}")


# -------------------------
# Utils for saving figs
# -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _savefig(fig, path: str):
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


# -------------------------
# Tree figures utilities
# -------------------------
def _get_transformed_feature_names(preprocess: ColumnTransformer):
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in preprocess.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    names.extend(list(trans.get_feature_names_out(cols)))
                except Exception:
                    names.extend([f"{name}__{c}" for c in cols])
            else:
                names.extend([f"{name}__{c}" for c in cols])
        return names


def _aggregate_importance_by_original_feature(feature_names, importances):
    agg = {}
    for fname, imp in zip(feature_names, importances):
        base = fname.split("__", 1)[-1]  # remove "num__" or "cat__"
        if base.startswith("QoD_model"):
            key = "QoD_model"
        else:
            key = base
        agg[key] = agg.get(key, 0.0) + float(imp)

    items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    return keys, vals


def _save_feature_importance_bar(keys, vals, title, save_path, top_n=20):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = keys[:top_n]
    vals = vals[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(keys)), list(reversed(vals)))
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.show()
    print(f"Saved: {save_path}")


def _save_permutation_importance_bar(perm_result, feature_cols, title, save_path, top_n=20):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    means = perm_result.importances_mean
    stds = perm_result.importances_std
    order = np.argsort(means)[::-1]

    order = order[:min(top_n, len(order))]
    feat = [feature_cols[i] for i in order]
    m = means[order]
    s = stds[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(feat)), list(reversed(m)), xerr=list(reversed(s)))
    ax.set_title(title)
    ax.set_xlabel("Permutation Importance (mean ± std)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.show()
    print(f"Saved: {save_path}")


def _save_tree_structure_fig(tree_estimator, feature_names, title, save_path, max_depth=3):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        tree_estimator,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=9,
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.show()
    print(f"Saved: {save_path}")


def _save_pdp_fig(pipeline_model, X_train, features, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        pipeline_model,
        X_train,
        features=features,
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.show()
    print(f"Saved: {save_path}")


def generate_tree_figures(model_name, pipeline_model, X_train, X_test, y_test,
                          out_dir="figures/tree_figures", max_tree_depth=3,
                          pdp_top_k=3, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    preprocess = pipeline_model.named_steps["preprocess"]
    estimator = pipeline_model.named_steps["model"]

    transformed_names = _get_transformed_feature_names(preprocess)

    # Feature importance (aggregated)
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        keys, vals = _aggregate_importance_by_original_feature(transformed_names, importances)

        _save_feature_importance_bar(
            keys, vals,
            title=f"{model_name} - Feature Importance (aggregated)",
            save_path=os.path.join(out_dir, f"{_safe_filename(model_name)}_feature_importance.png"),
            top_n=20
        )

        # PDP on top numeric raw features (exclude categorical)
        numeric_keys = [k for k in keys if k in FEATURE_COLS and k not in CATEGORICAL_FEATURES]
        pdp_feats = numeric_keys[:pdp_top_k]
        if len(pdp_feats) > 0:
            _save_pdp_fig(
                pipeline_model,
                X_train,
                features=pdp_feats,
                title=f"{model_name} - Partial Dependence (top numeric features)",
                save_path=os.path.join(out_dir, f"{_safe_filename(model_name)}_pdp_top{len(pdp_feats)}.png")
            )

    # Permutation importance (raw)
    try:
        perm = permutation_importance(
            pipeline_model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1
        )
        _save_permutation_importance_bar(
            perm,
            feature_cols=FEATURE_COLS,
            title=f"{model_name} - Permutation Importance (raw features)",
            save_path=os.path.join(out_dir, f"{_safe_filename(model_name)}_permutation_importance.png"),
            top_n=20
        )
    except Exception as e:
        print(f"[WARN] permutation importance failed for {model_name}: {e}")

    # Tree structure
    try:
        if isinstance(estimator, DecisionTreeRegressor):
            _save_tree_structure_fig(
                estimator,
                feature_names=transformed_names,
                title=f"{model_name} - Tree Structure (max_depth={max_tree_depth})",
                save_path=os.path.join(out_dir, f"{_safe_filename(model_name)}_tree_structure_depth{max_tree_depth}.png"),
                max_depth=max_tree_depth
            )
        elif isinstance(estimator, RandomForestRegressor):
            tree0 = estimator.estimators_[0]
            _save_tree_structure_fig(
                tree0,
                feature_names=transformed_names,
                title=f"{model_name} - One Tree (max_depth={max_tree_depth})",
                save_path=os.path.join(out_dir, f"{_safe_filename(model_name)}_one_tree_depth{max_tree_depth}.png"),
                max_depth=max_tree_depth
            )
    except Exception as e:
        print(f"[WARN] tree plotting failed for {model_name}: {e}")


# -------------------------
# SVM figures (model-agnostic)
# -------------------------
def _plot_pred_vs_true(y_true, y_pred, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi])
    ax.set_title(title)
    ax.set_xlabel("True MOS")
    ax.set_ylabel("Predicted MOS")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_path)


def _plot_residuals_vs_pred(y_true, y_pred, title, save_path):
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, resid, alpha=0.7)
    ax.axhline(0.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted MOS")
    ax.set_ylabel("Residual (True - Pred)")
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_path)


def _plot_residual_hist(y_true, y_pred, title, save_path, bins=25):
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(resid, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Residual (True - Pred)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _savefig(fig, save_path)


def _plot_permutation_importance(pipeline_model, X_test, y_test, feature_cols,
                                 title, save_path, n_repeats=10, random_state=42, top_n=20):
    perm = permutation_importance(
        pipeline_model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    means = perm.importances_mean
    stds = perm.importances_std
    order = np.argsort(means)[::-1][:min(top_n, len(feature_cols))]

    feats = [feature_cols[i] for i in order]
    m = means[order]
    s = stds[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(feats)), list(reversed(m)), xerr=list(reversed(s)))
    ax.set_title(title)
    ax.set_xlabel("Permutation importance (mean ± std)")
    ax.grid(True, axis="x", alpha=0.3)
    _savefig(fig, save_path)

    ranked_features = [feature_cols[i] for i in order]
    return ranked_features


def _plot_pdp(pipeline_model, X_train, features, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        pipeline_model,
        X_train,
        features=features,
        ax=ax
    )
    ax.set_title(title)
    _savefig(fig, save_path)


def generate_svm_figures(model_name, pipeline_model, X_train, X_test, y_test,
                         out_dir="figures/SVM_figures", random_state=42, pdp_top_k=3):
    _ensure_dir(out_dir)

    y_pred = pipeline_model.predict(X_test)

    _plot_pred_vs_true(
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        title=f"{model_name} - Predicted vs True",
        save_path=os.path.join(out_dir, "svr_pred_vs_true.png")
    )

    _plot_residuals_vs_pred(
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        title=f"{model_name} - Residuals vs Predicted",
        save_path=os.path.join(out_dir, "svr_residuals_vs_pred.png")
    )

    _plot_residual_hist(
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        title=f"{model_name} - Residual Histogram",
        save_path=os.path.join(out_dir, "svr_residual_hist.png")
    )

    ranked = _plot_permutation_importance(
        pipeline_model=pipeline_model,
        X_test=X_test,
        y_test=y_test,
        feature_cols=FEATURE_COLS,
        title=f"{model_name} - Permutation Importance (global)",
        save_path=os.path.join(out_dir, "svr_permutation_importance.png"),
        n_repeats=10,
        random_state=random_state,
        top_n=20
    )

    numeric_ranked = [f for f in ranked if f not in CATEGORICAL_FEATURES]
    pdp_feats = numeric_ranked[:pdp_top_k]
    if len(pdp_feats) > 0:
        _plot_pdp(
            pipeline_model=pipeline_model,
            X_train=X_train,
            features=pdp_feats,
            title=f"{model_name} - PDP (top numeric: {', '.join(pdp_feats)})",
            save_path=os.path.join(out_dir, f"svr_pdp_top{len(pdp_feats)}.png")
        )

    # optional: support vectors count
    try:
        svr = pipeline_model.named_steps["model"]
        n_sv = len(svr.support_)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["#Support Vectors"], [n_sv])
        ax.set_title(f"{model_name} - Model Complexity")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
        _savefig(fig, os.path.join(out_dir, "svr_support_vectors_count.png"))
    except Exception as e:
        print(f"[WARN] support vector plot skipped: {e}")


# -------------------------
# Train/eval
# -------------------------
def _train_eval_model(name, estimator, preprocess, X_train, X_test, y_train, y_test, show_first_n=5):
    model = Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = _evaluate_regression_and_exact_accuracy(y_test, y_pred)

    print(f"===== {name} =====")
    print(f"MAE  : {results['mae']:.3f}")
    print(f"RMSE : {results['rmse']:.3f}")
    print(f"R²   : {results['r2']:.3f}")
    print(f"Exact MOS Accuracy (rounded): {results['accuracy_exact']:.4f}")

    compare = pd.DataFrame({
        "MOS_true": results["y_true_int"],
        "MOS_pred_raw": results["y_pred_raw"],
        "MOS_pred_rounded": results["y_pred_rounded"],
    })
    print(f"\n===== Sample Predictions (first {show_first_n}) =====")
    print(compare.head(show_first_n).to_string(index=False))

    return model, results


def compare_models(csv_path="pokemon.csv", test_size=0.2, random_state=42, show_first_n=5,
                   cm_normalize="true"):
    df = pd.read_csv(csv_path)
    data = df[FEATURE_COLS + [TARGET_COL]].copy()

    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL])

    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocess_tree = _build_preprocess(FEATURE_COLS, CATEGORICAL_FEATURES, scale_numeric=False)
    preprocess_scaled = _build_preprocess(FEATURE_COLS, CATEGORICAL_FEATURES, scale_numeric=True)

    # 1) Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        min_samples_leaf=3,
        n_jobs=-1
    )
    rf_model, rf_res = _train_eval_model(
        "Random Forest (Regressor)", rf, preprocess_tree,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
    )

    print("\n" + "=" * 60 + "\n")

    # 2) Decision Tree
    dt = DecisionTreeRegressor(
        random_state=random_state,
        min_samples_leaf=3,
        max_depth=None
    )
    dt_model, dt_res = _train_eval_model(
        "Decision Tree (Regressor)", dt, preprocess_tree,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
    )

    print("\n" + "=" * 60 + "\n")

    # 3) SVR (RBF)
    svr = SVR(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        epsilon=0.1
    )
    svr_model, svr_res = _train_eval_model(
        "SVM (SVR-RBF)", svr, preprocess_scaled,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
    )

    # ---- SVM analysis figures -> figures/SVM_figures ----
    generate_svm_figures(
        model_name="SVM (SVR-RBF)",
        pipeline_model=svr_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        out_dir="figures/SVM_figures",
        random_state=random_state,
        pdp_top_k=3
    )

    print("\n" + "=" * 60 + "\n")

    # 4) MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate="adaptive",
        learning_rate_init=5e-4,
        batch_size=64,
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state
    )
    mlp_model, mlp_res = _train_eval_model(
        "Neural Network (MLP)", mlp, preprocess_scaled,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
    )

    # -------------------------
    # Confusion matrices (2 figures)
    # -------------------------
    labels = [1, 2, 3, 4, 5]
    os.makedirs("figures", exist_ok=True)

    _plot_confusion_matrices_pair(
        cm_items=[
            ("Random Forest", rf_res["y_true_int"], rf_res["y_pred_rounded"]),
            ("Decision Tree", dt_res["y_true_int"], dt_res["y_pred_rounded"]),
        ],
        labels=labels,
        normalize=cm_normalize,
        save_path="figures/confusion_matrices_RF_DT.png",
        suptitle=f"Confusion Matrices (RF + DT, normalize={cm_normalize})"
    )

    _plot_confusion_matrices_pair(
        cm_items=[
            ("SVR (RBF)", svr_res["y_true_int"], svr_res["y_pred_rounded"]),
            ("MLP", mlp_res["y_true_int"], mlp_res["y_pred_rounded"]),
        ],
        labels=labels,
        normalize=cm_normalize,
        save_path="figures/confusion_matrices_SVR_MLP.png",
        suptitle=f"Confusion Matrices (SVR + MLP, normalize={cm_normalize})"
    )

    # -------------------------
    # Tree interpretability figures -> figures/tree_figures
    # -------------------------
    tree_outdir = "figures/tree_figures"
    generate_tree_figures(
        model_name="Random Forest (Regressor)",
        pipeline_model=rf_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        out_dir=tree_outdir,
        max_tree_depth=3,
        pdp_top_k=3,
        random_state=random_state
    )

    generate_tree_figures(
        model_name="Decision Tree (Regressor)",
        pipeline_model=dt_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        out_dir=tree_outdir,
        max_tree_depth=3,
        pdp_top_k=3,
        random_state=random_state
    )

    # Summary table
    summary = pd.DataFrame([
        {"model": "RandomForestRegressor", "MAE": rf_res["mae"], "RMSE": rf_res["rmse"], "R2": rf_res["r2"], "ExactAccuracy": rf_res["accuracy_exact"]},
        {"model": "DecisionTreeRegressor", "MAE": dt_res["mae"], "RMSE": dt_res["rmse"], "R2": dt_res["r2"], "ExactAccuracy": dt_res["accuracy_exact"]},
        {"model": "SVR_RBF", "MAE": svr_res["mae"], "RMSE": svr_res["rmse"], "R2": svr_res["r2"], "ExactAccuracy": svr_res["accuracy_exact"]},
        {"model": "MLPRegressor", "MAE": mlp_res["mae"], "RMSE": mlp_res["rmse"], "R2": mlp_res["r2"], "ExactAccuracy": mlp_res["accuracy_exact"]},
    ])

    print("\n===== Model Comparison (Test Set) =====")
    print(summary.to_string(index=False))

    return summary


if __name__ == "__main__":
    compare_models(
        csv_path="pokemon.csv",
        show_first_n=5,
        cm_normalize="true"
    )
