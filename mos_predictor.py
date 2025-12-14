import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score


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


def _build_preprocess(feature_cols, categorical_features, scale_numeric=False):
    """
    Build a preprocessing transformer:
    - Numeric features: median imputation (+ optional standardization)
    - Categorical features: most_frequent imputation + one-hot encoding
    """
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    if scale_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
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


def _evaluate_regression_and_exact_accuracy(y_true, y_pred, mos_min=1, mos_max=5):
    """
    Compute regression metrics (MAE/RMSE/R2) using raw predictions,
    then round predictions to nearest MOS class and compute exact-match accuracy.
    """
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


def _train_eval_model(name, estimator, preprocess, X_train, X_test, y_train, y_test, show_first_n=5):
    """
    Train a model pipeline (preprocess + estimator), evaluate on test set,
    print metrics and first N prediction samples.
    """
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

    return results


def compare_models(csv_path="pokemon.csv", test_size=0.2, random_state=42, show_first_n=5):
    """
    Compare Random Forest, Decision Tree, SVM (SVR-RBF), and Neural Network (MLP)
    on the SAME train/test split for a fair comparison.
    """
    df = pd.read_csv(csv_path)
    data = df[FEATURE_COLS + [TARGET_COL]].copy()

    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL])

    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    # Use the same split for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Preprocess configs:
    # - Trees: no scaling
    # - SVM/NN: scaling is important for best performance
    preprocess_tree = _build_preprocess(FEATURE_COLS, CATEGORICAL_FEATURES, scale_numeric=False)
    preprocess_scaled = _build_preprocess(FEATURE_COLS, CATEGORICAL_FEATURES, scale_numeric=True)

    # 1) Random Forest (strong non-linear baseline, robust)
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        min_samples_leaf=3,
        n_jobs=-1
    )
    rf_res = _train_eval_model(
        "Random Forest (Regressor)", rf, preprocess_tree,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
    )

    print("\n" + "=" * 60 + "\n")

    # 2) Decision Tree (simple baseline)
    dt = DecisionTreeRegressor(
        random_state=random_state,
        min_samples_leaf=3,
        max_depth=None
    )
    dt_res = _train_eval_model(
        "Decision Tree (Regressor)", dt, preprocess_tree,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
    )

    print("\n" + "=" * 60 + "\n")

    # 3) SVM (Best default choice: SVR with RBF kernel for non-linear patterns)
    svr = SVR(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        epsilon=0.1
    )
    svr_res = _train_eval_model(
        "SVM (SVR-RBF)", svr, preprocess_scaled,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
    )

    print("\n" + "=" * 60 + "\n")

    # 4) Neural Network (Best default choice for tabular: MLP with early stopping)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,                 # stronger regularization
        learning_rate="adaptive",
        learning_rate_init=5e-4,    # slightly smaller LR
        batch_size=64,
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state
    )

    mlp_res = _train_eval_model(
        "Neural Network (MLP)", mlp, preprocess_scaled,
        X_train, X_test, y_train, y_test, show_first_n=show_first_n
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
    compare_models("pokemon.csv", show_first_n=5)
