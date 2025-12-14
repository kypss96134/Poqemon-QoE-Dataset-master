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


TARGET_COL = "MOS"
DROP_COLS = ["id", "user_id"]


def build_preprocess(X_train: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    """
    Preprocess ALL features in X_train:
    - Numeric: median imputation (+ optional standardization)
    - Categorical: most_frequent imputation + one-hot encoding
    """
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    num_pipe = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_pipe.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_pipe)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )


def evaluate(y_true, y_pred, mos_min=1, mos_max=5):
    """Regression metrics + exact MOS accuracy after rounding/clipping to [1,5]."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    y_pred_round = np.clip(np.rint(y_pred), mos_min, mos_max).astype(int)
    y_true_int = np.clip(np.rint(np.asarray(y_true)).astype(int), mos_min, mos_max)

    acc = accuracy_score(y_true_int, y_pred_round)

    return mae, rmse, r2, acc, y_true_int, y_pred, y_pred_round


def train_eval(name, estimator, preprocess, X_train, X_test, y_train, y_test, show_first_n=5):
    """Train a pipeline (preprocess + estimator) and print evaluation."""
    model = Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, rmse, r2, acc, y_true_int, y_pred_raw, y_pred_round = evaluate(y_test, y_pred)

    print(f"===== {name} =====")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print(f"Exact MOS Accuracy (rounded): {acc:.4f}")

    compare = pd.DataFrame({
        "MOS_true": y_true_int,
        "MOS_pred_raw": y_pred_raw,
        "MOS_pred_rounded": y_pred_round
    })
    print(f"\n===== Sample Predictions (first {show_first_n}) =====")
    print(compare.head(show_first_n).to_string(index=False))

    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "ExactAccuracy": acc}


def compare_models(csv_path="pokemon.csv", test_size=0.2, random_state=42, show_first_n=5):
    """
    Use ALL CSV columns except MOS, id, user_id as features.
    Automatically one-hot encode categorical columns.
    """
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c not in ([TARGET_COL] + DROP_COLS)]
    data = df[feature_cols + [TARGET_COL]].copy()

    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL])

    X = data[feature_cols]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocess_tree = build_preprocess(X_train, scale_numeric=False)
    preprocess_scaled = build_preprocess(X_train, scale_numeric=True)

    rf = RandomForestRegressor(n_estimators=300, random_state=random_state, min_samples_leaf=3, n_jobs=-1)
    dt = DecisionTreeRegressor(random_state=random_state, min_samples_leaf=3)
    svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
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

    results = []
    results.append(train_eval("RandomForestRegressor (all features)", rf, preprocess_tree, X_train, X_test, y_train, y_test, show_first_n))
    print("\n" + "=" * 60 + "\n")
    results.append(train_eval("DecisionTreeRegressor (all features)", dt, preprocess_tree, X_train, X_test, y_train, y_test, show_first_n))
    print("\n" + "=" * 60 + "\n")
    results.append(train_eval("SVR_RBF (all features)", svr, preprocess_scaled, X_train, X_test, y_train, y_test, show_first_n))
    print("\n" + "=" * 60 + "\n")
    results.append(train_eval("MLPRegressor (all features)", mlp, preprocess_scaled, X_train, X_test, y_train, y_test, show_first_n))

    summary = pd.DataFrame(results)
    print("\n===== Model Comparison (Test Set) =====")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    compare_models("pokemon.csv", show_first_n=5)
