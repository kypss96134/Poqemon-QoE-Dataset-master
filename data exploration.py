import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _encode_columns_with_mapping(
    df: pd.DataFrame,
    cols: list[str],
    out_dir: str,
    prefix: str = "enc_",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Encode columns using pd.factorize and save a mapping CSV for each encoded column.
    Works for object/category columns and also integer-coded categorical columns.

    Returns:
      - df with new encoded columns added
      - mapping dict: original_col -> encoded_col
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    enc_map: dict[str, str] = {}

    for col in cols:
        if col not in df.columns:
            print(f"[ENC] Skip (missing): {col}")
            continue

        # Factorize on string representation to support mixed dtypes safely
        codes, uniques = pd.factorize(df[col].astype(str), sort=True)
        new_col = f"{prefix}{col}"
        df[new_col] = codes

        mapping_df = pd.DataFrame({"code": range(len(uniques)), "category": uniques})
        mapping_path = os.path.join(out_dir, f"{new_col}_mapping.csv")
        mapping_df.to_csv(mapping_path, index=False)

        enc_map[col] = new_col
        print(f"[ENC] {col} -> {new_col} (saved mapping: {mapping_path})")

    return df, enc_map


def _plot_group_by_mos_boxplots(
    df: pd.DataFrame,
    group_name: str,
    cols: list[str],
    out_dir: str,
    mos_col: str = "MOS",
    mos_order: list[int] | None = None,
    logy_cols: set[str] | None = None,
    dpi: int = 300,
    # NEW: categorical encoding controls
    encode_object_and_category: bool = True,
    encode_low_cardinality_int: bool = True,
    low_cardinality_threshold: int = 20,
) -> None:
    """
    Create MOS-grouped boxplots for the given columns.
    Enhancements:
      - Encodes ALL categorical columns using factorize and saves mapping CSVs:
          * object/category dtypes
          * optionally low-cardinality integer columns (<= threshold unique values)
      - Uses Matplotlib 3.9+ tick_labels
      - Skips empty MOS groups safely
    """
    os.makedirs(out_dir, exist_ok=True)

    if mos_order is None:
        mos_order = [1, 2, 3, 4, 5]
    if logy_cols is None:
        logy_cols = set()

    if mos_col not in df.columns:
        raise ValueError(f"[{group_name}] Missing MOS column: {mos_col}")

    df = df.copy()
    df[mos_col] = pd.to_numeric(df[mos_col], errors="coerce").astype("Int64")

    # ---------- NEW: discover which columns to encode ----------
    cols_to_encode: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue

        dtype = df[c].dtype

        # object/category -> encode
        if encode_object_and_category and (dtype == "object" or str(dtype) == "category"):
            cols_to_encode.append(c)
            continue

        # integer-coded categorical -> encode + save mapping (optional)
        if encode_low_cardinality_int and pd.api.types.is_integer_dtype(dtype):
            nunique = df[c].nunique(dropna=True)
            if nunique <= low_cardinality_threshold:
                cols_to_encode.append(c)

    # Encode and replace columns with encoded versions for plotting
    if cols_to_encode:
        df, enc_map = _encode_columns_with_mapping(df, cols_to_encode, out_dir=out_dir)
        cols = [enc_map.get(c, c) for c in cols]
        # logy columns should apply to encoded names if needed (usually not),
        # so we keep logy_cols only for truly continuous variables.

    # ---------- Plotting ----------
    saved = 0
    for col in cols:
        if col not in df.columns:
            print(f"[{group_name}] Skip (missing): {col}")
            continue

        plot_df = df[[col, mos_col]].copy()
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=[col, mos_col])

        # log scale y only for selected continuous variables
        if col in logy_cols:
            plot_df = plot_df[plot_df[col] > 0]

        # build grouped arrays; skip empty groups
        data, tick_labels = [], []
        for m in mos_order:
            vals = plot_df.loc[plot_df[mos_col] == m, col].values
            if len(vals) == 0:
                continue
            data.append(vals)
            tick_labels.append(m)

        if not data:
            print(f"[{group_name}] Skip (no valid data): {col}")
            continue

        plt.figure()
        plt.boxplot(data, tick_labels=tick_labels, showfliers=False)  # Matplotlib 3.9+
        plt.xlabel("MOS (1=Bad ... 5=Excellent)")
        plt.ylabel(col)
        plt.title(f"{group_name}: {col} by MOS")

        if col in logy_cols:
            plt.yscale("log")

        plt.tight_layout()
        suffix = "_logy" if col in logy_cols else ""
        out_path = os.path.join(out_dir, f"{col}_by_MOS_boxplot{suffix}.png")
        plt.savefig(out_path, dpi=dpi)
        plt.close()
        saved += 1

    print(f"[{group_name}] Saved {saved} boxplots to ./{out_dir}/")


# ---------------- Group functions ----------------

def plot_qoa_by_mos_boxplots(csv_path="pokemon.csv", out_dir="figures/figure_QoA", mos_col="MOS", dpi=300):
    df = pd.read_csv(csv_path)
    qoa_cols = [
        "QoA_VLCresolution", "QoA_VLCbitrate", "QoA_VLCframerate", "QoA_VLCdropped",
        "QoA_VLCaudiorate", "QoA_VLCaudioloss", "QoA_BUFFERINGcount", "QoA_BUFFERINGtime",
    ]
    logy_cols = {"QoA_VLCbitrate", "QoA_BUFFERINGtime"}  # continuous, wide range
    _plot_group_by_mos_boxplots(df, "QoA", qoa_cols, out_dir, mos_col=mos_col, logy_cols=logy_cols, dpi=dpi)


def plot_qos_by_mos_boxplots(csv_path="pokemon.csv", out_dir="figures/figure_QoS", mos_col="MOS", dpi=300):
    df = pd.read_csv(csv_path)
    qos_cols = ["QoS_type", "QoS_operator"]
    _plot_group_by_mos_boxplots(df, "QoS", qos_cols, out_dir, mos_col=mos_col, dpi=dpi)


def plot_qod_by_mos_boxplots(csv_path="pokemon.csv", out_dir="figures/figure_QoD", mos_col="MOS", dpi=300):
    df = pd.read_csv(csv_path)
    qod_cols = ["QoD_model", "QoD_os-version", "QoD_api-level"]
    _plot_group_by_mos_boxplots(df, "QoD", qod_cols, out_dir, mos_col=mos_col, dpi=dpi)


def plot_qou_by_mos_boxplots(csv_path="pokemon.csv", out_dir="figures/figure_QoU", mos_col="MOS", dpi=300):
    df = pd.read_csv(csv_path)
    # your column name seems to be QoU_Ustedy in the file, not QoU_study
    qou_cols = ["QoU_sex", "QoU_age", "QoU_Ustedy"]
    _plot_group_by_mos_boxplots(df, "QoU", qou_cols, out_dir, mos_col=mos_col, dpi=dpi)


def plot_qof_by_mos_boxplots(csv_path="pokemon.csv", out_dir="figures/figure_QoF", mos_col="MOS", dpi=300):
    df = pd.read_csv(csv_path)
    # your file seems to have QoF_begin (not QoF_begin_time)
    qof_cols = ["QoF_begin", "QoF_shift", "QoF_audio", "QoF_video"]
    _plot_group_by_mos_boxplots(df, "QoF", qof_cols, out_dir, mos_col=mos_col, dpi=dpi)


def plot_correlation_matrix_selected_features_with_os_encoding(
    csv_path: str = "pokemon.csv",
    fig_dir: str = "figures",
    out_name: str = "correlation_matrix_selected_features.png",
    method: str = "spearman",   # "pearson" or "spearman"
    figsize: tuple[int, int] = (7, 6),
    tick_fontsize: int = 8,
    annot_fontsize: int = 7,
    title_fontsize: int = 10,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Correlation matrix for selected features + MOS.
    If QoD_os-version is categorical (object/category), it will be factor-encoded and a mapping CSV is saved.
    """

    df = pd.read_csv(csv_path)

    features = [
        "MOS",
        "QoA_VLCbitrate",
        "QoA_BUFFERINGtime",
        "QoA_VLCaudiorate",
        "QoA_VLCframerate",
        "QoD_os-version",   # will be encoded if non-numeric
        "QoF_audio",
        "QoF_begin",
        "QoF_video",
    ]

    os.makedirs(fig_dir, exist_ok=True)

    # --- Encode QoD_os-version if it is categorical ---
    os_col = "QoD_os-version"
    if os_col in df.columns and (df[os_col].dtype == "object" or str(df[os_col].dtype) == "category"):
        codes, uniques = pd.factorize(df[os_col].astype(str), sort=True)
        df[os_col] = codes  # replace with numeric codes

        mapping_df = pd.DataFrame({"code": range(len(uniques)), "category": uniques})
        mapping_path = os.path.join(fig_dir, "enc_QoD_os-version_mapping.csv")
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Saved os-version mapping to: {mapping_path}")

    # Convert selected columns to numeric
    corr_df = df[features].apply(pd.to_numeric, errors="coerce")

    # Compute correlation matrix
    corr_matrix = corr_df.corr(method=method)

    # Plot heatmap (Matplotlib only)
    plt.figure(figsize=figsize)
    im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    plt.xticks(
        ticks=np.arange(len(corr_matrix.columns)),
        labels=corr_matrix.columns,
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )
    plt.yticks(
        ticks=np.arange(len(corr_matrix.index)),
        labels=corr_matrix.index,
        fontsize=tick_fontsize,
    )

    # Annotate values
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            v = corr_matrix.iloc[i, j]
            if pd.notna(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=annot_fontsize, color="black")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Correlation Matrix Between Selected Features and MOS", fontsize=title_fontsize)
    plt.tight_layout()

    out_path = os.path.join(fig_dir, out_name)
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    print(f"Saved correlation matrix to: {out_path}")
    return corr_matrix


if __name__ == "__main__":
    # step 1: boxplots by MOS
    # plot_qoa_by_mos_boxplots()
    # plot_qos_by_mos_boxplots()
    # plot_qod_by_mos_boxplots()
    # plot_qou_by_mos_boxplots()
    # plot_qof_by_mos_boxplots()

    # step 2: correlation matrix
    plot_correlation_matrix_selected_features_with_os_encoding(
        csv_path="pokemon.csv",
        fig_dir="figures",
        out_name="correlation_matrix_selected_features_with_os_encoding.png",
        method="spearman",
    )
