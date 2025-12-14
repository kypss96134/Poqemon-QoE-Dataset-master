import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a missing-value report:
    - missing_count: number of missing values per column
    - missing_rate: fraction of missing values per column
    """
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean(),
        "dtype": df.dtypes.astype(str),
    }).sort_values(["missing_rate", "missing_count"], ascending=False)

    return report


if __name__ == "__main__":
    df = pd.read_csv("pokemon.csv")  # adjust path if needed
    report = missing_report(df)

    print("=== Missing report ===")
    print(report)

    # Optional: only show columns that actually have missing values
    # print("\n=== Columns with missing values ===")
    # print(report[report["missing_count"] > 0])
