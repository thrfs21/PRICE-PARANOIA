import pandas as pd


def load_price_history(path="data/price_history.csv"):
    """Load the historical price dataset."""
    df = pd.read_csv(path)

    required_columns = [
        "product_id",
        "product_name",
        "category",
        "platform",
        "date",
        "price_inr",
        "mrp_inr",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def clean_price_history(df):
    """Clean and validate product price records."""
    cleaned = df.copy()

    # Standardize text
    for column in ["product_name", "category", "platform"]:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

    # Convert columns to correct data types
    cleaned["date"] = pd.to_datetime(
        cleaned["date"], errors="coerce"
    )

    cleaned["price_inr"] = pd.to_numeric(
        cleaned["price_inr"], errors="coerce"
    )

    cleaned["mrp_inr"] = pd.to_numeric(
        cleaned["mrp_inr"], errors="coerce"
    )

    # Remove records missing essential values
    cleaned = cleaned.dropna(
        subset=[
            "product_id",
            "product_name",
            "date",
            "price_inr",
            "mrp_inr",
        ]
    )

    # Remove invalid prices
    cleaned = cleaned[
        (cleaned["price_inr"] > 0)
        & (cleaned["mrp_inr"] > 0)
    ]

    # Remove duplicate observations
    cleaned = cleaned.drop_duplicates(
        subset=["product_id", "date", "platform"],
        keep="last",
    )

    # Sort chronologically
    cleaned = cleaned.sort_values(
        ["product_id", "date"]
    ).reset_index(drop=True)

    return cleaned


def data_quality_summary(raw_df, cleaned_df):
    """Return basic data quality statistics."""
    return {
        "raw_rows": len(raw_df),
        "clean_rows": len(cleaned_df),
        "rows_removed": len(raw_df) - len(cleaned_df),
        "missing_prices": pd.to_numeric(
            raw_df["price_inr"], errors="coerce"
        ).isna().sum(),
        "duplicate_rows": raw_df.duplicated().sum(),
    }