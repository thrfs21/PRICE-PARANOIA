from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = {"date", "product_id", "price"}

def load_price_history(path: str) -> pd.DataFrame:
    """Load and standardise the price history CSV."""
    df = pd.read_csv(path)
    # Accept the supplied dataset's native price_inr column as well as generic price.
    if "price" not in df.columns and "price_inr" in df.columns:
        df = df.rename(columns={"price_inr": "price"})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    return clean_price_history(df)

def clean_price_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy; do not mutate the caller's data."""
    out = df.copy()
    if "price" not in out.columns and "price_inr" in out.columns:
        out = out.rename(columns={"price_inr": "price"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["product_id"] = out["product_id"].astype("string").str.strip()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    # Standardise optional columns when present.
    for col in ("product_name", "category", "currency", "source"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    out = out.dropna(subset=["date", "product_id", "price"])
    out = out[(out["product_id"] != "") & (out["price"] > 0)]
    out = out.drop_duplicates(subset=["date", "product_id"], keep="last")
    out = out.sort_values(["product_id", "date"]).reset_index(drop=True)
    return out

def data_quality_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Compare raw vs cleaned data for an explainable data-quality report."""
    required = [c for c in ("date", "product_id", "price") if c in raw.columns]
    invalid_dates = int(pd.to_datetime(raw["date"], errors="coerce").isna().sum()) if "date" in raw else 0
    price_col = "price" if "price" in raw.columns else ("price_inr" if "price_inr" in raw.columns else None)
    invalid_prices = int(pd.to_numeric(raw[price_col], errors="coerce").isna().sum()) if price_col else 0
    nonpositive_prices = int((pd.to_numeric(raw[price_col], errors="coerce") <= 0).fillna(False).sum()) if price_col else 0
    duplicate_keys = int(raw.duplicated(subset=[c for c in ("date", "product_id") if c in raw.columns]).sum()) if len(required) >= 2 else 0
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "removed_rows": int(max(len(raw) - len(clean), 0)),
        "duplicate_keys": duplicate_keys,
        "invalid_dates": invalid_dates,
        "invalid_prices": invalid_prices,
        "nonpositive_prices": nonpositive_prices,
        "missing_cells": int(raw.isna().sum().sum()),
        "products": int(clean["product_id"].nunique()) if "product_id" in clean else 0,
    }
