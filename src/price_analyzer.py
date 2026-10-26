import numpy as np
import pandas as pd


def product_summary(product_df):
    """Calculate historical price statistics."""
    prices = pd.to_numeric(
        product_df["price_inr"], errors="coerce"
    ).dropna()

    if prices.empty:
        return {}

    latest = product_df.sort_values("date").iloc[-1]

    return {
        "observations": len(prices),
        "current_price": float(latest["price_inr"]),
        "minimum": float(prices.min()),
        "maximum": float(prices.max()),
        "mean": float(prices.mean()),
        "median": float(prices.median()),
    }


def deal_score(current_price, historical_prices):
    """
    Score from 0 to 100.
    Higher means cheaper relative to the observed range.
    This is a descriptive heuristic, not a prediction.
    """
    prices = pd.Series(historical_prices).dropna().astype(float)

    if prices.empty:
        return None

    lowest = prices.min()
    highest = prices.max()

    if np.isclose(lowest, highest):
        return 50

    score = (
        (highest - current_price)
        / (highest - lowest)
    ) * 100

    return int(round(np.clip(score, 0, 100)))


def detect_iqr_anomalies(product_df):
    """Detect unusually high or low prices using IQR."""
    result = product_df.copy()

    prices = pd.to_numeric(
        result["price_inr"], errors="coerce"
    )

    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or np.isclose(iqr, 0):
        result["is_anomaly"] = False
    else:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        result["is_anomaly"] = (
            (prices < lower_bound)
            | (prices > upper_bound)
        )

    return result


def discount_assessment(
    current_price,
    mrp,
    historical_median,
):
    """Compare the current price with MRP and historical median."""

    if mrp and mrp > 0:
        discount_pct = (
            (mrp - current_price) / mrp
        ) * 100
    else:
        discount_pct = 0

    if historical_median and historical_median > 0:
        median_difference = (
            (current_price - historical_median)
            / historical_median
        ) * 100
    else:
        median_difference = 0

    if median_difference <= -8:
        verdict = "Below usual price"
    elif median_difference >= 8:
        verdict = "Above usual price"
    else:
        verdict = "Near usual price"

    return {
        "advertised_discount_pct": round(
            max(0, discount_pct), 2
        ),
        "vs_median_pct": round(
            median_difference, 2
        ),
        "verdict": verdict,
    }