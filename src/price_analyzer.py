from __future__ import annotations
import numpy as np
import pandas as pd

def product_summary(product_df: pd.DataFrame) -> dict:
    if product_df.empty:
        return {}
    df = product_df.sort_values("date")
    prices = pd.to_numeric(df["price"], errors="coerce").dropna()
    if prices.empty:
        return {}
    current = float(prices.iloc[-1])
    low = float(prices.min())
    high = float(prices.max())
    avg = float(prices.mean())
    median = float(prices.median())
    pct_vs_avg = ((current / avg) - 1) * 100 if avg else 0.0
    return {
        "current_price": current, "lowest_price": low, "highest_price": high,
        "average_price": avg, "median_price": median,
        "price_change_pct": ((current / float(prices.iloc[0])) - 1) * 100 if len(prices) > 1 and prices.iloc[0] else 0.0,
        "pct_vs_average": pct_vs_avg,
        "observations": int(len(prices)),
        "volatility_pct": float(prices.std(ddof=0) / avg * 100) if avg else 0.0,
        "last_date": pd.to_datetime(df["date"].iloc[-1]),
    }

def deal_score(product_df: pd.DataFrame) -> dict:
    """Transparent 0-100 score based on current-vs-average, current-vs-low and recent trend."""
    df = product_df.sort_values("date").copy()
    if df.empty:
        return {"score": 0, "verdict": "Insufficient data", "reasons": []}
    prices = pd.to_numeric(df["price"], errors="coerce").dropna()
    if len(prices) < 5:
        return {"score": 0, "verdict": "Insufficient data", "reasons": ["At least 5 observations are recommended."]}
    current = float(prices.iloc[-1])
    avg = float(prices.mean())
    low = float(prices.min())
    recent = prices.tail(min(14, len(prices)))
    previous = prices.iloc[-min(28, len(prices)):-min(14, len(prices))]
    recent_avg = float(recent.mean())
    prev_avg = float(previous.mean()) if len(previous) else avg

    # Components are clipped to avoid extreme values dominating the score.
    vs_avg = np.clip((avg-current) / avg * 100, -20, 35)
    avg_component = float(np.clip(50 + vs_avg * 1.4, 0, 100))
    low_gap = ((current-low) / low * 100) if low else 0
    low_component = float(np.clip(100 - low_gap * 2.2, 0, 100))
    trend_pct = ((recent_avg/prev_avg)-1)*100 if prev_avg else 0
    trend_component = float(np.clip(50 - trend_pct*2.0, 0, 100))
    score = int(round(0.55*avg_component + 0.25*low_component + 0.20*trend_component))
    if score >= 75:
        verdict = "Great deal"
    elif score >= 55:
        verdict = "Worth comparing"
    elif score >= 40:
        verdict = "Fair price"
    else:
        verdict = "Consider waiting"
    reasons = [
        f"Current price is {abs((current/avg-1)*100):.1f}% {'below' if current < avg else 'above'} its historical average.",
        f"Current price is {low_gap:.1f}% above the lowest recorded price.",
        f"Recent average is {abs(trend_pct):.1f}% {'higher' if trend_pct > 0 else 'lower'} than the preceding period."
    ]
    return {"score": score, "verdict": verdict, "reasons": reasons,
            "vs_average_pct": (current/avg-1)*100 if avg else 0,
            "above_low_pct": low_gap, "recent_trend_pct": trend_pct}

def discount_assessment(product_df: pd.DataFrame, advertised_price: float | None = None,
                        advertised_reference_price: float | None = None) -> dict:
    df = product_df.sort_values("date")
    prices = pd.to_numeric(df["price"], errors="coerce").dropna()
    if prices.empty:
        return {"message": "Insufficient price history."}
    current = float(advertised_price if advertised_price is not None else prices.iloc[-1])
    avg = float(prices.mean())
    low = float(prices.min())
    high = float(prices.max())
    ref = float(advertised_reference_price) if advertised_reference_price and advertised_reference_price > 0 else high
    advertised_pct = max(0.0, (ref-current)/ref*100) if ref else 0.0
    historical_pct = max(0.0, (avg-current)/avg*100) if avg else 0.0
    return {
        "current_price": current, "average_price": avg, "lowest_price": low, "highest_price": high,
        "advertised_reference": ref, "advertised_discount_pct": advertised_pct,
        "below_average_pct": historical_pct,
        "difference_pp": advertised_pct-historical_pct,
        "verdict": "Historically low" if current <= low*1.03 else ("Below average" if current < avg else "At/above average"),
        "note": "Historical comparisons are evidence, not proof of misleading pricing."
    }

def detect_iqr_anomalies(product_df: pd.DataFrame) -> pd.DataFrame:
    df = product_df.copy().sort_values("date")
    if df.empty:
        df["is_anomaly"] = False
        return df
    q1 = df["price"].quantile(.25); q3 = df["price"].quantile(.75)
    iqr = q3-q1
    low = q1-1.5*iqr; high = q3+1.5*iqr
    df["is_anomaly"] = (df["price"] < low) | (df["price"] > high)
    df["iqr_lower_bound"] = low; df["iqr_upper_bound"] = high
    return df

def forecast_prices(product_df: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """Simple linear trend forecast with a residual-based interval; educational baseline, not a guarantee."""
    df = product_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    if len(df) < 10:
        return pd.DataFrame(columns=["date", "predicted_price", "lower_bound", "upper_bound"])
    # Use last 60 observations to reduce the influence of old regime changes.
    train = df.tail(60)
    x = (train["date"] - train["date"].min()).dt.days.to_numpy(dtype=float)
    y = train["price"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope*x + intercept
    residual_std = float(np.std(y-fitted)) if len(y) > 2 else 0.0
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=int(horizon), freq="D")
    fx = (future_dates - train["date"].min()).days.to_numpy(dtype=float)
    pred = np.maximum(slope*fx + intercept, 0.01)
    # Approximate 80% band; use a minimum small band to avoid implying certainty.
    band = max(1.28*residual_std, float(np.mean(y))*0.01)
    return pd.DataFrame({"date": future_dates, "predicted_price": pred,
                         "lower_bound": np.maximum(pred-band, 0.01), "upper_bound": pred+band})
