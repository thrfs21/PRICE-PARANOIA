from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# Project paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "price_history.csv"

sys.path.insert(0, str(ROOT / "src"))

from src.data_cleaner import (
    load_price_history,
    clean_price_history,
    data_quality_summary,
)

from src.price_analyzer import (
    product_summary,
    deal_score,
    detect_iqr_anomalies,
    discount_assessment,
)


# Page configuration
st.set_page_config(
    page_title="PRICE PARANOIA",
    page_icon="🕵️",
    layout="wide",
)


# Basic styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
    }
    [data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Header
st.title("🕵️ PRICE PARANOIA")
st.caption(
    "Price Intelligence | Investigate the deal, "
    "not just the discount label."
)

st.info(
    "This project uses simulated educational price data, "
    "not actual retailer price histories."
)


# Load and clean data
if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Make sure "
        "price_history.csv is inside the data folder."
    )
    st.stop()

try:
    raw_df = load_price_history(DATA_PATH)
    df = clean_price_history(raw_df)
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

if df.empty:
    st.error("No valid records found in the dataset.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

quality = data_quality_summary(raw_df, df)


# Sidebar filters
st.sidebar.title("Explore Products")

categories = [
    "All categories"
] + sorted(df["category"].dropna().unique().tolist())

selected_category = st.sidebar.selectbox(
    "Category",
    categories,
)

if selected_category == "All categories":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["category"] == selected_category
    ].copy()

product_options = (
    filtered_df[
        ["product_id", "product_name"]
    ]
    .drop_duplicates()
    .sort_values("product_name")
)

if product_options.empty:
    st.warning("No products available in this category.")
    st.stop()

labels = {
    f"{row.product_name} ({row.product_id})": row.product_id
    for row in product_options.itertuples()
}

selected_label = st.sidebar.selectbox(
    "Choose a product",
    list(labels.keys()),
)

selected_id = labels[selected_label]

product_df = (
    df[df["product_id"] == selected_id]
    .sort_values("date")
    .copy()
)

summary = product_summary(product_df)

latest = product_df.iloc[-1]
current_price = float(latest["price_inr"])
mrp = float(latest["mrp_inr"])

score = deal_score(
    current_price,
    product_df["price_inr"],
)

assessment = discount_assessment(
    current_price,
    mrp,
    summary["median"],
)


# Overview
st.header("Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Price",
    f"₹{current_price:,.0f}",
)

col2.metric(
    "Historical Median",
    f"₹{summary['median']:,.0f}",
)

col3.metric(
    "Historical Lowest",
    f"₹{summary['minimum']:,.0f}",
)

col4.metric(
    "Deal Score",
    f"{score}/100" if score is not None else "N/A",
)

st.caption(
    "The deal score describes the current price's position "
    "within the observed historical range. It does not predict "
    "future prices."
)


# Dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Price History",
        "🔎 Discount Checker",
        "⚖️ Compare Products",
        "🧹 Data Quality",
    ]
)


# Tab 1: Price history
with tab1:
    st.subheader(f"Price History: {latest['product_name']}")

    chart_df = detect_iqr_anomalies(product_df)

    fig = px.line(
        chart_df,
        x="date",
        y="price_inr",
        markers=True,
        title="Historical Selling Price",
        labels={
            "date": "Date",
            "price_inr": "Price (₹)",
        },
    )

    anomalies = chart_df[
        chart_df["is_anomaly"]
    ]

    if not anomalies.empty:
        fig.add_scatter(
            x=anomalies["date"],
            y=anomalies["price_inr"],
            mode="markers",
            marker={
                "size": 11,
                "symbol": "x",
            },
            name="Potential anomaly",
        )

    fig.update_layout(hovermode="x unified")

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Historical Maximum",
        f"₹{summary['maximum']:,.0f}",
    )

    c2.metric(
        "Historical Average",
        f"₹{summary['mean']:,.0f}",
    )

    c3.metric(
        "Potential Anomalies",
        len(anomalies),
    )

    if not anomalies.empty:
        st.warning(
            "These are statistical flags, not proof of "
            "fraud or incorrect pricing."
        )
        st.dataframe(
            anomalies[
                ["date", "price_inr", "is_anomaly"]
            ],
            use_container_width=True,
            hide_index=True,
        )


# Tab 2: Discount checker
with tab2:
    st.subheader("Is This Discount Actually Good?")

    c1, c2 = st.columns(2)

    c1.metric(
        "Illustrative MRP",
        f"₹{mrp:,.0f}",
        f"{assessment['advertised_discount_pct']:.1f}% below MRP",
    )

    c2.metric(
        "Current vs Historical Median",
        f"{assessment['vs_median_pct']:+.1f}%",
    )

    verdict = assessment["verdict"]

    if verdict == "Below usual price":
        st.success(verdict)
    elif verdict == "Above usual price":
        st.warning(verdict)
    else:
        st.info(verdict)

    st.write(
        "Compare the current price against its historical "
        "median as well as the listed MRP. A large MRP-based "
        "discount alone does not establish that a retailer "
        "has misled customers."
    )


# Tab 3: Compare products
with tab3:
    st.subheader("Compare Products")

    latest_records = (
        df.sort_values("date")
        .groupby("product_id")
        .tail(1)
        .copy()
    )

    latest_records = latest_records[
        latest_records["product_id"].isin(
            filtered_df["product_id"].unique()
        )
    ]

    median_prices = df.groupby(
        "product_id"
    )["price_inr"].median()

    score_lookup = {}

    for product_id, group in df.groupby("product_id"):
        group = group.sort_values("date")
        score_lookup[product_id] = deal_score(
            group.iloc[-1]["price_inr"],
            group["price_inr"],
        )

    latest_records["historical_median_inr"] = (
        latest_records["product_id"].map(median_prices)
    )

    latest_records["deal_score"] = (
        latest_records["product_id"].map(score_lookup)
    )

    comparison = latest_records[
        [
            "product_name",
            "category",
            "date",
            "price_inr",
            "mrp_inr",
            "historical_median_inr",
            "deal_score",
        ]
    ].sort_values(
        "deal_score",
        ascending=False,
    )

    st.dataframe(
        comparison.rename(
            columns={
                "product_name": "Product",
                "category": "Category",
                "date": "Latest Date",
                "price_inr": "Latest Price (₹)",
                "mrp_inr": "MRP (₹)",
                "historical_median_inr": "Median Price (₹)",
                "deal_score": "Deal Score",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        comparison,
        x="product_name",
        y="deal_score",
        color="category",
        title="Deal Score by Product",
        labels={
            "product_name": "Product",
            "deal_score": "Deal Score",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# Tab 4: Data quality
with tab4:
    st.subheader("Dataset Health")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Raw Rows",
        f"{quality['raw_rows']:,}",
    )

    c2.metric(
        "Clean Rows",
        f"{quality['clean_rows']:,}",
    )

    c3.metric(
        "Rows Removed",
        f"{quality['rows_removed']:,}",
    )

    c4.metric(
        "Missing Raw Prices",
        f"{quality['missing_prices']:,}",
    )

    st.write(
        "Cleaning operations include type conversion, "
        "whitespace normalization, invalid-value removal, "
        "and duplicate removal."
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )


# Footer
st.divider()

st.caption(
    "PRICE PARANOIA | DAV LA-2 | "
    "Python · Pandas · Streamlit · Plotly"
)