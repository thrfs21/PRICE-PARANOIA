from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_cleaner import load_price_history, clean_price_history, data_quality_summary
from src.price_analyzer import product_summary, deal_score, discount_assessment, detect_iqr_anomalies, forecast_prices

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "price_history.csv"
CATALOG_PATH = ROOT / "data" / "product_catalog.csv"
WATCHLIST_PATH = ROOT / "watchlist.json"

st.set_page_config(page_title="PRICE PARANOIA", page_icon="📉", layout="wide", initial_sidebar_state="expanded")

# ---------- Design system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#0b1020; --panel:#141b2f; --panel2:#1b2540; --purple:#9b7bff; --cyan:#47e1d1; --muted:#9aa8c7; }
.stApp { background: radial-gradient(circle at 80% 0%, #202044 0, #0b1020 42%); color:#f5f7ff; font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
.block-container { padding-top:1.7rem; padding-bottom:3rem; max-width:1500px; }
[data-testid="stSidebar"] { background:#0e1427; border-right:1px solid #27314c; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#18213a,#12182a); border:1px solid #2a3553; padding:16px 18px; border-radius:18px; }
[data-testid="stMetricLabel"] { color:#aab6d1 !important; }
[data-testid="stMetricValue"] { color:#f5f7ff !important; font-family:'Space Grotesk',sans-serif; }
div[data-testid="stTabs"] button { border-radius:12px; font-weight:600; }
div[data-testid="stTabs"] button[aria-selected="true"] { background:#282449; color:#bba8ff; }
div.stButton > button { border:1px solid #514581; background:linear-gradient(90deg,#7958df,#a06dff); color:white; border-radius:12px; font-weight:700; }
div.stButton > button:hover { border-color:#47e1d1; color:white; }
div[data-testid="stAlert"] { border-radius:14px; }
.pp-hero { padding:26px 30px; border-radius:24px; background:linear-gradient(120deg,rgba(122,88,224,.22),rgba(71,225,209,.07)); border:1px solid #30385b; margin-bottom:20px; }
.pp-eyebrow { color:#47e1d1; font-size:12px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.pp-title { font-family:'Space Grotesk',sans-serif; font-size:40px; font-weight:700; line-height:1.05; margin:8px 0; }
.pp-sub { color:#aab6d1; font-size:15px; max-width:850px; }
.pp-chip { display:inline-block; padding:6px 10px; border-radius:99px; background:#242b49; color:#c7bcff; font-size:11px; font-weight:700; margin-right:5px; }
.pp-panel { padding:18px 20px; border:1px solid #2a3553; background:rgba(20,27,47,.9); border-radius:18px; }
.small-muted { color:#9aa8c7; font-size:12px; }
</style>
""", unsafe_allow_html=True)

def money(value):
    return f"₹{value:,.0f}"

def load_catalog():
    if CATALOG_PATH.exists():
        c = pd.read_csv(CATALOG_PATH)
        return c
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_data():
    raw = pd.read_csv(DATA_PATH)
    clean = load_price_history(str(DATA_PATH))
    return raw, clean

def load_watchlist():
    try:
        return json.loads(WATCHLIST_PATH.read_text(encoding="utf-8")) if WATCHLIST_PATH.exists() else {}
    except Exception:
        return {}

def save_watchlist(data):
    WATCHLIST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}. Place price_history.csv inside the data folder.")
    st.stop()

raw_df, df = get_data()
catalog = load_catalog()
if "product_id" not in df:
    st.error("Dataset needs a product_id column.")
    st.stop()

# Merge catalog metadata where available
if not catalog.empty and "product_id" in catalog.columns:
    merge_cols = [c for c in catalog.columns if c == "product_id" or c not in df.columns]
    df = df.merge(catalog[merge_cols], on="product_id", how="left", suffixes=("", "_catalog"))
if "product_name" not in df.columns:
    df["product_name"] = df["product_id"].astype(str)
df["product_name"] = df["product_name"].fillna(df["product_id"].astype(str))
if "category" not in df.columns:
    df["category"] = "Uncategorised"
df["category"] = df["category"].fillna("Uncategorised").astype(str)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["date", "price", "product_id"]).sort_values(["product_id", "date"])

# Sidebar filters
st.sidebar.markdown("## PRICE **PARANOIA**")
st.sidebar.caption("Price intelligence • DAV mini project")
categories = ["All categories"] + sorted(df["category"].dropna().astype(str).unique().tolist())
category = st.sidebar.selectbox("Category", categories)
filtered = df if category == "All categories" else df[df["category"] == category]
products = filtered[["product_id", "product_name"]].drop_duplicates().sort_values("product_name")
if products.empty:
    st.warning("No products found for this category.")
    st.stop()
product_options = {f"{r.product_name} · {r.product_id}": r.product_id for r in products.itertuples()}
selected_label = st.sidebar.selectbox("Focus product", list(product_options.keys()))
selected_id = product_options[selected_label]
product_df = filtered[filtered["product_id"].astype(str) == str(selected_id)].sort_values("date")
if product_df.empty:
    product_df = df[df["product_id"].astype(str) == str(selected_id)].sort_values("date")

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Analysis period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    date_start, date_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    date_start, date_end = pd.Timestamp(min_date), pd.Timestamp(max_date)
period_df = product_df[(product_df["date"] >= date_start) & (product_df["date"] <= date_end)]
if period_df.empty:
    period_df = product_df

summary = product_summary(product_df)
score_info = deal_score(product_df)
quality = data_quality_summary(raw_df, clean_price_history(raw_df))
watchlist = load_watchlist()
watch_key = str(selected_id)

# Header
st.markdown(f"""
<div class="pp-hero">
  <div class="pp-eyebrow">DATA ANALYTICS / PRICE INTELLIGENCE</div>
  <div class="pp-title">PRICE <span style="color:#a58aff">PARANOIA</span></div>
  <div class="pp-sub">Don't trust the discount tag. Inspect the data. Track price history, evaluate deal quality, spot anomalies and estimate where prices may go next.</div>
  <div style="margin-top:14px"><span class="pp-chip">HISTORICAL ANALYSIS</span><span class="pp-chip">DEAL INTEGRITY</span><span class="pp-chip">FORECAST BASELINE</span></div>
</div>
""", unsafe_allow_html=True)

# KPI row
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Current price", money(summary.get("current_price",0)))
c2.metric("Historical low", money(summary.get("lowest_price",0)))
c3.metric("Historical average", money(summary.get("average_price",0)))
c4.metric("vs. average", f"{summary.get('pct_vs_average',0):+.1f}%", delta=f"{summary.get('pct_vs_average',0):+.1f}%")
c5.metric("Deal score", f"{score_info.get('score',0)}/100", delta=score_info.get("verdict","—"))

tabs = st.tabs(["Overview", "Deal Integrity", "Forecast", "Compare", "Watchlist", "Data Quality"])

# Overview
with tabs[0]:
    left, right = st.columns([1.8,1])
    with left:
        st.subheader("Price behaviour")
        plot_df = period_df.copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["price"], mode="lines", name="Price",
                                 line=dict(color="#47e1d1", width=3), fill="tozeroy",
                                 fillcolor="rgba(71,225,209,.08)"))
        if len(plot_df) >= 7:
            plot_df["moving_avg"] = plot_df["price"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["moving_avg"], mode="lines", name="7-day average",
                                     line=dict(color="#a58aff", width=2, dash="dot")))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10,r=10,t=20,b=10), height=410, legend=dict(orientation="h",y=1.1),
                          xaxis_title=None,yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Deal score breakdown")
        st.markdown(f"<div class='pp-panel'><div class='pp-eyebrow'>VERDICT</div><div style='font-size:28px;font-weight:700;margin:8px 0'>{score_info.get('verdict','Insufficient data')}</div><div style='font-size:44px;font-weight:700;color:#a58aff'>{score_info.get('score',0)}<span style='font-size:16px;color:#9aa8c7'> / 100</span></div></div>", unsafe_allow_html=True)
        for reason in score_info.get("reasons", []):
            st.markdown(f"• {reason}")
        st.caption("Score is a heuristic, not financial advice or a guarantee of future prices.")
    st.subheader("Recent observations")
    recent = product_df.tail(10).sort_values("date", ascending=False).copy()
    recent["date"] = recent["date"].dt.strftime("%d %b %Y")
    recent["price"] = recent["price"].map(money)
    st.dataframe(recent[["date","product_name","price","category"]], use_container_width=True, hide_index=True)

# Deal integrity
with tabs[1]:
    st.subheader("Discount Reality Checker")
    st.write("Compare a current offer against the selected product's historical prices. Enter a reference price from an advertised offer if you want to test the claimed discount.")
    a,b,c = st.columns(3)
    current_offer = a.number_input("Current offer price (₹)", min_value=1.0, value=float(summary["current_price"]), step=50.0)
    reference_default = max(float(summary["highest_price"]), float(summary["current_price"]))
    reference_price = b.number_input("Advertised reference price (₹)", min_value=1.0, value=float(reference_default), step=50.0)
    c.metric("Claimed discount", f"{max(0,(reference_price-current_offer)/reference_price*100):.1f}%")
    assessment = discount_assessment(product_df, current_offer, reference_price)
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Historical average", money(assessment["average_price"]))
    d2.metric("Historical minimum", money(assessment["lowest_price"]))
    d3.metric("Below average", f"{assessment['below_average_pct']:.1f}%")
    d4.metric("Claim vs history", f"{assessment['difference_pp']:+.1f} pp")
    st.markdown(f"### Historical verdict: **{assessment['verdict']}**")
    if assessment["verdict"] == "Historically low":
        st.success("This offer is close to the lowest observed price in the available history.")
    elif assessment["verdict"] == "Below average":
        st.info("This offer is below the historical average, but compare it with the recorded low.")
    else:
        st.warning("This offer is not below the historical average. Check the trend before deciding.")
    st.caption(assessment["note"])
    st.subheader("Statistical anomaly review")
    anomalies = detect_iqr_anomalies(product_df)
    flagged = anomalies[anomalies["is_anomaly"]]
    if flagged.empty:
        st.success("No IQR outliers detected for this product in the current history.")
    else:
        st.warning(f"{len(flagged)} observation(s) flagged for review. An outlier is not automatically a fraudulent or incorrect price.")
        st.dataframe(flagged[["date","product_name","price","iqr_lower_bound","iqr_upper_bound"]], use_container_width=True, hide_index=True)

# Forecast
with tabs[2]:
    st.subheader("Buy now or wait? — exploratory forecast")
    st.info("This is a simple linear-trend baseline trained on the most recent observations. It is an educational estimate, not a guarantee.")
    horizon = st.slider("Forecast horizon (days)", 7, 14, 7)
    fc = forecast_prices(product_df, horizon)
    hist = product_df.tail(60)
    if fc.empty:
        st.warning("At least 10 valid observations are required to produce a forecast.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="Historical price", line=dict(color="#47e1d1",width=2.5)))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper_bound"], name="Upper band", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower_bound"], name="Forecast range", fill="tonexty",
                                 fillcolor="rgba(155,123,255,.18)", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["predicted_price"], name="Forecast", line=dict(color="#a58aff",width=3,dash="dash")))
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10,r=10,t=20,b=10),height=430,yaxis_title="Price (₹)")
        st.plotly_chart(fig,use_container_width=True)
        last_actual=float(product_df.sort_values("date")["price"].iloc[-1])
        end_pred=float(fc["predicted_price"].iloc[-1])
        x1,x2,x3=st.columns(3)
        x1.metric("Last observed",money(last_actual))
        x2.metric(f"Predicted in {horizon} days",money(end_pred),delta=f"{(end_pred/last_actual-1)*100:+.1f}%")
        x3.metric("Forecast range",f"{money(fc['lower_bound'].iloc[-1])} – {money(fc['upper_bound'].iloc[-1])}")
        st.caption("Forecast uses a linear regression baseline with a residual-based approximate band. Validate it on held-out dates before relying on it.")

# Compare
with tabs[3]:
    st.subheader("Product comparison")
    all_products = df[["product_id","product_name","category"]].drop_duplicates()
    labels = {f"{r.product_name} · {r.product_id}": str(r.product_id) for r in all_products.itertuples()}
    default_labels = list(labels.keys())[:min(3,len(labels))]
    selected_labels = st.multiselect("Choose up to 5 products", list(labels.keys()), default=default_labels, max_selections=5)
    compare_ids = [labels[x] for x in selected_labels]
    compare_df = df[df["product_id"].astype(str).isin(compare_ids)].copy()
    if not compare_df.empty:
        fig = px.line(compare_df, x="date", y="price", color="product_name", template="plotly_dark",
                      title="Historical price comparison")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=430,margin=dict(l=10,r=10,t=45,b=10),yaxis_title="Price (₹)")
        st.plotly_chart(fig,use_container_width=True)
        rows=[]
        for pid in compare_ids:
            p=df[df["product_id"].astype(str)==pid]
            sm=product_summary(p); sc=deal_score(p)
            if sm:
                rows.append({"Product":p["product_name"].iloc[0],"Current":sm["current_price"],"Low":sm["lowest_price"],"Average":sm["average_price"],"Score":sc["score"],"Verdict":sc["verdict"]})
        st.dataframe(pd.DataFrame(rows).style.format({"Current":"₹{:,.0f}","Low":"₹{:,.0f}","Average":"₹{:,.0f}"}),use_container_width=True,hide_index=True)
    else:
        st.info("Select products above to compare.")

# Watchlist
with tabs[4]:
    st.subheader("Watchlist & target-price alerts")
    st.write("Save a target price for a product. The app checks the latest available dataset price whenever you open the dashboard.")
    target = st.number_input("Alert me when price is at or below (₹)", min_value=1.0,
                             value=float(watchlist.get(watch_key, {}).get("target", max(1, summary["current_price"]*0.9))), step=50.0)
    w1,w2=st.columns([1,2])
    with w1:
        if st.button("Save target price",use_container_width=True):
            watchlist[watch_key]={"product_name":str(product_df["product_name"].iloc[0]),"target":float(target)}
            save_watchlist(watchlist)
            st.success("Target saved in watchlist.json.")
            st.rerun()
    with w2:
        if watch_key in watchlist:
            st.metric("Latest available price",money(summary["current_price"]),delta=f"Target {money(watchlist[watch_key]['target'])}")
            if summary["current_price"] <= watchlist[watch_key]["target"]:
                st.success("Target reached in the available dataset.")
            else:
                st.info(f"Target not reached. Difference: {money(summary['current_price']-watchlist[watch_key]['target'])}.")
    if watchlist:
        watch_rows=[]
        for pid,item in watchlist.items():
            p=df[df["product_id"].astype(str)==str(pid)]
            if not p.empty:
                current=float(p.sort_values("date")["price"].iloc[-1])
                watch_rows.append({"Product":item.get("product_name",pid),"Current price":current,"Target":item["target"],"Status":"Target reached" if current<=item["target"] else "Waiting"})
        if watch_rows:
            st.dataframe(pd.DataFrame(watch_rows).style.format({"Current price":"₹{:,.0f}","Target":"₹{:,.0f}"}),use_container_width=True,hide_index=True)
    else:
        st.caption("Your watchlist is empty. Save a target price to begin.")

# Data quality
with tabs[5]:
    st.subheader("Data quality & cleaning report")
    st.caption("The figures compare the CSV as loaded with the cleaned records used by the dashboard.")
    q1,q2,q3,q4,q5=st.columns(5)
    q1.metric("Raw rows",f"{quality['raw_rows']:,}")
    q2.metric("Clean rows",f"{quality['clean_rows']:,}")
    q3.metric("Rows removed",f"{quality['removed_rows']:,}")
    q4.metric("Duplicate keys",f"{quality['duplicate_keys']:,}")
    q5.metric("Missing cells",f"{quality['missing_cells']:,}")
    quality_df=pd.DataFrame({"Check":["Invalid dates","Invalid prices","Non-positive prices","Duplicate date/product keys"],
                             "Count":[quality["invalid_dates"],quality["invalid_prices"],quality["nonpositive_prices"],quality["duplicate_keys"]]})
    fig=px.bar(quality_df,x="Check",y="Count",template="plotly_dark",color="Count",color_continuous_scale=["#47e1d1","#a58aff"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=330,margin=dict(l=10,r=10,t=20,b=10),coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df.head(100),use_container_width=True,hide_index=True)
    st.download_button("Download cleaned data",data=df.to_csv(index=False).encode("utf-8"),file_name="price_history_cleaned.csv",mime="text/csv")
    st.download_button("Download quality report",data=quality_df.to_csv(index=False).encode("utf-8"),file_name="data_quality_report.csv",mime="text/csv")
