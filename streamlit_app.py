
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Retail Banking Intelligence Dashboard",
    layout="wide",
    page_icon="🏦"
)

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data_with_segmentation_and_clusters.csv")

df = load_data()

def sidebar_filters(data):
    st.sidebar.header("🔍 Filters")
    out = data.copy()

    if "Segment" in out.columns:
        seg = st.sidebar.multiselect("Customer Segment",
            sorted(out["Segment"].dropna().unique()), key="segment_filter")
        if seg:
            out = out[out["Segment"].isin(seg)]

    if "Cluster" in out.columns:
        clu = st.sidebar.multiselect("Cluster",
            sorted(out["Cluster"].dropna().unique()), key="cluster_filter")
        if clu:
            out = out[out["Cluster"].isin(clu)]

    if "Monetary" in out.columns:
        mn, mx = float(out["Monetary"].min()), float(out["Monetary"].max())
        rng = st.sidebar.slider("Monetary range", min_value=mn, max_value=mx,
            value=(mn, mx), step=1.0, key="monetary_slider")
        out = out[(out["Monetary"] >= rng[0]) & (out["Monetary"] <= rng[1])]

    return out

filtered = sidebar_filters(df)

st.markdown("<h1 style='text-align: center;'>🏦 Retail Banking Intelligence Dashboard</h1>",
    unsafe_allow_html=True)

cust_count = filtered["CustomerID"].nunique()
tx_count = len(filtered)
total_monetary = filtered["Monetary"].sum()
avg_monetary = filtered["Monetary"].mean()

metric_df = pd.DataFrame({
    "Metric": ["Unique Customers", "Rows (Tx)", "Total Monetary", "Avg Monetary"],
    "Value": [cust_count, tx_count, total_monetary, avg_monetary]
})

fig = px.pie(metric_df, names="Metric", values="Value", hole=0.55,
             title="Key Metrics Breakdown")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Filtered Data Preview")
st.dataframe(filtered.head(20), use_container_width=True)

st.download_button(
    "⬇️ Download Filtered CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_retail_data.csv",
    mime="text/csv"
)
