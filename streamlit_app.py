import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import glob
from typing import Optional, Dict

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="🏦 BankTrust BI",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Simple CSS to look like a polished LinkedIn demo
# -------------------------------------------------
st.markdown(
    """
<style>
/* remove extra padding and default chrome */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* KPI cards */
.kpi-card {
    padding: 0.8rem 1rem;
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(0, 0, 0, 0.06);
}
.kpi-title {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 4px;
}
.small-note {
    color: #6b7280;
    font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Data helpers
# -------------------------------------------------
SEARCH_DIRS = [
    ".", "Output", "Notebook", "Data", "Data/Output", "Data/Output/Notebook", "Visualization"
]


def discover_files() -> Dict[str, str]:
    """
    Look for CSV/Parquet files in common folders and return
    {label: path} sorted so your main files appear first.
    """
    found = {}
    for d in SEARCH_DIRS:
        for pat in ("*.csv", "*.parquet", "**/*.csv", "**/*.parquet"):
            for p in glob.glob(str(Path(d) / pat), recursive=True):
                path = Path(p)
                name = path.name
                # priority: your main tables first
                priority = 0
                if name in {
                    "cleaned_data_with_segmentation_and_clusters.csv",
                    "cleaned_data_with_segmentation.csv",
                    "rfm_with_segments_kmeans.csv",
                    "rfm_with_segments.csv",
                    "rfm_table.csv",
                }:
                    priority = -1
                label = f"{name}  —  {p}"
                found[label] = (priority, str(path))

    # sort by priority then label text
    sorted_items = sorted(
        ((k, v[1]) for k, v in found.items()),
        key=lambda item: (found[item[0]][0], item[0].lower()),
    )
    return dict(sorted_items)


@st.cache_data(show_spinner=False)
def load_table(path: str, sample_max: Optional[int]) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if sample_max is not None and len(df) > sample_max:
        df = df.sample(sample_max, random_state=42)
    return df


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names so the app works even if your
    CSV uses lowercase names.
    """
    mapping = {
        "segment": "Segment",
        "cluster": "Cluster",
        "customer_id": "CustomerID",
        "cust_id": "CustomerID",
        "transaction_date": "TransactionDate",
        "date": "Date",
        "amount": "Amount",
        "monetary": "Monetary",
        "frequency": "Frequency",
        "recency": "Recency",
        "gender": "Gender",
    }
    rename = {k: v for k, v in mapping.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=rename)
    return df


# -------------------------------------------------
# Header (hero + subtitle)
# -------------------------------------------------
left, right = st.columns([0.78, 0.22])
with left:
    st.markdown("### 🏦 BankTrust Analytics")
    st.markdown(
        "<div class='small-note'>Customer Intelligence • RFM • Clustering • Seasonality</div>",
        unsafe_allow_html=True,
    )
with right:
    st.write("")

# -------------------------------------------------
# Sidebar – data source
# -------------------------------------------------
st.sidebar.title("⚙️ Controls")

files = discover_files()
if files:
    dataset_label = st.sidebar.selectbox(
        "Dataset",
        list(files.keys()),
        index=0,
        key="file_select",
    )
else:
    dataset_label = None

uploaded = st.sidebar.file_uploader(
    "…or upload CSV/Parquet",
    type=["csv", "parquet"],
    key="uploader",
)

sample_choice = st.sidebar.selectbox(
    "Row limit (sampling)",
    ["No limit", "300k", "100k", "50k", "10k"],
    index=2,
    key="sample_sel",
)
sample_max = None if sample_choice == "No limit" else int(sample_choice.replace("k", "000"))

# Load data (uploaded takes priority)
if uploaded is not None:
    if uploaded.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded)
    else:
        df = pd.read_csv(uploaded)
elif dataset_label:
    df = load_table(files[dataset_label], sample_max)
else:
    st.info("➡️ Add a dataset to your repo (e.g. Output/ or Notebook/) or upload one from the sidebar.")
    st.stop()

df = normalize_cols(df)
if df.empty:
    st.error("Loaded dataset is empty.")
    st.stop()

# -------------------------------------------------
# Sidebar – filters (all with unique keys)
# -------------------------------------------------
st.sidebar.subheader("🔎 Filters")
work = df.copy()


def multiselect_filter(label: str, col: str) -> pd.DataFrame:
    global work
    values = sorted(work[col].dropna().astype(str).unique())
    selected = st.sidebar.multiselect(label, values, default=[], key=f"ms_{col}")
    if selected:
        work = work[work[col].astype(str).isin(selected)]
    return work


# Segment / Cluster / Gender
for col in ("Segment", "Cluster", "Gender"):
    if col in work.columns:
        work = multiselect_filter(col, col)

# Location-like column
loc_col = next(
    (c for c in ["Location", "Branch", "City", "State", "Region", "Country"] if c in work.columns),
    None,
)
if loc_col:
    work = multiselect_filter(loc_col, loc_col)


def add_range_filter(col: str, step: float = 1.0) -> pd.DataFrame:
    global work
    col_min, col_max = float(work[col].min()), float(work[col].max())
    a, b = st.sidebar.slider(
        f"{col} range",
        min_value=col_min,
        max_value=col_max,
        value=(col_min, col_max),
        step=step,
        key=f"rng_{col}",
    )
    work = work[(work[col] >= a) & (work[col] <= b)]
    return work


for col, step in (("Monetary", 1.0), ("Frequency", 1.0), ("Recency", 1.0)):
    if col in work.columns and np.isfinite(work[col].min()) and np.isfinite(work[col].max()):
        work = add_range_filter(col, step)

# Date range
date_col = next((c for c in ["TransactionDate", "Date"] if c in work.columns), None)
if date_col:
    dt = pd.to_datetime(work[date_col], errors="coerce")
    if dt.notna().any():
        dmin, dmax = dt.min().date(), dt.max().date()
        d_start, d_end = st.sidebar.date_input(
            "Date window",
            value=(dmin, dmax),
            key="date_window",
        )
        try:
            mask = (dt.dt.date >= d_start) & (dt.dt.date <= d_end)
            work = work[mask]
        except Exception:
            pass

# CustomerID contains
if "CustomerID" in work.columns:
    q = st.sidebar.text_input("CustomerID contains", key="cust_contains")
    if q.strip():
        work = work[work["CustomerID"].astype(str).str.contains(q.strip(), case=False, na=False)]

st.sidebar.caption(f"Filtered rows: {len(work):,}")

# -------------------------------------------------
# KPI donuts + cards
# -------------------------------------------------
def donut_kpi(title: str, value: float, fmt: str = ",.0f") -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=[title],
            values=[max(float(value), 1e-9)],
            hole=0.72,
            textinfo="none",
        )
    )
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                text=f"{value:{fmt}}",
                x=0.5,
                y=0.5,
                font_size=22,
                showarrow=False,
            )
        ],
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


c1, c2, c3, c4 = st.columns(4)

custs = work["CustomerID"].nunique() if "CustomerID" in work.columns else len(work)
rows = len(work)
amt_col = next(
    (c for c in ["Monetary", "Amount", "TransactionAmount", "TotalAmount"] if c in work.columns),
    None,
)
total_amt = float(work[amt_col].sum()) if amt_col else 0.0
avg_amt = float(work[amt_col].mean()) if amt_col else 0.0

with c1:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Unique Customers</div>", unsafe_allow_html=True)
    st.plotly_chart(donut_kpi("Unique Customers", custs), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Rows (Tx)</div>", unsafe_allow_html=True)
    st.plotly_chart(donut_kpi("Rows (Tx)", rows), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Total Monetary</div>", unsafe_allow_html=True)
    st.plotly_chart(donut_kpi("Total Monetary", total_amt, fmt=",.2f"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Avg Monetary</div>", unsafe_allow_html=True)
    st.plotly_chart(donut_kpi("Avg Monetary", avg_amt, fmt=",.2f"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# Tabs (LinkedIn-style navigation)
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Segments", "Customers", "Seasonality", "Geography", "Data / Export"]
)

# --- Overview ---
with tab1:
    st.subheader("📊 Distributions")
    dist_cols = [c for c in ["Monetary", "Frequency", "Recency"] if c in work.columns]
    cols = st.columns(len(dist_cols) if dist_cols else 1)
    for ax, col in zip(cols, dist_cols):
        with ax:
            fig = px.histogram(work, x=col, nbins=40, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

# --- Segments ---
with tab2:
    st.subheader("🧭 Segment Breakdown")
    if "Segment" in work.columns:
        seg_counts = work["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.bar(seg_counts, x="Segment", y="Count", title="Customers per Segment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'Segment' column found.")

# --- Customers ---
with tab3:
    st.subheader("👤 Customer 360")
    if "CustomerID" in df.columns:
        cid = st.text_input("Enter CustomerID (exact)", key="cust_exact")
        if cid:
            sub = df[df["CustomerID"].astype(str) == str(cid)]
            if sub.empty:
                st.warning("No records found for that CustomerID.")
            else:
                st.dataframe(sub.head(500), use_container_width=True)
    else:
        st.info("No CustomerID column.")

# --- Seasonality ---
with tab4:
    st.subheader("📆 Seasonality")
    dcol = next((c for c in ["TransactionDate", "Date"] if c in work.columns), None)
    if dcol and amt_col:
        t = work.copy()
        t[dcol] = pd.to_datetime(t[dcol], errors="coerce")
        t = t.dropna(subset=[dcol])
        if not t.empty:
            s = t.groupby(t[dcol].dt.date)[amt_col].sum().reset_index()
            s.columns = ["Date", "Total"]
            fig = px.line(s, x="Date", y="Total", title="Daily Total Amount")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need a Date/TransactionDate column and an amount column.")

# --- Geography ---
with tab5:
    st.subheader("🗺️ Geography (Top Regions)")
    reg = next(
        (c for c in ["City", "State", "Region", "Branch", "Location", "Country"] if c in work.columns),
        None,
    )
    if reg:
        if amt_col:
            top = (
                work.groupby(reg)[amt_col]
                .sum()
                .reset_index()
                .sort_values(amt_col, ascending=False)
                .head(30)
            )
            fig = px.bar(top, x=reg, y=amt_col, title=f"Top {reg} by {amt_col}")
        else:
            top = (
                work[reg]
                .value_counts()
                .reset_index()
                .rename(columns={"index": reg, reg: "Count"})
                .head(30)
            )
            fig = px.bar(top, x=reg, y="Count", title=f"Top {reg} by Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No region/city/branch columns detected.")

# --- Data / Export ---
with tab6:
    st.subheader("📋 Filtered Data Preview")
    st.dataframe(work.head(1000), use_container_width=True)
    st.download_button(
        "⬇️ Download filtered CSV",
        work.to_csv(index=False).encode("utf-8"),
        file_name="filtered_retail_data.csv",
        mime="text/csv",
    )
