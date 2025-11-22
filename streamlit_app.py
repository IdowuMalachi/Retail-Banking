import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import glob

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="🏦 BankTrust BI",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# CSS: colorful background + cards
# -------------------------------------------------
st.markdown(
    """
<style>
/* App background with soft gradient */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #e0edff 0, #f5f7fb 40%, #f9fafb 100%) !important;
}

/* main container */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* header area */
h3 {
    color: #0f172a;
}

/* KPI cards */
.kpi-card {
    padding: 0.9rem 1.1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #ffffff 0%, #eef2ff 45%, #e0f2fe 100%);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(148, 163, 184, 0.4);
}
.kpi-title {
    font-size: 0.9rem;
    color: #475569;
    margin-bottom: 4px;
}
.small-note {
    color: #64748b;
    font-size: 0.86rem;
}

/* Tabs underline color */
[data-baseweb="tab-highlight"] {
    background-color: #2563eb !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Data helpers
# -------------------------------------------------
SEARCH_DIRS = [
    ".",
    "Output",
    "Notebook",
    "Data",
    "Data/Output",
    "Data/Output/Notebook",
    "Visualization",
]


def discover_files():
    """Find CSV / Parquet files in common project folders."""
    found = {}
    for d in SEARCH_DIRS:
        for pat in ("*.csv", "*.parquet", "**/*.csv", "**/*.parquet"):
            for p in glob.glob(str(Path(d) / pat), recursive=True):
                path = Path(p)
                name = path.name
                priority = 0
                if name in {
                    "cleaned_data_with_segmentation_and_clusters.csv",
                    "cleaned_data_with_segmentation.csv",
                    "rfm_with_segments_kmeans.csv",
                    "rfm_with_segments.csv",
                    "rfm_table.csv",
                }:
                    priority = -1  # bubble these to the top
                label = f"{name}  —  {p}"
                found[label] = (priority, str(path))

    sorted_items = sorted(
        ((k, v[1]) for k, v in found.items()),
        key=lambda item: (found[item[0]][0], item[0].lower()),
    )
    return dict(sorted_items)


@st.cache_data(show_spinner=False)
def load_table(path, sample_max):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if sample_max is not None and len(df) > sample_max:
        df = df.sample(sample_max, random_state=42)
    return df


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise common column names to TitleCase."""
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
    return df.rename(columns=rename)


# -------------------------------------------------
# Short number formatter (max ~4 characters before suffix)
# -------------------------------------------------
def format_short(value: float) -> str:
    """Format large numbers like 98.3k, 2.1M, 216M."""
    if value is None or not np.isfinite(value):
        return "-"
    v = float(value)
    sign = "-" if v < 0 else ""
    v = abs(v)

    if v >= 1_000_000_000:
        num, suffix = v / 1_000_000_000, "B"
    elif v >= 1_000_000:
        num, suffix = v / 1_000_000, "M"
    elif v >= 1_000:
        num, suffix = v / 1_000, "k"
    else:
        return f"{sign}{v:.0f}"

    s = f"{num:.2f}".rstrip("0").rstrip(".")
    if len(s) > 4:
        s = s[:4]
    return f"{sign}{s}{suffix}"


# -------------------------------------------------
# Header (hero)
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
    "…or upload CSV/Parquet", type=["csv", "parquet"], key="uploader"
)

sample_choice = st.sidebar.selectbox(
    "Row limit (sampling)",
    ["No limit", "300k", "100k", "50k", "10k"],
    index=2,
    key="sample_sel",
)
sample_max = None if sample_choice == "No limit" else int(sample_choice.replace("k", "000"))

# Load data
if uploaded is not None:
    df = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else pd.read_csv(uploaded)
elif dataset_label:
    df = load_table(files[dataset_label], sample_max)
else:
    st.info("➡️ Add a dataset to your repo (Output/ or Notebook/) or upload one from the sidebar.")
    st.stop()

df = normalize_cols(df)
if df.empty:
    st.error("Loaded dataset is empty.")
    st.stop()

# -------------------------------------------------
# Sidebar – filters
# -------------------------------------------------
st.sidebar.subheader("🔎 Filters")
work = df.copy()


def multiselect_filter(label: str, col: str) -> None:
    global work
    values = sorted(work[col].dropna().astype(str).unique())
    selected = st.sidebar.multiselect(label, values, default=[], key=f"ms_{col}")
    if selected:
        work = work[work[col].astype(str).isin(selected)]


for col in ("Segment", "Cluster", "Gender"):
    if col in work.columns:
        multiselect_filter(col, col)

loc_col = next(
    (c for c in ["Location", "Branch", "City", "State", "Region", "Country"] if c in work.columns),
    None,
)
if loc_col:
    multiselect_filter(loc_col, loc_col)


def add_range_filter(col: str, step: float = 1.0) -> None:
    global work
    mn, mx = float(work[col].min()), float(work[col].max())
    a, b = st.sidebar.slider(
        f"{col} range",
        min_value=mn,
        max_value=mx,
        value=(mn, mx),
        step=step,
        key=f"rng_{col}",
    )
    work = work[(work[col] >= a) & (work[col] <= b)]


for col_name, step_val in (("Monetary", 1.0), ("Frequency", 1.0), ("Recency", 1.0)):
    if col_name in work.columns and np.isfinite(work[col_name].min()) and np.isfinite(work[col_name].max()):
        add_range_filter(col_name, step_val)

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

# CustomerID search
if "CustomerID" in work.columns:
    q = st.sidebar.text_input("CustomerID contains", key="cust_contains")
    if q.strip():
        work = work[work["CustomerID"].astype(str).str.contains(q.strip(), case=False, na=False)]

st.sidebar.caption(f"Filtered rows: {len(work):,}")

# -------------------------------------------------
# KPI donuts + colorful cards
# -------------------------------------------------
def donut_kpi(title: str, value: float) -> go.Figure:
    """Donut with short center text and blue ring."""
    short = format_short(value)
    fig = go.Figure(
        go.Pie(
            labels=[title],
            values=[max(float(value), 1e-9)],
            hole=0.7,
            textinfo="none",
            marker=dict(colors=["#2563eb", "#93c5fd"]),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(
                text=short,
                x=0.5,
                y=0.5,
                font_size=22,
                font_color="#0f172a",
                showarrow=False,
            )
        ],
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


c1, c2, c3, c4 = st.columns(4)

custs = work["CustomerID"].nunique() if "CustomerID" in work.columns else len(work)
rows_count = len(work)
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
    st.plotly_chart(donut_kpi("Rows (Tx)", rows_count), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Total Monetary</div>", unsafe_allow_html=True)
    st.plotly_chart(donut_kpi("Total Monetary", total_amt), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='kpi-card'><div class='kpi-title'>Avg Monetary</div>", unsafe_allow_html=True)
    st.plotly_chart(donut_kpi("Avg Monetary", avg_amt), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# Tabs – Overview / Segments / Customers / Seasonality / Geography / Export
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Segments", "Customers", "Seasonality", "Geography", "Data / Export"]
)

# --- Overview (advanced histograms + 3D RFM if available) ---
with tab1:
    st.subheader("📊 Advanced Distributions & RFM Space")

    dist_cols = [c for c in ["Monetary", "Frequency", "Recency"] if c in work.columns]
    cols = st.columns(len(dist_cols) if dist_cols else 1)
    for ax, col in zip(cols, dist_cols):
        with ax:
            fig = px.histogram(
                work,
                x=col,
                nbins=40,
                title=f"{col} Distribution",
                color_discrete_sequence=["#0ea5e9"],
            )
            fig.update_traces(opacity=0.75)
            fig.update_layout(
                xaxis_tickformat=".2s",
                yaxis_tickformat=".2s",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(248,250,252,1)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3D RFM scatter
    if all(c in work.columns for c in ["Recency", "Frequency", "Monetary"]):
        st.markdown("### 🎯 RFM 3D Space")
        sample = work.copy()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)

        fig3d = px.scatter_3d(
            sample,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color=sample["Segment"] if "Segment" in sample.columns else None,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            opacity=0.85,
        )
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Recency",
                yaxis_title="Frequency",
                zaxis_title="Monetary",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3d, use_container_width=True)

# --- Segments ---
with tab2:
    st.subheader("🧭 Segment Breakdown")

    if "Segment" in work.columns:
        seg_counts = work["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        # colorful donut for mix
        fig_seg_pie = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Segment Mix",
        )
        fig_seg_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_seg_pie, use_container_width=True)

        # stacked bar by segment vs maybe Gender or Cluster
        breakdown_dim = None
        if "Gender" in work.columns:
            breakdown_dim = "Gender"
        elif "Cluster" in work.columns:
            breakdown_dim = "Cluster"

        if breakdown_dim:
            st.markdown(f"#### Segment vs {breakdown_dim}")
            tmp = (
                work.groupby(["Segment", breakdown_dim])
                .size()
                .reset_index(name="Count")
            )
            fig_stack = px.bar(
                tmp,
                x="Segment",
                y="Count",
                color=breakdown_dim,
                barmode="stack",
                color_discrete_sequence=px.colors.sequential.Blues,
            )
            fig_stack.update_layout(
                yaxis_tickformat=".2s",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(248,250,252,1)",
            )
            st.plotly_chart(fig_stack, use_container_width=True)
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
    st.subheader("📆 Seasonality (Gradient Area)")

    dcol = next((c for c in ["TransactionDate", "Date"] if c in work.columns), None)
    if dcol and amt_col:
        t = work.copy()
        t[dcol] = pd.to_datetime(t[dcol], errors="coerce")
        t = t.dropna(subset=[dcol])
        if not t.empty:
            s = t.groupby(t[dcol].dt.date)[amt_col].sum().reset_index()
            s.columns = ["Date", "Total"]

            fig_area = go.Figure()
            fig_area.add_trace(
                go.Scatter(
                    x=s["Date"],
                    y=s["Total"],
                    mode="lines",
                    line=dict(color="#1d4ed8", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.25)",
                )
            )
            fig_area.update_layout(
                title="Daily Total Amount",
                yaxis_tickformat=".2s",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(248,250,252,1)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_area, use_container_width=True)
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
                .head(25)
            )
            fig_geo = px.bar(
                top,
                x=amt_col,
                y=reg,
                orientation="h",
                title=f"Top {reg} by {amt_col}",
                color=amt_col,
                color_continuous_scale=px.colors.sequential.Blues,
            )
        else:
            top = (
                work[reg]
                .value_counts()
                .reset_index()
                .rename(columns={"index": reg, reg: "Count"})
                .head(25)
            )
            fig_geo = px.bar(
                top,
                x="Count",
                y=reg,
                orientation="h",
                title=f"Top {reg} by Count",
                color="Count",
                color_continuous_scale=px.colors.sequential.Blues,
            )

        fig_geo.update_layout(
            xaxis_tickformat=".2s",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,1)",
           margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_geo, use_container_width=True)
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
