
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="🏦 Retail Banking Intelligence", layout="wide", page_icon="🏦")

# -------------------- Data utils --------------------
SEARCH_DIRS = [
    ".", "Output", "Notebook", "Data", "Data/Output", "Data/Output/Notebook", "Visualization"
]

def discover_files():
    """Return dict {label: path} of candidate CSV/Parquet files in common folders."""
    found = {}
    for d in SEARCH_DIRS:
        for pat in ("*.csv","*.parquet","**/*.csv","**/*.parquet"):
            for p in glob.glob(str(Path(d) / pat), recursive=True):
                name = Path(p).name
                pri = 0
                if name in {
                    "cleaned_data_with_segmentation_and_clusters.csv",
                    "cleaned_data_with_segmentation.csv",
                    "rfm_with_segments_kmeans.csv",
                    "rfm_with_segments.csv",
                    "rfm_table.csv"
                }:
                    pri = -1
                found[f"{name}  —  {p}"] = (pri, p)
    return dict(sorted(((k,v[1]) for k,v in found.items()), key=lambda kv: (found[kv[0]][0], kv[0].lower())))

@st.cache_data(show_spinner=False)
def load_table(path: str, sample_max: int | None):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if sample_max and len(df) > sample_max:
        df = df.sample(sample_max, random_state=42)
    return df

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "segment":"Segment", "cluster":"Cluster",
        "customer_id":"CustomerID", "cust_id":"CustomerID",
        "transaction_date":"TransactionDate", "date":"Date",
        "amount":"Amount", "monetary":"Monetary",
        "frequency":"Frequency", "recency":"Recency",
        "gender":"Gender",
    }
    ren = {k:v for k,v in mapping.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=ren)

# -------------------- Header --------------------
st.title("🏦 Retail Banking Intelligence")

# -------------------- Source picker --------------------
st.sidebar.header("📁 Data source")
files = discover_files()
choice = None
if files:
    choice = st.sidebar.selectbox("Select a dataset", list(files.keys()), index=0, key="file_select")

uploaded = st.sidebar.file_uploader("...or upload a CSV/Parquet", type=["csv","parquet"], key="uploader")

sample_size = st.sidebar.selectbox("Row limit (sampling)", ["No limit","100k","50k","10k"], index=1, key="sample_sel")
sample_max = None if sample_size=="No limit" else int(sample_size.replace("k","000"))

if uploaded is not None:
    if uploaded.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded)
    else:
        df = pd.read_csv(uploaded)
elif choice is not None:
    df = load_table(files[choice], sample_max)
else:
    st.warning("No dataset found. Put a CSV/Parquet in your repo (e.g., Output/ or Notebook/) or upload one via the sidebar.")
    st.stop()

df = normalize_cols(df)
if df.empty:
    st.error("Loaded dataset is empty.")
    st.stop()

# -------------------- Filters (unique keys) --------------------
st.sidebar.header("🔎 Filters")
work = df.copy()

def multiselect_filter(label, col):
    vals = sorted(work[col].dropna().astype(str).unique())
    sel = st.sidebar.multiselect(label, vals, key=f"ms_{col}")
    if sel:
        return work[work[col].astype(str).isin(sel)]
    return work

for col in ("Segment","Cluster","Gender"):
    if col in work.columns:
        work = multiselect_filter(col, col)

loc_col = next((c for c in ["Location","Branch","City","State","Region","Country"] if c in work.columns), None)
if loc_col:
    work = multiselect_filter(loc_col, loc_col)

def range_slider(col, step=1.0):
    mn, mx = float(work[col].min()), float(work[col].max())
    a,b = st.sidebar.slider(f"{col} range", min_value=mn, max_value=mx, value=(mn,mx), step=step, key=f"rng_{col}")
    return work[(work[col] >= a) & (work[col] <= b)]

for col, step in (("Monetary", 1.0), ("Frequency", 1.0), ("Recency", 1.0)):
    if col in work.columns and np.isfinite(work[col].min()) and np.isfinite(work[col].max()):
        work = range_slider(col, step=step)

date_col = next((c for c in ["TransactionDate","Date"] if c in work.columns), None)
if date_col:
    dt = pd.to_datetime(work[date_col], errors="coerce")
    if dt.notna().any():
        dmin, dmax = dt.min().date(), dt.max().date()
        a,b = st.sidebar.date_input("Date window", value=(dmin,dmax), key="date_window")
        try:
            mask = (dt.dt.date >= a) & (dt.dt.date <= b)
            work = work[mask]
        except Exception:
            pass

if "CustomerID" in work.columns:
    q = st.sidebar.text_input("CustomerID contains", key="cust_contains")
    if q.strip():
        work = work[work["CustomerID"].astype(str).str.contains(q.strip(), case=False, na=False)]

st.sidebar.caption(f"Filtered rows: {len(work):,}")

# -------------------- KPI Donuts --------------------
def donut_kpi(title, value, fmt=",.0f"):
    fig = go.Figure(go.Pie(labels=[title], values=[max(float(value),1e-9)], hole=0.72, textinfo="none"))
    fig.update_layout(showlegend=False,
                      annotations=[dict(text=f"{value:{fmt}}", x=0.5, y=0.5, font_size=22, showarrow=False)],
                      height=220, margin=dict(l=0,r=0,t=0,b=0))
    return fig

custs = work["CustomerID"].nunique() if "CustomerID" in work.columns else len(work)
rows = len(work)
amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in work.columns), None)
total_amt = float(work[amt_col].sum()) if amt_col else 0.0
avg_amt = float(work[amt_col].mean()) if amt_col else 0.0

c1,c2,c3,c4 = st.columns(4)
with c1: st.subheader("Unique Customers"); st.plotly_chart(donut_kpi("Unique Customers", custs), use_container_width=True)
with c2: st.subheader("Rows (Tx)"); st.plotly_chart(donut_kpi("Rows (Tx)", rows), use_container_width=True)
with c3: st.subheader("Total Monetary"); st.plotly_chart(donut_kpi("Total Monetary", total_amt, fmt=",.2f"), use_container_width=True)
with c4: st.subheader("Avg Monetary"); st.plotly_chart(donut_kpi("Avg Monetary", avg_amt, fmt=",.2f"), use_container_width=True)

st.markdown("---")

# -------------------- Distributions --------------------
st.subheader("📊 Distributions")
dist_cols = [c for c in ["Monetary","Frequency","Recency"] if c in work.columns]
cols = st.columns(len(dist_cols) if dist_cols else 1)
for ax, col in zip(cols, dist_cols):
    with ax:
        st.plotly_chart(px.histogram(work, x=col, nbins=40, title=f"{col} Distribution"), use_container_width=True)

# -------------------- Segment bar --------------------
if "Segment" in work.columns:
    st.subheader("📌 Segment Breakdown")
    seg_counts = work["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment","Count"]
    st.plotly_chart(px.bar(seg_counts, x="Segment", y="Count"), use_container_width=True)

# -------------------- Data + Download --------------------
st.subheader("📋 Filtered Data Preview")
st.dataframe(work.head(1000), use_container_width=True)
st.download_button("⬇️ Download Filtered CSV",
                   work.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_retail_data.csv",
                   mime="text/csv")
