import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(
    page_title="🏦 Retail Banking Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA_CANDIDATES = [
    ROOT / "Output",
    ROOT / "Data" / "Output" / "Notebook",
    ROOT / "Notebook",
    ROOT,
]
CSV_PRIORITY = [
    "cleaned_data_with_segmentation_and_clusters.csv",
    "cleaned_data_with_segmentation.csv",
    "rfm_with_segments_kmeans.csv",
    "rfm_with_segments.csv",
    "rfm_table.csv",
]

@st.cache_data(show_spinner=False)
def find_first_csv():
    found = {}
    for folder in DATA_CANDIDATES:
        if not folder.exists():
            continue
        for name in CSV_PRIORITY:
            p = folder / name
            if p.exists():
                found[name] = p
    return found

@st.cache_data(show_spinner=True)
def load_csv(path: Path) -> pd.DataFrame:
    try:
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def kpi_row(df: pd.DataFrame, amount_cols=("Monetary","Amount","TransactionAmount")):
    c1, c2, c3, c4 = st.columns(4)
    n_customers = df["CustomerID"].nunique() if "CustomerID" in df.columns else len(df)
    n_tx = df.shape[0]
    amt_col = next((c for c in amount_cols if c in df.columns), None)
    total_amt = float(df[amt_col].sum()) if amt_col else 0.0
    avg_amt = float(df[amt_col].mean()) if amt_col else 0.0
    with c1: metric_card("Unique Customers", f"{n_customers:,}")
    with c2: metric_card("Rows (Tx)", f"{n_tx:,}")
    with c3: metric_card("Total Monetary", f"{total_amt:,.2f}")
    with c4: metric_card("Avg Monetary", f"{avg_amt:,.2f}")

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("🔎 Filters")
    pick_gender = "Gender" if "Gender" in df.columns else None
    pick_location = None
    for candidate in ["Location", "Branch", "City", "State"]:
        if candidate in df.columns:
            pick_location = candidate
            break
    if pick_gender:
        genders = ["All"] + sorted([g for g in df[pick_gender].dropna().unique().tolist()])
        g_sel = st.sidebar.selectbox("Gender", genders, index=0)
    else:
        g_sel = "All"
    if pick_location:
        locs = ["All"] + sorted([x for x in df[pick_location].dropna().unique().tolist()])
        l_sel = st.sidebar.selectbox(pick_location, locs, index=0)
    else:
        l_sel = "All"
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount"] if c in df.columns), None)
    if amt_col:
        min_v, max_v = float(df[amt_col].min()), float(df[amt_col].max())
        v1, v2 = st.sidebar.slider(f"{amt_col} range", min_value=float(min_v), max_value=float(max_v),
                                   value=(float(min_v), float(max_v)))
    else:
        v1, v2 = None, None
    out = df.copy()
    if pick_gender and g_sel != "All":
        out = out[out[pick_gender] == g_sel]
    if pick_location and l_sel != "All":
        out = out[out[pick_location] == l_sel]
    if amt_col and v1 is not None:
        out = out[(out[amt_col] >= v1) & (out[amt_col] <= v2)]
    if "CustomerID" in out.columns:
        st.sidebar.text_input("Search CustomerID", key="cust_search")
        q = st.session_state.get("cust_search", "").strip()
        if q:
            out = out[out["CustomerID"].astype(str).str.contains(q, case=False, na=False)]
    return out

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

found = find_first_csv()
st.sidebar.success(f"Detected {len(found)} candidate tables")
if not found:
    st.error("No CSVs found. Please add one of the expected files under Output/ or Notebook/.")
    st.stop()

chosen_name = st.sidebar.selectbox("Primary dataset", list(found.keys()), index=0)
main_df = load_csv(found[chosen_name])

if main_df.empty:
    st.error("The selected dataset is empty or failed to load.")
    st.stop()

rename_map = {}
if "segment" in main_df.columns and "Segment" not in main_df.columns:
    rename_map["segment"] = "Segment"
if "cluster" in main_df.columns and "Cluster" not in main_df.columns:
    rename_map["cluster"] = "Cluster"
main_df = main_df.rename(columns=rename_map)

st.sidebar.title("🏦 Retail Banking Intelligence")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📂 Data Explorer",
        "🧭 RFM Analytics",
        "🧩 Clustering",
        "📈 EDA Visuals",
        "⬇️ Export",
    ],
)

filtered = sidebar_filters(main_df)

if page == "🏠 Overview":
    st.title("🏦 Retail Banking Intelligence Dashboard")
    st.caption(f"Primary table: **{chosen_name}** — Rows: {len(main_df):,}")
    with st.container():
        kpi_row(filtered)
    st.markdown("---")
    section_header("Distributions")
    possible = ["Monetary","Amount","TransactionAmount","Frequency","Recency"]
    numeric_available = [c for c in possible if c in filtered.columns]
    if numeric_available:
        ncols = min(3, len(numeric_available))
        cols = st.columns(ncols)
        for i, col in enumerate(numeric_available[:ncols]):
            with cols[i]:
                fig = px.histogram(filtered, x=col, nbins=40, title=f"{col} Distribution")
                st.plotly_chart(fig, use_container_width=True)
    if "Segment" in filtered.columns:
        section_header("Segment Mix")
        seg_counts = filtered["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        fig = px.bar(seg_counts, x="Segment", y="Count", title="Customers by Segment")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📂 Data Explorer":
    st.title("📂 Data Explorer")
    st.write("Preview the filtered dataset and download selections.")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.caption(f"Showing up to 1,000 rows. Filtered total: {len(filtered):,}")
    with st.expander("Quick Group-By (pivot)"):
        numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
        cat_cols = [c for c in filtered.columns if c not in numeric_cols]
        by = st.multiselect("Group by (categorical)", cat_cols[:10], max_selections=3)
        agg_col = st.selectbox("Aggregate column", numeric_cols) if numeric_cols else None
        agg_fn = st.selectbox("Aggregation", ["mean","sum","count","median"]) if agg_col else None
        if by and agg_col and agg_fn:
            if agg_fn == "count":
                pivot = filtered.groupby(by)[agg_col].count().reset_index(name="count")
            else:
                pivot = getattr(filtered.groupby(by)[agg_col], agg_fn)().reset_index(name=f"{agg_fn}_{agg_col}")
            st.dataframe(pivot, use_container_width=True)
            fig = px.bar(pivot, x=by[0], y=pivot.columns[-1], color=by[1] if len(by) > 1 else None,
                         title="Group-By Chart")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🧭 RFM Analytics":
    st.title("🧭 RFM Analytics")
    df = filtered.copy()
    has_rfm = all(c in df.columns for c in ["Recency","Frequency","Monetary"])
    if not has_rfm:
        st.info("RFM base columns not found; showing numeric overview instead.")
        st.dataframe(df.describe(include="all").T)
        st.stop()
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(px.histogram(df, x="Recency", nbins=40, title="Recency"), use_container_width=True)
    with c2: st.plotly_chart(px.histogram(df, x="Frequency", nbins=40, title="Frequency"), use_container_width=True)
    with c3: st.plotly_chart(px.histogram(df, x="Monetary", nbins=40, title="Monetary"), use_container_width=True)
    if not {"R_Score","F_Score","M_Score"}.issubset(df.columns):
        df["R_Score"] = pd.qcut(df["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
        df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
        df["M_Score"] = pd.qcut(df["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)
    df["RFM_Sum"] = df[["R_Score","F_Score","M_Score"]].sum(axis=1)
    st.plotly_chart(px.histogram(df, x="RFM_Sum", nbins=15, title="RFM Score Sum"), use_container_width=True)
    if "Segment" in df.columns:
        section_header("Segment Profiles", "Average R/F/M by Segment")
        prof = df.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().reset_index()
        prof_melt = prof.melt(id_vars="Segment", var_name="Metric", value_name="Mean")
        fig = px.bar(prof_melt, x="Segment", y="Mean", color="Metric", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    section_header("Pairwise Relationships")
    num_pair = [c for c in ["Recency","Frequency","Monetary"] if c in df.columns]
    if len(num_pair) >= 2:
        fig = px.scatter(df, x=num_pair[0], y=num_pair[1], color="Segment" if "Segment" in df.columns else None,
                         title=f"{num_pair[0]} vs {num_pair[1]}")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🧩 Clustering":
    st.title("🧩 K-Means / Cluster Views")
    df = filtered.copy()
    features = [c for c in ["Recency","Frequency","Monetary"] if c in df.columns]
    if not features:
        st.info("No R/F/M columns detected. Showing numeric columns summary.")
        st.dataframe(df.describe().T)
        st.stop()
    if "Cluster" not in df.columns:
        st.warning("No 'Cluster' column detected in the dataset. Visualizing features only.")
        if len(features) >= 3:
            fig = px.scatter_3d(df, x=features[0], y=features[1], z=features[2], title="Feature Space")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(df, x=features[0], y=features[1], title="Feature Space")
            st.plotly_chart(fig, use_container_width=True)
    else:
        n_clusters = int(df["Cluster"].nunique())
        st.caption(f"Detected {n_clusters} clusters")
        agg = df.groupby("Cluster")[features].mean().reset_index()
        melted = agg.melt(id_vars="Cluster", var_name="Feature", value_name="Mean")
        fig = px.bar(melted, x="Feature", y="Mean", color="Cluster", barmode="group",
                     title="Mean Feature Values by Cluster")
        st.plotly_chart(fig, use_container_width=True)
        if len(features) >= 2:
            fig = px.scatter(df, x=features[0], y=features[1], color=df["Cluster"].astype(str),
                             title=f"{features[0]} vs {features[1]} by Cluster",
                             hover_data=[c for c in df.columns if c not in features[:2]][:5])
            st.plotly_chart(fig, use_container_width=True)

elif page == "📈 EDA Visuals":
    st.title("📈 EDA Visuals")
    candidates = [
        "Visualization/TransactionAmount_(capped_1%-99%).png",
        "Visualization/CustAccountBalance_(capped_1%-99%).png",
        "Visualization/Top_10_Locations_by_Transaction_Count.png",
        "Visualization/Transactions_by_Gender.png",
        "Visualization/plot_histograms_features.png",
        "Visualization/Customer_Segments_by_RFM_Score.png",
        "Visualization/plot_elbow_inertia.png",
        "Visualization/plot_silhouette_by_k.png",
        "Visualization/plot_pca_clusters.png",
        "Visualization/plot_radar_cluster_profiles.png",
        "Visualization/plot_stacked_cluster_segment.png",
        "Visualization/plot_heatmap_segment_cluster.png",
        "Visualization/plot_boxplots_features_by_cluster.png",
    ]
    grid_cols = st.slider("Images per row", 1, 4, 2)
    paths = []
    for rel in candidates:
        p = ROOT / rel
        if p.exists():
            paths.append(p)
    if not paths:
        st.info("No images found under Visualization/. Add PNGs to show static EDA charts.")
    else:
        rows = [paths[i:i+grid_cols] for i in range(0, len(paths), grid_cols)]
        for row in rows:
            cols = st.columns(len(row))
            for c, p in zip(cols, row):
                with c:
                    st.image(str(p), caption=p.name, use_container_width=True)

elif page == "⬇️ Export":
    st.title("⬇️ Export")
    st.write("Download the **filtered** dataset as CSV for offline analysis.")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        csv,
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )
    st.write("Copy a few rows as CSV:")
    st.code(filtered.head(10).to_csv(index=False))

st.markdown("---")
st.caption("Built for Idowu Malachi • Streamlit app using your repo's CSVs and visualizations.")
