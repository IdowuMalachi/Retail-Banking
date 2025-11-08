import streamlit as st
from pathlib import Path
import io, zipfile

st.set_page_config(page_title="BankTrust Analytics Dashboard", layout="wide")

# ---- Paths & fallbacks ----
ROOT = Path(__file__).parent
# Your repo uses these folders:
# - Visualization/ (PNGs, many in Git LFS)
# - Output/ (CSVs, etc., many in Git LFS)
# - Notebook/ (ipynb and some CSV artifacts)
SEARCH_ROOTS = [
    ROOT,  # "."
    ROOT / "Visualization",
    ROOT / "Output",
    ROOT / "Data" / "Output" / "Notebook",  # if you later reorganize
    ROOT / "Notebook",
]

# Optional: try to show images directly from GitHub if local files aren't present
# (Works for most cases, including LFS, because GitHub serves a media URL.)
GITHUB_RAW_BASES = [
    "https://raw.githubusercontent.com/IdowuMalachi/Retail-Banking/main/Visualization",
    "https://raw.githubusercontent.com/IdowuMalachi/Retail-Banking/main/Output",
]

def resolve_first_existing(name: str):
    # Try local files first
    for r in SEARCH_ROOTS:
        p = r / name
        if p.exists():
            return str(p)
    # Fallback to raw GitHub URL
    for base in GITHUB_RAW_BASES:
        url = f"{base}/{name}"
        # Streamlit will fetch lazily; no HEAD request needed
        return url
    return None

def resolve_many(names):
    out = []
    for n in names:
        path_or_url = resolve_first_existing(n)
        if path_or_url:
            out.append(path_or_url)
    return out

EDA_FIGS = [
    'TransactionAmount_(capped_1%-99%).png',
    'CustAccountBalance_(capped_1%-99%).png',
    'Top_10_Locations_by_Transaction_Count.png',
    'Transactions_by_Gender.png',
    'plot_histograms_features.png'
]
RFM_FIGS = [
    'Customer_Segments_by_RFM_Score.png',
    '31_rfm_segments_yellow_bar.png',
    'plot_cluster_profiles_means.png'
]
KMEANS_FIGS = [
    'plot_elbow_inertia.png',
    'plot_silhouette_by_k.png',
    'plot_pca_clusters.png',
    'plot_radar_cluster_profiles.png',
    'plot_stacked_cluster_segment.png',
    'plot_heatmap_segment_cluster.png',
    'plot_boxplots_features_by_cluster.png'
]

page = st.sidebar.radio("Navigate", ["📊 Exploratory Data Analysis", "🧭 RFM Segmentation", "🧩 K-Means Clustering"])

def show_gallery(files, cols=2):
    paths = resolve_many(files)
    if not paths:
        st.info("No figures found yet. Push images to Visualization/ (or Output/) or re-run the notebook to generate them.")
        return
    rows = [paths[i:i+cols] for i in range(0, len(paths), cols)]
    for row in rows:
        cols_sp = st.columns(len(row))
        for c, p in zip(cols_sp, row):
            with c:
                st.image(p, caption=p.split('/')[-1], use_column_width=True)

st.title("BankTrust Analytics Dashboard")

if page == "📊 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.markdown(
        "- Transaction distributions and basic splits help detect skews, seasonal effects, and potential data issues.\n"
        "- Use these to calibrate thresholds for later RFM bucketing."
    )
    show_gallery(EDA_FIGS, cols=2)

elif page == "🧭 RFM Segmentation":
    st.header("RFM Segmentation")
    st.markdown(
        "- Customer segments are derived from Recency, Frequency, and Monetary behavior.\n"
        "- Business-led rules emphasize *Recency* to support retention objectives."
    )
    show_gallery(RFM_FIGS, cols=2)

elif page == "🧩 K-Means Clustering":
    st.header("K-Means Clustering")
    st.markdown(
        "- Model selection via Elbow and sampled Silhouette.\n"
        "- PCA/radar/boxplots describe cluster shapes; heatmap compares clusters with rule-based segments."
    )
    show_gallery(KMEANS_FIGS, cols=2)

# ---- Optional: Zip download of local figures only (skip URLs) ----
def zip_local_files():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in EDA_FIGS + RFM_FIGS + KMEANS_FIGS:
            for r in SEARCH_ROOTS:
                p = r / name
                if p.exists():
                    zf.write(p, arcname=p.name)
                    break
    buf.seek(0)
    return buf

st.sidebar.markdown("---")
st.sidebar.write("Download your local charts (if present):")
st.sidebar.download_button(
    "Download all charts (.zip)",
    data=zip_local_files(),
    file_name="banktrust_charts.zip",
    mime="application/zip"
)
