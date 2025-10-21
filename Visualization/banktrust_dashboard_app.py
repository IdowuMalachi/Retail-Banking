
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="BankTrust Analytics Dashboard", layout="wide")

def resolve_paths(names):
    roots = [Path("."), Path("artifacts/figures"), Path("artifacts")]
    out = []
    for n in names:
        for r in roots:
            p = r / n
            if p.exists():
                out.append(p)
                break
    return out

EDA_FIGS = ['TransactionAmount_(capped_1%-99%).png', 'CustAccountBalance_(capped_1%-99%).png', 'Top_10_Locations_by_Transaction_Count.png', 'Transactions_by_Gender.png', 'plot_histograms_features.png']
RFM_FIGS = ['Customer_Segments_by_RFM_Score.png', '31_rfm_segments_yellow_bar.png', 'plot_cluster_profiles_means.png']
KMEANS_FIGS = ['plot_elbow_inertia.png', 'plot_silhouette_by_k.png', 'plot_pca_clusters.png', 'plot_radar_cluster_profiles.png', 'plot_stacked_cluster_segment.png', 'plot_heatmap_segment_cluster.png', 'plot_boxplots_features_by_cluster.png']

page = st.sidebar.radio("Navigate", ["📊 Exploratory Data Analysis", "🧭 RFM Segmentation", "🧩 K-Means Clustering"])

def show_gallery(files, cols=2):
    paths = resolve_paths(files)
    if not paths:
        st.info("No figures found yet. Run the notebook to generate them.")
        return
    rows = [paths[i:i+cols] for i in range(0, len(paths), cols)]
    for row in rows:
        cols_sp = st.columns(len(row))
        for c, p in zip(cols_sp, row):
            with c:
                st.image(str(p), caption=p.name, use_column_width=True)

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

# Zip download of all known figures
import io, zipfile
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in EDA_FIGS + RFM_FIGS + KMEANS_FIGS:
        for root in [Path("."), Path("artifacts/figures"), Path("artifacts")]:
            p = root / name
            if p.exists():
                zf.write(p, arcname=p.name)
                break
buf.seek(0)
st.sidebar.download_button("Download all charts (.zip)", data=buf, file_name="banktrust_charts.zip", mime="application/zip")
