import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

st.set_page_config(
    page_title="🏦 Retail Banking BI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA_CANDIDATES = [ROOT / "Output", ROOT / "Data" / "Output" / "Notebook", ROOT / "Notebook", ROOT]
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

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "segment" in df.columns and "Segment" not in df.columns:
        rename_map["segment"] = "Segment"
    if "cluster" in df.columns and "Cluster" not in df.columns:
        rename_map["cluster"] = "Cluster"
    if "customer_id" in df.columns and "CustomerID" not in df.columns:
        rename_map["customer_id"] = "CustomerID"
    if "transaction_date" in df.columns and "TransactionDate" not in df.columns:
        rename_map["transaction_date"] = "TransactionDate"
    return df.rename(columns=rename_map)

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def kpi_row(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    n_customers = df["CustomerID"].nunique() if "CustomerID" in df.columns else len(df)
    n_rows = len(df)
    # Monetary detection
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in df.columns), None)
    total_amt = float(df[amt_col].sum()) if amt_col else 0.0
    avg_amt = float(df[amt_col].mean()) if amt_col else 0.0
    with c1: metric_card("Unique Customers", f"{n_customers:,}")
    with c2: metric_card("Rows (Tx)", f"{n_rows:,}")
    with c3: metric_card("Total Monetary", f"{total_amt:,.2f}")
    with c4: metric_card("Avg Monetary", f"{avg_amt:,.2f}")

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("🔎 Filters")
    out = df.copy()

    # Gender
    if "Gender" in out.columns:
        gender = st.sidebar.selectbox("Gender", ["All"] + sorted(out["Gender"].dropna().unique().tolist()))
        if gender != "All":
            out = out[out["Gender"] == gender]

    # Location-ish
    loc_field = next((c for c in ["Location","Branch","City","State","Region"] if c in out.columns), None)
    if loc_field:
        loc = st.sidebar.selectbox(loc_field, ["All"] + sorted(out[loc_field].dropna().unique().tolist()))
        if loc != "All":
            out = out[out[loc_field] == loc]

    # Monetary slider
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in out.columns), None)
    if amt_col:
        mn, mx = float(out[amt_col].min()), float(out[amt_col].max())
        a, b = st.sidebar.slider(f"{amt_col} range", min_value=mn, max_value=mx, value=(mn, mx))
        out = out[(out[amt_col] >= a) & (out[amt_col] <= b)]

    # Date filter if available
    date_col = next((c for c in ["TransactionDate","Date"] if c in out.columns), None)
    if date_col:
        try:
            tmp = pd.to_datetime(out[date_col], errors="coerce")
            min_d, max_d = tmp.min(), tmp.max()
            sel = st.sidebar.date_input("Date window", value=(min_d, max_d))
            if isinstance(sel, tuple) and len(sel) == 2:
                start, end = sel
                mask = (tmp >= pd.to_datetime(start)) & (tmp <= pd.to_datetime(end))
                out = out[mask]
        except Exception:
            pass

    # CustomerID text search
    if "CustomerID" in out.columns:
        q = st.sidebar.text_input("Search CustomerID contains")
        if q.strip():
            out = out[out["CustomerID"].astype(str).str.contains(q.strip(), case=False, na=False)]
    return out

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

# ---------- Load data ----------
found = find_first_csv()
st.sidebar.success(f"Detected {len(found)} candidate tables")
if not found:
    st.error("No CSVs found. Please add one of the expected files under Output/ or Notebook/.")
    st.stop()

chosen_name = st.sidebar.selectbox("Primary dataset", list(found.keys()), index=0)
df = load_csv(found[chosen_name])
df = normalize_columns(df)
if df.empty:
    st.error("The selected dataset is empty or failed to load.")
    st.stop()

# ---------- Navigation ----------
st.sidebar.title("🏦 Retail Banking BI")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "👤 Customer Profile",
    "🧭 Segment Insights",
    "🗺️ Geo View",
    "📆 Seasonality",
    "🤖 Churn Prediction",
    "📂 Data Explorer",
    "⬇️ Export",
])

# Apply global filters for most pages
filtered = sidebar_filters(df) if page not in {"👤 Customer Profile","🤖 Churn Prediction"} else df

# ---------- Overview ----------
if page == "🏠 Overview":
    st.title("🏦 Retail Banking Intelligence Dashboard")
    st.caption(f"Primary table: **{chosen_name}** — Rows: {len(df):,}")
    kpi_row(filtered)

    st.markdown("---")
    section_header("Distributions")
    cols = [c for c in ["Monetary","Amount","TransactionAmount","Frequency","Recency"] if c in filtered.columns]
    if cols:
        c1, c2, c3 = st.columns(3)
        for ax, col in zip([c1,c2,c3], cols[:3]):
            with ax:
                st.plotly_chart(px.histogram(filtered, x=col, nbins=40, title=f"{col} Distribution"),
                                use_container_width=True)

    if "Segment" in filtered.columns:
        section_header("Segment Mix")
        mix = filtered["Segment"].value_counts().reset_index()
        mix.columns = ["Segment","Count"]
        st.plotly_chart(px.bar(mix, x="Segment", y="Count", title="Customers by Segment"), use_container_width=True)

# ---------- Customer Profile ----------
elif page == "👤 Customer Profile":
    st.title("👤 Customer 360° Profile")
    if "CustomerID" not in df.columns:
        st.info("No CustomerID column found.")
    else:
        cust_id = st.text_input("Enter CustomerID (exact):")
        if cust_id:
            sub = df[df["CustomerID"].astype(str) == str(cust_id)]
            if sub.empty:
                st.warning("No records found for that CustomerID.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in sub.columns), None)
                freq = sub["Frequency"].iloc[0] if "Frequency" in sub.columns else np.nan
                rec = sub["Recency"].iloc[0] if "Recency" in sub.columns else np.nan
                seg = sub["Segment"].iloc[0] if "Segment" in sub.columns else "N/A"
                with c1: metric_card("Segment", str(seg))
                with c2: metric_card("Frequency", f"{freq}" if pd.notna(freq) else "N/A")
                with c3: metric_card("Recency", f"{rec}" if pd.notna(rec) else "N/A")
                with c4:
                    if amt_col:
                        metric_card("Total Monetary", f"{sub[amt_col].sum():,.2f}")
                    else:
                        metric_card("Total Monetary", "N/A")

                # Timeline if dates exist
                date_col = next((c for c in ["TransactionDate","Date"] if c in sub.columns), None)
                if date_col:
                    try:
                        t = sub[[date_col]].copy()
                        t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
                        t = t.dropna().sort_values(date_col)
                        if amt_col:
                            fig = px.line(sub.sort_values(date_col), x=date_col, y=amt_col, title="Customer Spend Over Time")
                            st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(sub.head(200), use_container_width=True)
                    except Exception:
                        st.dataframe(sub.head(200), use_container_width=True)
                else:
                    st.dataframe(sub.head(200), use_container_width=True)

# ---------- Segment Insights ----------
elif page == "🧭 Segment Insights":
    st.title("🧭 Segment Insights & Recommendations")
    if "Segment" not in filtered.columns:
        st.info("No 'Segment' column found.")
    else:
        # Profiles
        base_cols = [c for c in ["Recency","Frequency","Monetary"] if c in filtered.columns]
        prof = filtered.groupby("Segment")[base_cols].mean().reset_index()
        melted = prof.melt(id_vars="Segment", var_name="Metric", value_name="Mean")
        st.plotly_chart(px.bar(melted, x="Segment", y="Mean", color="Metric", barmode="group",
                               title="Average R/F/M by Segment"),
                        use_container_width=True)

        # Recommendations (generic mapping)
        rec_map = {
            "VIP": "High spend & loyalty. Offer premium rewards, exclusive access, and early-bird promos.",
            "Loyal": "Consistent activity. Maintain engagement with points multipliers and refer-a-friend bonuses.",
            "At Risk": "Falling activity. Send win-back offers, reminders, and personalized reactivation bundles.",
            "Hibernating": "Very low activity. Use soft-touch campaigns, surveys, and seasonal incentives.",
            "New": "Onboarding stage. Educate about benefits, set up alerts, and welcome discounts.",
        }
        st.subheader("Recommendations by Segment")
        for seg, txt in rec_map.items():
            st.markdown(f"**{seg}:** {txt}")

        # Segment drill-down
        seg_sel = st.selectbox("Drill into segment", sorted(filtered["Segment"].dropna().unique().tolist()))
        seg_df = filtered[filtered["Segment"] == seg_sel]
        st.dataframe(seg_df.head(500), use_container_width=True)
        if len(base_cols) >= 2:
            st.plotly_chart(px.scatter(seg_df, x=base_cols[0], y=base_cols[1],
                                       title=f"{seg_sel}: {base_cols[0]} vs {base_cols[1]}"),
                            use_container_width=True)

# ---------- Geo View ----------
elif page == "🗺️ Geo View":
    st.title("🗺️ Geographic Distribution")
    # Try to find geo columns
    lat_col = next((c for c in ["lat","latitude","Lat","Latitude"] if c in filtered.columns), None)
    lon_col = next((c for c in ["lon","lng","longitude","Long","Longitude"] if c in filtered.columns), None)
    region_col = next((c for c in ["State","Region","City","Location","Branch"] if c in filtered.columns), None)
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in filtered.columns), None)

    if lat_col and lon_col:
        st.caption("Detected latitude/longitude; showing bubble map (Plotly scattergeo).")
        fig = px.scatter_geo(filtered,
                             lat=lat_col, lon=lon_col,
                             size=amt_col if amt_col else None,
                             hover_name=region_col if region_col else None,
                             title="Customer/Transaction Locations")
        st.plotly_chart(fig, use_container_width=True)
    elif region_col:
        st.caption(f"Using {region_col} to show totals.")
        top = filtered.groupby(region_col)[amt_col].sum().reset_index().sort_values(amt_col, ascending=False).head(30) if amt_col else filtered[region_col].value_counts().reset_index().rename(columns={"index":region_col, region_col:"Count"}).head(30)
        if amt_col:
            fig = px.bar(top, x=region_col, y=amt_col, title=f"Top {region_col} by {amt_col}")
        else:
            fig = px.bar(top, x=region_col, y="Count", title=f"Top {region_col} by Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geographic columns detected. Add 'City', 'State', 'Region', or lat/lon to enable maps.")

# ---------- Seasonality ----------
elif page == "📆 Seasonality":
    st.title("📆 Seasonality & Calendar Heatmap")
    date_col = next((c for c in ["TransactionDate","Date"] if c in filtered.columns), None)
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in filtered.columns), None)
    if not date_col:
        st.info("No date column found (looking for TransactionDate or Date).")
    else:
        t = filtered[[date_col]].copy()
        t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
        t = t.dropna()
        t["Year"] = t[date_col].dt.year
        t["Month"] = t[date_col].dt.month
        t["Day"] = t[date_col].dt.day
        if amt_col:
            daily = filtered.copy()
            daily[date_col] = pd.to_datetime(daily[date_col], errors="coerce")
            daily = daily.dropna(subset=[date_col])
            s = daily.groupby(daily[date_col].dt.date)[amt_col].sum().reset_index()
            s.columns = ["Date","Total"]
            fig = px.line(s, x="Date", y="Total", title="Daily Total Amount")
            st.plotly_chart(fig, use_container_width=True)
        # Month-Day heatmap (counts)
        heat = t.groupby(["Month","Day"]).size().reset_index(name="Count")
        fig = px.density_heatmap(heat, x="Day", y="Month", z="Count", title="Calendar Heatmap (Count)",
                                 nbinsx=31, nbinsy=12, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Churn Prediction ----------
elif page == "🤖 Churn Prediction":
    st.title("🤖 Churn Prediction (Baseline)")
    work = df.copy()

    # Identify features
    features = [c for c in ["Recency","Frequency","Monetary"] if c in work.columns]
    if len(features) < 2:
        st.info("Need at least two of Recency/Frequency/Monetary to build a simple model.")
    else:
        # Target: prefer explicit 'Churn' if present; else derive using Recency > 80th percentile
        if "Churn" in work.columns:
            work["target"] = work["Churn"].astype(int)
            target_note = "Using existing 'Churn' column as target."
        else:
            if "Recency" not in work.columns:
                st.info("No explicit 'Churn' and no 'Recency' to derive. Provide a 'Churn' column or Recency.")
                st.stop()
            thr = work["Recency"].quantile(0.80)
            work["target"] = (work["Recency"] >= thr).astype(int)
            target_note = f"Derived churn label: 1 if Recency ≥ {thr:.1f} (top 20% most inactive)."

        st.caption(target_note)

        X = work[features].copy().fillna(0.0)
        y = work["target"].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_sc, y_train)

        # Metrics
        proba = model.predict_proba(X_test_sc)[:,1]
        auc = roc_auc_score(y_test, proba)
        st.metric("ROC-AUC", f"{auc:.3f}")
        st.caption("Baseline logistic regression. Use with caution; tune for your business definition of churn.")

        # Feature effects
        coefs = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_[0]
        }).sort_values("Coefficient", key=lambda s: s.abs(), ascending=False)
        st.subheader("Feature Influence (Logistic Coefficients)")
        st.plotly_chart(px.bar(coefs, x="Feature", y="Coefficient", title="Model Coefficients"),
                        use_container_width=True)

        # Score current filtered customers
        if set(features).issubset(filtered.columns):
            X_curr = filtered[features].copy().fillna(0.0)
            scores = model.predict_proba(scaler.transform(X_curr))[:,1]
            out = filtered.copy()
            out["Churn_Risk_Score"] = np.round(scores, 4)
            st.subheader("Scored Customers (Filtered View)")
            st.dataframe(out.head(1000), use_container_width=True)
            st.download_button("Download scored customers (CSV)",
                               out.to_csv(index=False).encode("utf-8"),
                               file_name="churn_scored_customers.csv",
                               mime="text/csv")

# ---------- Data Explorer ----------
elif page == "📂 Data Explorer":
    st.title("📂 Data Explorer")
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

# ---------- Export ----------
elif page == "⬇️ Export":
    st.title("⬇️ Export")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="filtered_dataset.csv", mime="text/csv")
    st.write("Copy a few rows as CSV:")
    st.code(filtered.head(10).to_csv(index=False))

st.markdown("---")
st.caption("Built for Idowu Malachi • BI app with Customer 360, Segment insights, Geo view, Seasonality, and Churn baseline.")
