
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
    ren = {}
    for a,b in {
        "segment":"Segment",
        "cluster":"Cluster",
        "customer_id":"CustomerID",
        "transaction_date":"TransactionDate",
        "amount":"Amount",
        "monetary":"Monetary",
        "frequency":"Frequency",
        "recency":"Recency",
        "gender":"Gender"
    }.items():
        if a in df.columns and b not in df.columns:
            ren[a] = b
    return df.rename(columns=ren)

# ---------- KPI helpers ----------
def compute_kpis(df: pd.DataFrame):
    n_customers = df["CustomerID"].nunique() if "CustomerID" in df.columns else len(df)
    n_rows = len(df)
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in df.columns), None)
    total_amt = float(df[amt_col].sum()) if amt_col else 0.0
    avg_amt = float(df[amt_col].mean()) if amt_col else 0.0
    return n_customers, n_rows, total_amt, avg_amt, amt_col

def donut_kpi(title: str, value: float, fmt: str = ",.0f"):
    # single-slice donut with number annotation
    fig = go.Figure(go.Pie(labels=[title], values=[max(value, 1e-12)], hole=0.72, textinfo="none"))
    fig.update_layout(
        showlegend=False,
        annotations=[dict(text=f"{value:{fmt}}", x=0.5, y=0.5, font_size=22, showarrow=False)],
        height=220, margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# ---------- Sidebar filters (expanded) ----------
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("🔎 Filters")
    out = df.copy()

    # Sample toggle to keep UI fast
    sample_on = st.sidebar.checkbox("Sample for speed (100k rows max)", value=True)
    if sample_on and len(out) > 100_000:
        out = out.sample(100_000, random_state=42)

    # Standard categoricals
    for col in ["Gender","Segment","Cluster"]:
        if col in out.columns:
            choices = sorted([x for x in out[col].dropna().astype(str).unique().tolist()])
            sel = st.sidebar.multiselect(col, choices, default=choices if len(choices) <= 8 else [])
            if sel:
                out = out[out[col].astype(str).isin(sel)]

    # Location-like field
    loc_field = next((c for c in ["Location","Branch","City","State","Region"] if c in out.columns), None)
    if loc_field:
        loc_choices = sorted([x for x in out[loc_field].dropna().astype(str).unique().tolist()])
        loc_sel = st.sidebar.multiselect(loc_field, loc_choices, default=[])
        if loc_sel:
            out = out[out[loc_field].astype(str).isin(loc_sel)]

    # Numeric sliders for R/F/M
    for col in ["Recency","Frequency","Monetary"]:
        if col in out.columns:
            col_min, col_max = float(out[col].min()), float(out[col].max())
            if np.isfinite(col_min) and np.isfinite(col_max):
                r = st.sidebar.slider(f"{col} range", min_value=float(col_min), max_value=float(col_max),
                                      value=(float(col_min), float(col_max)))
                out = out[(out[col] >= r[0]) & (out[col] <= r[1])]

    # Amount-style global slider
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in out.columns), None)
    if amt_col:
        mn, mx = float(out[amt_col].min()), float(out[amt_col].max())
        rng = st.sidebar.slider(f"{amt_col} range", min_value=float(mn), max_value=float(mx), value=(float(mn), float(mx)))
        out = out[(out[amt_col] >= rng[0]) & (out[amt_col] <= rng[1])]

    # Date range
    date_col = next((c for c in ["TransactionDate","Date"] if c in out.columns), None)
    if date_col:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        if dt.notna().any():
            dmin = dt.min().date()
            dmax = dt.max().date()
            start, end = st.sidebar.date_input("Date window", value=(dmin, dmax))
            if isinstance(start, date) and isinstance(end, date):
                mask = (dt.dt.date >= start) & (dt.dt.date <= end)
                out = out[mask]

    # Customer text search
    if "CustomerID" in out.columns:
        q = st.sidebar.text_input("CustomerID contains")
        if q.strip():
            out = out[out["CustomerID"].astype(str).str.contains(q.strip(), case=False, na=False)]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Filtered rows: **{len(out):,}**")
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

filtered = sidebar_filters(df) if page not in {"👤 Customer Profile","🤖 Churn Prediction"} else df

# ---------- Pages ----------
if page == "🏠 Overview":
    st.title("🏦 Retail Banking Intelligence Dashboard")
    st.caption(f"Primary table: **{chosen_name}** — Rows: {len(df):,}")

    n_customers, n_rows, total_amt, avg_amt, amt_col = compute_kpis(filtered)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("Unique Customers"); st.plotly_chart(donut_kpi("Unique Customers", n_customers, fmt=",.0f"), use_container_width=True)
    with c2:
        st.subheader("Rows (Tx)"); st.plotly_chart(donut_kpi("Rows (Tx)", n_rows, fmt=",.0f"), use_container_width=True)
    with c3:
        st.subheader("Total Monetary"); st.plotly_chart(donut_kpi("Total Monetary", total_amt, fmt=",.2f"), use_container_width=True)
    with c4:
        st.subheader("Avg Monetary"); st.plotly_chart(donut_kpi("Avg Monetary", avg_amt, fmt=",.2f"), use_container_width=True)

    st.markdown("---"); section_header("Distributions")
    dist_cols = [c for c in ["Monetary","Amount","TransactionAmount","Frequency","Recency"] if c in filtered.columns]
    if dist_cols:
        cols = st.columns(min(3, len(dist_cols)))
        for ax, col in zip(cols, dist_cols[:3]):
            try:
                st.plotly_chart(px.histogram(filtered, x=col, nbins=40, title=f"{col} Distribution"), use_container_width=True)
            except Exception as e:
                st.info(f"Could not draw {col} histogram: {e}")

    if "Segment" in filtered.columns:
        section_header("Segment Mix")
        mix = filtered["Segment"].value_counts().reset_index()
        mix.columns = ["Segment","Count"]
        st.plotly_chart(px.bar(mix, x="Segment", y="Count", title="Customers by Segment"), use_container_width=True)

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
                n_customers, n_rows, total_amt, avg_amt, amt_col = compute_kpis(sub)
                c1, c2, c3, c4 = st.columns(4)
                seg = sub["Segment"].iloc[0] if "Segment" in sub.columns else "N/A"
                rec = sub["Recency"].iloc[0] if "Recency" in sub.columns else np.nan
                freq = sub["Frequency"].iloc[0] if "Frequency" in sub.columns else np.nan
                with c1: st.plotly_chart(donut_kpi("Segment", 1, fmt=""), use_container_width=True); st.caption(f"Segment: **{seg}**")
                with c2: st.plotly_chart(donut_kpi("Frequency", float(freq) if pd.notna(freq) else 0, fmt=",.0f"), use_container_width=True)
                with c3: st.plotly_chart(donut_kpi("Recency", float(rec) if pd.notna(rec) else 0, fmt=",.0f"), use_container_width=True)
                with c4: st.plotly_chart(donut_kpi("Total Monetary", total_amt, fmt=",.2f"), use_container_width=True)

                date_col = next((c for c in ["TransactionDate","Date"] if c in sub.columns), None)
                if date_col and amt_col:
                    t = sub.copy()
                    t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
                    t = t.dropna(subset=[date_col])
                    if not t.empty:
                        st.plotly_chart(px.line(t.sort_values(date_col), x=date_col, y=amt_col, title="Spend Over Time"),
                                        use_container_width=True)
                st.dataframe(sub.head(300), use_container_width=True)

elif page == "🧭 Segment Insights":
    st.title("🧭 Segment Insights & Recommendations")
    if "Segment" not in filtered.columns:
        st.info("No 'Segment' column found.")
    else:
        base_cols = [c for c in ["Recency","Frequency","Monetary"] if c in filtered.columns]
        if base_cols:
            prof = filtered.groupby("Segment")[base_cols].mean().reset_index()
            melted = prof.melt(id_vars="Segment", var_name="Metric", value_name="Mean")
            st.plotly_chart(px.bar(melted, x="Segment", y="Mean", color="Metric", barmode="group",
                                   title="Average R/F/M by Segment"), use_container_width=True)
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
        seg_sel = st.selectbox("Drill into segment", sorted([x for x in filtered["Segment"].dropna().unique().tolist()]))
        seg_df = filtered[filtered["Segment"] == seg_sel]
        st.dataframe(seg_df.head(500), use_container_width=True)
        if base_cols and len(base_cols) >= 2:
            st.plotly_chart(px.scatter(seg_df, x=base_cols[0], y=base_cols[1],
                                       title=f"{seg_sel}: {base_cols[0]} vs {base_cols[1]}"),
                            use_container_width=True)

elif page == "🗺️ Geo View":
    st.title("🗺️ Geographic Distribution")
    lat_col = next((c for c in ["lat","latitude","Lat","Latitude"] if c in filtered.columns), None)
    lon_col = next((c for c in ["lon","lng","longitude","Long","Longitude"] if c in filtered.columns), None)
    region_col = next((c for c in ["State","Region","City","Location","Branch"] if c in filtered.columns), None)
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in filtered.columns), None)
    if lat_col and lon_col:
        fig = px.scatter_geo(filtered, lat=lat_col, lon=lon_col,
                             size=amt_col if amt_col else None,
                             hover_name=region_col if region_col else None,
                             title="Customer/Transaction Locations")
        st.plotly_chart(fig, use_container_width=True)
    elif region_col:
        if amt_col:
            top = filtered.groupby(region_col)[amt_col].sum().reset_index().sort_values(amt_col, ascending=False).head(30)
            fig = px.bar(top, x=region_col, y=amt_col, title=f"Top {region_col} by {amt_col}")
        else:
            top = filtered[region_col].value_counts().reset_index().rename(columns={"index":region_col, region_col:"Count"}).head(30)
            fig = px.bar(top, x=region_col, y="Count", title=f"Top {region_col} by Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geographic columns detected. Add 'City', 'State', 'Region', or lat/lon to enable maps.")

elif page == "📆 Seasonality":
    st.title("📆 Seasonality & Calendar Heatmap")
    date_col = next((c for c in ["TransactionDate","Date"] if c in filtered.columns), None)
    amt_col = next((c for c in ["Monetary","Amount","TransactionAmount","TotalAmount"] if c in filtered.columns), None)
    if not date_col:
        st.info("No date column found (looking for TransactionDate or Date).")
    else:
        t = filtered.copy()
        t[date_col] = pd.to_datetime(t[date_col], errors="coerce")
        t = t.dropna(subset=[date_col])
        if not t.empty and amt_col:
            s = t.groupby(t[date_col].dt.date)[amt_col].sum().reset_index()
            s.columns = ["Date","Total"]
            st.plotly_chart(px.line(s, x="Date", y="Total", title="Daily Total Amount"), use_container_width=True)
        # Month-Day heatmap with counts (robust)
        if not t.empty:
            heat = t.groupby([t[date_col].dt.month.rename("Month"), t[date_col].dt.day.rename("Day")]).size().reset_index(name="Count")
            fig = px.imshow(
                heat.pivot(index="Month", columns="Day", values="Count").fillna(0).values,
                aspect="auto",
                labels=dict(x="Day", y="Month", color="Count"),
                title="Calendar Heatmap (Count)"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Churn Prediction":
    st.title("🤖 Churn Prediction (Baseline)")
    work = df.copy()
    features = [c for c in ["Recency","Frequency","Monetary"] if c in work.columns]
    if len(features) < 2:
        st.info("Need at least two of Recency/Frequency/Monetary to build a simple model.")
    else:
        if "Churn" in work.columns:
            work["target"] = work["Churn"].astype(int)
            st.caption("Using existing 'Churn' column as target.")
        else:
            base = "Recency" if "Recency" in work.columns else features[0]
            thr = work[base].quantile(0.80)
            work["target"] = (work[base] >= thr).astype(int)
            st.caption(f"Derived churn label: 1 if {base} ≥ {thr:.1f} (top 20% most inactive).")
        # Downsample for speed if very large
        if len(work) > 300_000:
            work = work.sample(300_000, random_state=42)
        X = work[features].fillna(0.0)
        y = work["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, n_jobs=None)
        model.fit(X_train_sc, y_train)
        proba = model.predict_proba(X_test_sc)[:,1]
        auc = roc_auc_score(y_test, proba)
        st.metric("ROC-AUC", f"{auc:.3f}")
        coefs = pd.DataFrame({"Feature": features, "Coefficient": model.coef_[0]}).sort_values("Coefficient", key=lambda s: s.abs(), ascending=False)
        st.subheader("Feature Influence (Logistic Coefficients)")
        st.plotly_chart(px.bar(coefs, x="Feature", y="Coefficient", title="Model Coefficients"), use_container_width=True)
        if set(features).issubset(filtered.columns):
            scores = model.predict_proba(scaler.transform(filtered[features].fillna(0.0)))[:,1]
            out = filtered.copy()
            out["Churn_Risk_Score"] = np.round(scores, 4)
            st.subheader("Scored Customers (Filtered View)")
            st.dataframe(out.head(1000), use_container_width=True)
            st.download_button("Download scored customers (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="churn_scored_customers.csv", mime="text/csv")

elif page == "📂 Data Explorer":
    st.title("📂 Data Explorer")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.caption(f"Showing up to 1,000 rows. Filtered total: {len(filtered):,}")
    with st.expander("Quick Group-By (pivot)"):
        numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
        cat_cols = [c for c in filtered.columns if c not in numeric_cols]
        by = st.multiselect("Group by (categorical)", cat_cols[:10])
        agg_col = st.selectbox("Aggregate column", numeric_cols) if numeric_cols else None
        agg_fn = st.selectbox("Aggregation", ["mean","sum","count","median"]) if agg_col else None
        if by and agg_col and agg_fn:
            if agg_fn == "count":
                pivot = filtered.groupby(by)[agg_col].count().reset_index(name="count")
            else:
                pivot = getattr(filtered.groupby(by)[agg_col], agg_fn)().reset_index(name=f"{agg_fn}_{agg_col}")
            st.dataframe(pivot, use_container_width=True)
            st.plotly_chart(px.bar(pivot, x=by[0], y=pivot.columns[-1], color=by[1] if len(by) > 1 else None,
                                   title="Group-By Chart"), use_container_width=True)

elif page == "⬇️ Export":
    st.title("⬇️ Export")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="filtered_dataset.csv", mime="text/csv")
    st.write("Copy a few rows as CSV:")
    st.code(filtered.head(10).to_csv(index=False))

st.markdown("---")
st.caption("Built for Idowu Malachi • Robust BI app with richer filters, KPI donuts, Customer 360, Segment insights, Geo view, Seasonality, and Churn baseline.")
