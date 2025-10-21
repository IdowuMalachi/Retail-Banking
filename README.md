"# Retail-Banking" 
<h1 align="center">🏦 BankTrust Retail Banking Analytics</h1>
<h3 align="center">Optimizing Customer Relationships Through RFM Segmentation & K-Means Clustering</h3>

<p align="center">
  <b>Specialization:</b> Data Science & Business Analytics ·
  <b>Domain:</b> Financial Services<br/>
  <b>Tools:</b> Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit · Git LFS
</p>

---

## 🌟 Executive Summary
This project demonstrates how a financial institution can use **data-driven segmentation** to uncover customer behavior and improve retention.  
Using **RFM (Recency, Frequency, Monetary)** analysis integrated with **K-Means clustering**, we built a framework to:
- Identify high-value and at-risk customers.
- Personalize engagement campaigns.
- Improve marketing efficiency and ROI.

The analysis pipeline transforms millions of raw transaction records into **clear insights**, **visual dashboards**, and **actionable strategies** for customer relationship teams.

---

## 🧭 Business Problem
BankTrust, a mid-sized retail bank, wanted to understand:
- Which customers drive the most revenue?
- Who is likely to churn or disengage?
- How can campaigns be tailored for loyalty, reactivation, or cross-selling?

The dataset includes ~1 million transactions, covering gender, location, account balance, transaction amount, and timestamps.

---

## ⚙️ Workflow Overview

| Phase | Description | Output |
|-------|--------------|---------|
| **1️⃣ Data Preparation** | Cleaning, formatting, outlier capping, and feature engineering | Cleaned dataset (`Output/cleaned_data_with_segmentation.csv`) |
| **2️⃣ Exploratory Data Analysis (EDA)** | Visual exploration to understand customer patterns | Visualizations under `Visualization/` |
| **3️⃣ RFM Segmentation** | Calculate Recency, Frequency, Monetary metrics | `Output/rfm_customer_table.csv` |
| **4️⃣ K-Means Clustering** | Unsupervised learning to validate business segments | Cluster assignments + evaluation metrics |
| **5️⃣ Dashboards** | 3-page visual summary (EDA, RFM, K-Means) | `Dashboard/` snapshots + Streamlit app |

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA process uncovered **skewed distributions**, **location dominance**, and **behavioral diversity**.

### Key Visuals

<p align="center">
  <img src="Visualization/TransactionAmount_(capped_1%-99%).png" width="48%">
  <img src="Visualization/CustAccountBalance_(capped_1%-99%).png" width="48%">
</p>

<p align="center">
  <img src="Visualization/Transactions_by_Gender.png" width="48%">
  <img src="Visualization/Top_10_Locations_by_Transaction_Count.png" width="48%">
</p>

<p align="center">
  <img src="Visualization/plot_histograms_features.png" width="95%">
</p>

**Insights:**
- 70% of transactions were below the median amount — heavy right skew.
- Males performed 3× more transactions than females.
- Major activity concentrated in **Mumbai**, **Delhi**, and **Bangalore**.
- Recency peaked around 40–60 days, ideal for re-engagement analysis.

---

## 💎 RFM Segmentation

### Formulae
| Metric | Definition | Computation |
|--------|-------------|-------------|
| **Recency (R)** | Days since last transaction | `max(TransactionDate) - last_date_per_customer` |
| **Frequency (F)** | Number of transactions | `count(TransactionID)` |
| **Monetary (M)** | Total amount spent | `sum(TransactionAmount)` |

### Weighting Strategy
A **condition-based weighting** was adopted:
- Recency → 50% (most important for churn risk)
- Frequency → 30%
- Monetary → 20%

The scores were ranked into quantiles (1–5) and combined into 6 named segments:
> `Champions`, `Loyal Customers`, `Potential Loyalists`, `Needs Attention`, `At Risk`, and `Lost`

### Visuals

<p align="center">
  <img src="Visualization/Customer_Segments_by_RFM_Score.png" width="70%">
</p>

<p align="center">
  <img src="Visualization/plot_cluster_profiles_means.png" width="70%">
</p>

**Highlights:**
- 22% of customers are *Needs Attention*.
- *Champions* account for the top 5% of Monetary contribution.
- *At-Risk* group shows long recency but high potential for targeted outreach.

---

## 🤖 K-Means Clustering

### Step 1 — Feature Scaling & Sampling
Scaled `Recency`, `Frequency`, and `Monetary` with StandardScaler to normalize magnitudes.

### Step 2 — Finding Optimal k
We used both the **Elbow** and **Silhouette** methods.

<p align="center">
  <img src="Visualization/plot_elbow_inertia.png" width="48%">
  <img src="Visualization/plot_silhouette_by_k.png" width="48%">
</p>

Best k = **3**, balancing low inertia with high silhouette (~0.53).

### Step 3 — Cluster Profiling

<p align="center">
  <img src="Visualization/plot_pca_clusters.png" width="48%">
  <img src="Visualization/plot_radar_cluster_profiles.png" width="48%">
</p>

### Step 4 — Validating Business Segments

<p align="center">
  <img src="Visualization/plot_stacked_cluster_segment.png" width="48%">
  <img src="Visualization/plot_heatmap_segment_cluster.png" width="48%">
</p>

### Step 5 — Cluster Feature Spread

<p align="center">
  <img src="Visualization/plot_boxplots_features_by_cluster.png" width="95%">
</p>

**Findings:**
- Cluster 1 → Low recency, high frequency & monetary → likely “Champions”.
- Cluster 2 → Moderate frequency, recent transactions → “Potential Loyalists”.
- Cluster 0 → Long recency, low frequency → “Dormant / Lost”.

---

## 🧮 Quantitative Evaluation

| Metric | Description | Value |
|--------|-------------|-------|
| **Adjusted Rand Index** | Similarity between business segments & clusters | 0.68 |
| **Silhouette Score** | Separation & cohesion measure | 0.53 |
| **Homogeneity** | Cluster purity | 0.74 |
| **Completeness** | Segment coverage | 0.70 |
| **V-Measure** | Combined homogeneity + completeness | 0.72 |

---

## 📊 Dashboards


<p align="center">
  <img src="Dashboard/dashboard_eda.png" width="31%"/>
  <img src="Dashboard/dashboard_rfm.png" width="31%"/>
  <img src="Dashboard/dashboard_kmeans.png" width="31%"/>
</p>

<!-- If images still don't show, swap to RAW links:
<img src="https://raw.githubusercontent.com/<your-user>/<your-repo>/main/Dashboard/dashboard_eda.png" width="31%"/>
-->


Each dashboard is backed by Streamlit and Git LFS snapshots.

---

## 🧾 Outputs (Data Links)

| Dataset | Description | Link |
|----------|--------------|------|
| Cleaned Data | Post-EDA cleaned transactions | [Output/cleaned_data_with_segmentation.csv](Output/cleaned_data_with_segmentation.csv) |
| RFM Summary | Customer-level R/F/M and segment | [Output/rfm_customer_table.csv](Output/rfm_customer_table.csv) |
| Crosstab | Segment × Cluster comparison | [Output/segment_vs_cluster_crosstab.csv](Output/segment_vs_cluster_crosstab.csv) |
| Visualization Assets | PNGs for dashboards | [Visualization/](Visualization) |

> All large artifacts are tracked with **Git LFS** for version control and reproducibility.

---

## 🧠 Interpretation & Insights

- **Champions (15%)**  
  *High frequency + high spend + recent activity* → reward with premium benefits and personalized offers.

- **Loyal Customers (18%)**  
  *Moderate frequency and spend* → maintain with loyalty programs and tier upgrades.

- **At-Risk (14%)**  
  *High recency, declining frequency* → send reactivation reminders and fee waivers.

- **Potential Loyalists (12%)**  
  *Frequent new customers with growing spend* → push cross-sell and product education.

- **Lost (20%)**  
  *No activity in months* → evaluate retention cost vs. revenue.

---

## 🚀 Deployment Path
A Streamlit prototype dashboard (`app_streamlit.py`) enables:
- Live customer filtering by recency range.
- On-the-fly cluster visualization.
- Simulated “what-if” marketing scenarios.

Launch locally:
```bash
streamlit run app_streamlit.py
