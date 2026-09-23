
The 5 CBMS institutional tables are already built into the code.
""")
st.stop()

df = preprocess(df_raw)

with st.expander("🔍 Data Summary", expanded=False):
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Barangays", df['barangay'].nunique() if 'barangay' in df.columns else 0)
c3.metric("Tribes", df['tribe'].nunique() if 'tribe' in df.columns else 0)
c4.metric("Years Covered", df['year'].nunique() if 'year' in df.columns else 0)

st.write("**Records per Barangay:**")
if 'barangay' in df.columns:
    st.dataframe(df['barangay'].value_counts().rename('Records').to_frame(),
                 use_container_width=True)

st.write("**Records per Tribe:**")
if 'tribe' in df.columns:
    st.dataframe(df['tribe'].value_counts().rename('Records').to_frame(),
                 use_container_width=True)

st.write("**Sample Rows:**")
st.dataframe(df.head(5), use_container_width=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Analysis Controls")

barangay_filter = st.sidebar.multiselect(
"Filter by Barangay:",
options=sorted(df['barangay'].dropna().unique()) if 'barangay' in df.columns else [],
default=sorted(df['barangay'].dropna().unique()) if 'barangay' in df.columns else []
)

tribe_filter = st.sidebar.multiselect(
"Filter by Tribe:",
options=sorted(df['tribe'].dropna().unique()) if 'tribe' in df.columns else [],
default=sorted(df['tribe'].dropna().unique()) if 'tribe' in df.columns else []
)

k_clusters = st.sidebar.slider("K-Means: number of clusters", 2, 6, 3)
min_support = st.sidebar.slider("Apriori: min support", 0.05, 0.50, 0.15, 0.05)
min_confidence = st.sidebar.slider("Apriori: min confidence", 0.3, 1.0, 0.6, 0.05)

# Apply filters
df_f = df.copy()
if barangay_filter and 'barangay' in df_f.columns:
df_f = df_f[df_f['barangay'].isin(barangay_filter)]
if tribe_filter and 'tribe' in df_f.columns:
df_f = df_f[df_f['tribe'].isin(tribe_filter)]

if len(df_f) < 20:
st.warning("⚠️ Too few records after filtering. Adjust the sidebar filters.")
st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
"📊 Overview", "📈 EDA",
"🎯 K-Means Clustering",
"🔗 Apriori Rules",
"💡 Recommendations"
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
st.header("📊 Executive Overview")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card"><p>Respondents</p><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><p>Barangays</p><h2>{df_f["barangay"].nunique() if "barangay" in df_f.columns else 0}</h2></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><p>Tribes</p><h2>{df_f["tribe"].nunique() if "tribe" in df_f.columns else 0}</h2></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><p>Service Types</p><h2>{df_f["service_type"].nunique() if "service_type" in df_f.columns else 0}</h2></div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("📌 Respondents per Barangay")
if 'barangay' in df_f.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    df_f['barangay'].value_counts().plot(kind='bar', color='#1f4e79', edgecolor='black', ax=ax)
    ax.set_ylabel("Respondents")
    ax.set_title("Respondents per Barangay")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.subheader("🌱 Indigenous Group Distribution")
if 'tribe' in df_f.columns:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = df_f['tribe'].value_counts()
        ax.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Tribe Distribution")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(df_f['tribe'].value_counts().rename('Count').to_frame().assign(
            Percentage=lambda d: (d['Count'] / d['Count'].sum() * 100).round(2)
        ), use_container_width=True)


# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
st.header("📈 Exploratory Data Analysis")

st.subheader("👥 Socio-Demographic Profile")

col1, col2 = st.columns(2)
with col1:
    if 'age' in df_f.columns:
        ages = df_f['age'].dropna()
        if len(ages) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(ages, bins=15, color='#16a085', edgecolor='black')
            ax.set_xlabel("Age")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Age Distribution (mean = {ages.mean():.1f})")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

with col2:
    if 'gender' in df_f.columns:
        vc = df_f['gender'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.plot(kind='bar', color='#27ae60', edgecolor='black', ax=ax)
        ax.set_ylabel("Respondents")
        ax.set_title("Gender Distribution")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

col3, col4 = st.columns(2)
with col3:
    if 'education_level' in df_f.columns:
        vc = df_f['education_level'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.sort_values().plot(kind='barh', color='#3498db', edgecolor='black', ax=ax)
        ax.set_xlabel("Respondents")
        ax.set_title("Education Level")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with col4:
    if 'employment_status' in df_f.columns:
        vc = df_f['employment_status'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(7, 4))
        vc.sort_values().plot(kind='barh', color='#e67e22', edgecolor='black', ax=ax)
        ax.set_xlabel("Respondents")
        ax.set_title("Employment Status")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.subheader("💰 Household Income")
if 'income_weekly' in df_f.columns:
    inc = df_f['income_weekly'].dropna()
    if len(inc) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(inc, bins=15, color='#9b59b6', edgecolor='black')
            ax.set_xlabel("Weekly Income (PHP)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Weekly Household Income (mean = ₱{inc.mean():.0f})")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("**Income statistics:**")
            st.dataframe(inc.describe().round(2).rename('Income (PHP/week)').to_frame(),
                         use_container_width=True)

st.markdown("---")
st.subheader("🏘️ Barangay-level Service Access (CBMS Institutional)")
cbms = get_cbms_tables()

tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
    "Social Assistance", "Feeding Program", "Labor Market",
    "Agri/Fisheries", "Bayanihan"
])
with tab_a:
    st.dataframe(cbms['social_assistance'], use_container_width=True)
with tab_b:
    st.dataframe(cbms['feeding_programs'], use_container_width=True)
with tab_c:
    st.dataframe(cbms['labor_market'], use_container_width=True)
with tab_d:
    st.dataframe(cbms['agri_fisheries'], use_container_width=True)
with tab_e:
    st.dataframe(cbms['bayanihan'], use_container_width=True)

st.markdown("---")
st.subheader("🚧 Barriers to Service Access")

barrier_cols = ['financial_barrier', 'distance_barrier', 'language_barrier', 'info_barrier']
available_barriers = [c for c in barrier_cols if c in df_f.columns]

if available_barriers:
    for i in range(0, len(available_barriers), 2):
        cols = st.columns(2)
        for j, col_name in enumerate(available_barriers[i:i+2]):
            with cols[j]:
                vc = df_f[col_name].dropna().value_counts().head(8)
                if len(vc) > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    vc.sort_values().plot(kind='barh', color='#c0392b', edgecolor='black', ax=ax)
                    ax.set_xlabel("Respondents")
                    ax.set_title(col_name.replace('_', ' ').title())
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()


# ------------------------------------------------------------
# TAB 3: K-MEANS
# ------------------------------------------------------------
with tab3:
st.header("🎯 K-Means Clustering — Service Access Patterns")
st.caption("Groups households by similar socio-economic profile and access characteristics.")

# ---- Build feature matrix ----
feature_frame = pd.DataFrame(index=df_f.index)

if 'age' in df_f.columns:
    feature_frame['age'] = df_f['age']
if 'household_size' in df_f.columns:
    feature_frame['household_size'] = df_f['household_size']
if 'income_weekly' in df_f.columns:
    feature_frame['income_weekly'] = df_f['income_weekly']
if 'time_spent_hours' in df_f.columns:
    feature_frame['time_spent_hours'] = df_f['time_spent_hours']

# Encode categorical features
cat_features = ['gender', 'education_level', 'employment_status',
                'access_transportation', 'access_electricity', 'access_internet',
                'service_frequency_clean', 'civil_status']

for col in cat_features:
    if col in df_f.columns:
        le = LabelEncoder()
        vals = df_f[col].astype(str).fillna('Unknown')
        feature_frame[col + '_enc'] = le.fit_transform(vals)

# Drop rows with any NaN
feature_frame = feature_frame.dropna()

if len(feature_frame) < k_clusters * 2:
    st.warning("Not enough complete records for clustering after filtering.")
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_frame)

    # ---- K-Means ----
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
    dbi = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters (K)", k_clusters)
    c2.metric("Records Clustered", len(labels))
    c3.metric("Silhouette Score", f"{sil:.4f}")
    c4.metric("Davies-Bouldin Index", f"{dbi:.4f}")

    st.caption(
        f"**Interpretation:** Silhouette closer to **+1.0** = better-separated clusters. "
        f"Score of **{sil:.4f}** suggests "
        f"{'excellent' if sil > 0.7 else 'acceptable' if sil > 0.5 else 'weak' if sil > 0.25 else 'poor'} "
        f"cluster quality. Lower DBI (closer to 0) = better separation."
    )

    # ---- PCA plot ----
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cluster Sizes")
        counts = pd.Series(labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_km = plt.cm.Set2(np.linspace(0, 1, k_clusters))
        ax.bar([f"C{i}" for i in counts.index], counts.values,
               color=colors_km, edgecolor='black')
        ax.set_ylabel("Respondents")
        ax.set_title("Respondents per Cluster")
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("PCA Visualization")
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10',
                        s=40, alpha=0.8, edgecolors='black')
        plt.colorbar(sc, label='Cluster')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Households in PCA Space")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ---- Elbow curve for reference ----
    st.markdown("---")
    st.subheader("📐 Elbow Method (WCSS) — Reference")
    wcss = []
    max_k = min(10, len(feature_frame) - 1)
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, max_k + 1), wcss, marker='o', color='#1f4e79')
    ax.axvline(k_clusters, color='red', linestyle='--', label=f'Chosen K = {k_clusters}')
    ax.set_xlabel("K")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method — Within-Cluster Sum of Squares")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ---- Cluster profiles ----
    st.markdown("---")
    st.subheader("🔍 Cluster Profiles")
    cluster_df = feature_frame.copy()
    cluster_df['_cluster'] = labels
    cluster_df['_barangay'] = df_f.loc[feature_frame.index, 'barangay'].values
    cluster_df['_tribe'] = df_f.loc[feature_frame.index, 'tribe'].values if 'tribe' in df_f.columns else None

    for c in sorted(cluster_df['_cluster'].unique()):
        sub = cluster_df[cluster_df['_cluster'] == c]
        st.markdown(f"### 🔹 Cluster {c} — {len(sub)} respondents ({len(sub)/len(cluster_df)*100:.1f}%)")

        m1, m2, m3 = st.columns(3)
        if 'age' in sub.columns:
            m1.metric("Mean Age", f"{sub['age'].mean():.1f}")
        if 'income_weekly' in sub.columns:
            m2.metric("Mean Weekly Income", f"₱{sub['income_weekly'].mean():.0f}")
        if 'household_size' in sub.columns:
            m3.metric("Mean HH Size", f"{sub['household_size'].mean():.1f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Barangays in this Cluster:**")
            st.dataframe(sub['_barangay'].value_counts().head(5).rename('Count'),
                         use_container_width=True)
        with col2:
            if sub['_tribe'].notna().any():
                st.markdown("**Top Tribes in this Cluster:**")
                st.dataframe(sub['_tribe'].value_counts().head(5).rename('Count'),
                             use_container_width=True)

        st.markdown("---")

    # ---- Interpretation helper ----
    st.markdown("### 🏷️ Cluster Interpretation")
    st.caption("Based on your Chapter 3 — Cluster 1 = High Access, Cluster 2 = Moderate, Cluster 3 = Low")
    for c in sorted(cluster_df['_cluster'].unique()):
        sub = cluster_df[cluster_df['_cluster'] == c]
        income_mean = sub['income_weekly'].mean() if 'income_weekly' in sub.columns else 0
        if income_mean > 1000:
            label = "High Access (Higher income, more resources)"
        elif income_mean > 700:
            label = "Moderate Access"
        else:
            label = "Low Access (Lower income, more barriers)"
        st.markdown(f"- **Cluster {c}** → **{label}** (mean weekly income = ₱{income_mean:.0f})")


# ------------------------------------------------------------
# TAB 4: APRIORI
# ------------------------------------------------------------
with tab4:
st.header("🔗 Apriori Association Rule Mining")
st.caption("Discovers patterns like 'if respondent is in tribe X and has barrier Y, they access service Z'.")

# Build transactions
txn_cols = ['barangay', 'tribe', 'service_type', 'service_frequency_clean',
            'mode_of_access', 'financial_barrier', 'distance_barrier',
            'language_barrier', 'info_barrier', 'employment_status']

available_txn = [c for c in txn_cols if c in df_f.columns]

if not available_txn:
    st.warning("No suitable columns found for transaction encoding.")
else:
    transactions_df = df_f[available_txn].astype(str).copy()
    transactions = transactions_df.values.tolist()
    transactions = [
        [f"{col}={val}" for col, val in zip(available_txn, row)
         if val and val.lower() not in ('nan', 'none', '')]
        for row in transactions
    ]
    transactions = [t for t in transactions if len(t) >= 2]

    if len(transactions) < 20:
        st.warning("Not enough transactions after filtering.")
    else:
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_ary, columns=te.columns_)

        try:
            frequent = apriori(basket, min_support=min_support, use_colnames=True)
        except Exception as e:
            frequent = pd.DataFrame()
            st.error(f"Apriori error: {e}")

        if frequent.empty:
            st.warning(f"No frequent itemsets found with min support = {min_support}. Lower it.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Frequent Itemsets", len(frequent))
            c2.metric("Min Support", f"{min_support:.3f}")
            c3.metric("Min Confidence", f"{min_confidence:.2f}")

            st.subheader("📋 Top 15 Frequent Itemsets")
            top_itemsets = frequent.sort_values('support', ascending=False).head(15).copy()
            top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(x)))
            st.dataframe(top_itemsets, use_container_width=True)

            try:
                rules = association_rules(frequent, metric='confidence', min_threshold=min_confidence)
                rules = rules.sort_values('lift', ascending=False)
            except Exception as e:
                rules = pd.DataFrame()
                st.warning(f"Rule generation error: {e}")

            if not rules.empty:
                st.markdown("---")
                st.subheader(f"📌 Top 15 Association Rules (out of {len(rules)})")

                display = rules.head(15).copy()
                display['antecedents'] = display['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                display['consequents'] = display['consequents'].apply(lambda x: ', '.join(sorted(x)))

                st.dataframe(
                    display[['antecedents', 'consequents', 'support',
                             'confidence', 'lift']].round(4),
                    use_container_width=True
                )

                st.caption(
                    "**Support** = how often the rule appears. "
                    "**Confidence** = reliability. "
                    "**Lift > 1** = positive association (better than random)."
                )

                # Download
                dl_rules = rules.copy()
                dl_rules['antecedents'] = dl_rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                dl_rules['consequents'] = dl_rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
                st.download_button(
                    "📥 Download Association Rules CSV",
                    dl_rules.to_csv(index=False).encode('utf-8'),
                    "apriori_rules.csv", "text/csv"
                )
            else:
                st.info(f"No rules found with confidence ≥ {min_confidence}. Lower the threshold.")


# ------------------------------------------------------------
# TAB 5: RECOMMENDATIONS
# ------------------------------------------------------------
with tab5:
st.header("💡 Recommendations & Interpretation")

st.markdown(f"""
### 📊 Analysis Summary

Based on **{len(df_f):,}** indigenous respondents across **{df_f['barangay'].nunique() if 'barangay' in df_f.columns else 0}** barangays in Bunawan, Agusan del Sur, using **K-Means Clustering** and **Apriori Association Rule Mining**.

#### 1. For the Local Government Unit (LGU) of Bunawan
- **Prioritize low-access clusters** identified in the K-Means tab — they have the
  lowest income and most barriers to accessing government services.
- **Deploy mobile service units** to remote barangays where distance and transportation
  barriers are most frequent.
- **Allocate social assistance budgets** proportionally to high-need clusters,
  using CBMS data as evidence.

#### 2. For the National Commission on Indigenous Peoples (NCIP)
- Use cluster profiles to design **culturally appropriate interventions** that respect
  the distinct needs of Manobo, Banwaon, and Talaandig households.
- Align program delivery with the **Indigenous Peoples' Rights Act (IPRA)** and
  Free, Prior, and Informed Consent (FPIC) principles.

#### 3. For Government Agencies (DSWD, DOLE, DA, DTI)
- **Coordinate** program delivery to avoid duplicating benefits in high-access clusters
  while leaving low-access clusters underserved.
- **Translate program materials** into local languages to reduce language barriers.

#### 4. For Indigenous Communities
- **Engage** with barangay officials for a more transparent picture of available
  programs and how to access them.
- **Participate in community consultations** that shape locally responsive policies.

#### 5. For Future Researchers
- **Extend** with Random Forest, XGBoost, or Neural Networks for predictive modeling.
- **Add geospatial analysis** (GIS) to visualize service access across barangays.
- **Longitudinal studies** to track changes in service access over time (2022–2026).
- **Qualitative interviews** to complement the quantitative clusters with lived
  experiences of the indigenous community.

#### 6. Limitations
- Only **reported** respondents are analyzed — the actual population is larger.
- Cluster results depend on the **22 variables** collected; missing variables may
  hide barriers.
- Cross-sectional design limits causal inference.
- Findings specific to Bunawan; may not generalize to other municipalities.
""")

st.markdown("---")
st.subheader("📥 Download Results")
try:
    summary = pd.DataFrame({
        'Metric': ['Total Respondents', 'Barangays', 'Tribes', 'K-Means K'],
        'Value': [
            len(df_f),
            df_f['barangay'].nunique() if 'barangay' in df_f.columns else 0,
            df_f['tribe'].nunique() if 'tribe' in df_f.columns else 0,
            k_clusters
        ]
    })
    st.download_button(
        "📊 Download Summary CSV",
        summary.to_csv(index=False).encode('utf-8'),
        "indigenous_access_summary.csv", "text/csv"
    )
except Exception:
    pass


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
"🎓 Capstone Dashboard • Analysis of Service Access Patterns Among Indigenous Communities "
"in Bunawan, Agusan del Sur Using Data Mining • "
"Falcasantos, K.M.P. & Adrales, K.J. • Agusan del Sur State University • 2026"
)