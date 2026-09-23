import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Indigenous Service Access Dashboard",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: bold; color: #1f4e79;
        text-align: center; padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
    }
    .sub-header {
        font-size: 1.05rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #16a085 0%, #27ae60 100%);
        padding: 1.3rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 1.9rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def safe_str(val):
    """Convert any value to a clean string, handling NaN/None/pd.NA."""
    if val is None:
        return ''
    try:
        if pd.isna(val):
            return ''
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def safe_lower(val):
    return safe_str(val).lower()


def is_blank(val):
    return safe_lower(val) in ('', 'nan', 'none', 'nat', 'null', '<na>')


# ============================================================
# CBMS INSTITUTIONAL TABLES (hardcoded)
# ============================================================
def get_cbms_tables():
    social_assistance_df = pd.DataFrame({
        'Barangay': ['Bunawan Brook', 'Consuelo', 'Imelda', 'Libertad', 'Mambalili',
                     'Nueva Era', 'Poblacion', 'San Andres', 'San Marcos', 'San Teodoro'],
        'Total_Households': [1526, 2825, 429, 1789, 810, 505, 1236, 1202, 274, 1996],
        'Benefited_At_Least_One_SAP': [542, 1047, 115, 309, 244, 185, 208, 144, 147, 643],
        'Regular_CCT_4Ps': [385, 411, 107, 454, 259, 129, 396, 238, 85, 273],
        'Modified_CCT_4Ps': [45, 35, 18, 42, 111, 10, 11, 31, 7, 58],
        'Unconditional_Cash_Transfer_UCT': [39, 21, 1, 27, 30, 4, 9, 6, 0, 13],
        'SPISC_SocPen': [116, 134, 21, 209, 36, 4, 8, 64, 0, 53],
        'IMAP': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        'StuFAP': [5, 0, 0, 4, 1, 1, 1, 0, 0, 19],
        'Senior_High_Voucher': [8, 1, 0, 27, 0, 0, 2, 14, 0, 2],
        'ESA': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        'Housing_Program': [17, 2, 0, 5, 0, 0, 1, 0, 0, 1],
        'Health_Assistance': [6, 10, 0, 3, 0, 0, 0, 9, 4, 2],
    })

    feeding_programs_df = pd.DataFrame({
        'Barangay': ['Bunawan Brook', 'Consuelo', 'Imelda', 'Libertad', 'Mambalili',
                     'Nueva Era', 'Poblacion', 'San Andres', 'San Marcos', 'San Teodoro'],
        'Benefited_From_Feeding': [106, 83, 2, 55, 9, 36, 133, 16, 1, 35],
        'Did_Not_Benefit': [1420, 2742, 427, 1734, 801, 469, 1103, 1186, 273, 1961],
        'Total': [1526, 2825, 429, 1789, 810, 505, 1236, 1202, 274, 1996],
        'Percent_Benefited': [6.95, 2.94, 0.47, 3.07, 1.11, 7.13, 10.76, 1.33, 0.36, 1.75],
        'Percent_Not_Benefited': [93.05, 97.06, 99.53, 96.93, 98.89, 92.87, 89.24, 98.67, 99.64, 98.25],
    })

    labor_market_df = pd.DataFrame({
        'Barangay': ['Bunawan Brook', 'Consuelo', 'Imelda', 'Libertad', 'Mambalili',
                     'Nueva Era', 'Poblacion', 'San Andres', 'San Marcos', 'San Teodoro'],
        'Total_Households': [1526, 2825, 429, 1789, 810, 505, 1236, 1202, 274, 1996],
        'Benefited_At_Least_One_LMI': [24, 101, 23, 82, 9, 180, 31, 12, 17, 98],
        'Micro_Enterprise': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Employment_Facilitation_SLP': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'Integrated_Livelihood_DOLE': [0, 1, 1, 2, 3, 0, 2, 2, 1, 2],
        'Cash_For_Work': [8, 68, 21, 65, 2, 179, 13, 5, 14, 69],
        'Food_For_Work': [0, 2, 0, 1, 0, 1, 0, 0, 1, 2],
        'Community_Based_Employment': [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        'DOLE_Tupad': [15, 29, 2, 23, 3, 13, 16, 5, 1, 39],
    })

    agri_fisheries_df = pd.DataFrame({
        'Barangay': ['Bunawan Brook', 'Consuelo', 'Imelda', 'Libertad', 'Mambalili',
                     'Nueva Era', 'Poblacion', 'San Andres', 'San Marcos', 'San Teodoro'],
        'Total_Agri_Households': [262, 509, 123, 310, 68, 240, 602, 272, 91, 176],
        'Benefited_At_Least_One_AF': [76, 112, 71, 82, 6, 99, 241, 37, 27, 31],
        'Production_Support_Services': [72, 109, 69, 80, 6, 98, 240, 26, 21, 28],
        'Production_PostProduction_Irrigation': [0, 4, 5, 1, 2, 0, 1, 1, 1, 2],
        'Capacity_Development': [2, 3, 3, 0, 1, 1, 1, 4, 1, 0],
        'Cash': [6, 19, 8, 6, 0, 2, 13, 9, 9, 4],
        'Other_Agri_Fisheries': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    })

    bayanihan_df = pd.DataFrame({
        'Barangay': ['Bunawan Brook', 'Consuelo', 'Imelda', 'Libertad', 'Mambalili',
                     'Nueva Era', 'Poblacion', 'San Andres', 'San Marcos', 'San Teodoro'],
        'Total_Households': [1526, 2825, 429, 1789, 810, 505, 1236, 1202, 274, 1996],
        'Benefited_Bayanihan': [542, 1047, 115, 309, 244, 185, 208, 144, 147, 643],
        'SAP_DSWD': [302, 198, 75, 95, 172, 45, 146, 31, 92, 132],
        'DOLE_CAMP': [6, 16, 2, 10, 3, 20, 5, 6, 3, 21],
        'DOLE_AKAP': [6, 7, 2, 11, 3, 1, 4, 4, 2, 7],
        'DTI_Livelihood_Seeding': [4, 19, 2, 2, 3, 3, 2, 1, 2, 4],
        'DA_RFFA_FSRF': [44, 46, 30, 29, 37, 95, 49, 16, 9, 10],
        'DSWD_COVID19_Relief': [336, 917, 6, 203, 65, 134, 7, 87, 61, 547],
        'Relief_Other_Than_Gov': [158, 354, 34, 84, 2, 18, 8, 23, 59, 21],
        'Bayanihan2_Health_Workers': [1, 7, 0, 1, 1, 1, 0, 2, 0, 9],
        'Bayanihan2_ARBS': [0, 4, 0, 2, 0, 0, 1, 2, 0, 4],
        'Bayanihan2_Students': [0, 6, 2, 7, 1, 0, 1, 0, 1, 6],
        'Bayanihan2_Teaching_NonTeaching': [3, 3, 3, 18, 1, 0, 1, 0, 0, 13],
    })

    return {
        'social_assistance': social_assistance_df,
        'feeding_programs': feeding_programs_df,
        'labor_market': labor_market_df,
        'agri_fisheries': agri_fisheries_df,
        'bayanihan': bayanihan_df,
    }


# ============================================================
# COLUMN ALIASES
# ============================================================
COLUMN_ALIASES = {
    'year': 'year',
    'age': 'age',
    'gender': 'gender',
    'occupation': 'occupation',
    'education level': 'education_level',
    'civil status': 'civil_status',
    'household size': 'household_size',
    'barangay': 'barangay',
    'lndigenous group/tribe': 'tribe',
    'indigenous group/tribe': 'tribe',
    'barangay/community location': 'community_location',
    'household income': 'household_income',
    'employment status': 'employment_status',
    'access to transportation': 'access_transportation',
    'access to electricity': 'access_electricity',
    'access to internet': 'access_internet',
    'type of service': 'service_type',
    'access to status frequency  of service access': 'service_frequency',
    'access status frequency  of service access': 'service_frequency',
    'time spent accessing  the service': 'time_spent',
    'time spent accessing the service': 'time_spent',
    'mode of access': 'mode_of_access',
    'financial constraints': 'financial_barrier',
    'distance barrier': 'distance_barrier',
    'language barrier': 'language_barrier',
    'lack of information': 'info_barrier',
}


def _normalize_columns(df):
    df = df.copy()
    df.columns = [safe_str(c).replace('\ufeff', '').lower() for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


def _find_household_file(search_paths):
    patterns = ['dataset indigeous', 'dataset indigenous', 'indigeous', 'indigenous']
    for path in search_paths:
        if not os.path.isdir(path):
            continue
        all_files = glob.glob(os.path.join(path, '*.xlsx')) + \
                    glob.glob(os.path.join(path, '*.xls'))
        for f in all_files:
            fname = os.path.basename(f).lower()
            for p in patterns:
                if p in fname:
                    return f
    return None


@st.cache_data(show_spinner=True)
def load_household_data():
    search_paths = ['.', '/content/data']
    target = _find_household_file(search_paths)

    if target is None:
        return None, "File 'dataset indigeous.xlsx' not found in the repo root."

    try:
        xls = pd.ExcelFile(target)
        sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        df = _normalize_columns(df)
        df = df.dropna(how='all')
        return df, os.path.basename(target)
    except Exception as e:
        return None, f"Could not read {target}: {e}"


# ============================================================
# PREPROCESSING HELPERS
# ============================================================
def _clean_age(val):
    if val is None:
        return np.nan
    try:
        if pd.isna(val):
            return np.nan
    except (TypeError, ValueError):
        pass
    s = safe_str(val).replace('/', ' ')
    for token in s.split():
        if token.isdigit() and 0 < int(token) < 120:
            return int(token)
    return np.nan


def _clean_income(val):
    if val is None:
        return np.nan
    try:
        if pd.isna(val):
            return np.nan
    except (TypeError, ValueError):
        pass
    import re
    s = safe_str(val).lower().replace(',', '')
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m:
        return float(m.group(1))
    return np.nan


def _clean_frequency(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = safe_str(val).lower()
    if s in ('', 'nan', 'none'):
        return None
    if s in ('no access',):
        return 'None'
    if 'full' in s:
        return 'Full'
    if 'limited' in s or 'basic' in s or 'unimproved' in s:
        return 'Limited'
    if s == 'access':
        return 'Limited'
    return None


def _clean_time_spent(val):
    if val is None:
        return np.nan
    try:
        if pd.isna(val):
            return np.nan
    except (TypeError, ValueError):
        pass
    s = safe_str(val).lower().replace('-', ' ')
    nums = []
    for token in s.split():
        try:
            nums.append(float(token))
        except ValueError:
            continue
    if not nums:
        return np.nan
    avg = float(np.mean(nums))
    if 'day' in s:
        return avg * 24
    return avg


# ============================================================
# PREPROCESS
# ============================================================
@st.cache_data(show_spinner=True)
def preprocess(df):
    df = df.copy()

    if 'age' in df.columns:
        df['age'] = df['age'].apply(_clean_age)
    if 'household_income' in df.columns:
        df['income_weekly'] = df['household_income'].apply(_clean_income)
    if 'time_spent' in df.columns:
        df['time_spent_hours'] = df['time_spent'].apply(_clean_time_spent)
    if 'household_size' in df.columns:
        df['household_size'] = pd.to_numeric(df['household_size'], errors='coerce')

    if 'service_frequency' in df.columns:
        df['service_frequency_clean'] = df['service_frequency'].apply(_clean_frequency)

    # Force all categorical columns to clean strings
    categorical_cols = [
        'gender', 'barangay', 'tribe', 'employment_status',
        'education_level', 'civil_status', 'access_transportation',
        'access_electricity', 'access_internet', 'service_type',
        'mode_of_access', 'financial_barrier', 'distance_barrier',
        'language_barrier', 'info_barrier', 'service_frequency_clean'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: safe_str(x).title() if not is_blank(x) else None
            )

    if 'barangay' in df.columns:
        df['barangay'] = df['barangay'].apply(
            lambda x: 'Bunawan Brook' if safe_str(x) == 'Bunawan Brok' else x
        )
        df['barangay'] = df['barangay'].apply(
            lambda x: 'Imelda' if safe_str(x).lower() in ('lmelda', 'imelda') else x
        )

    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(
            lambda x: 'Female' if safe_str(x) == 'Famale' else x
        )

    if 'barangay' in df.columns:
        df = df[df['barangay'].apply(lambda x: not is_blank(x))]

    return df


# ============================================================
# LOAD
# ============================================================
df_raw, load_msg = load_household_data()

st.markdown('<div class="main-header">Indigenous Service Access Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analysis of Service Access Patterns Among Indigenous Communities in Bunawan, Agusan del Sur<br>'
            '<i>Using K-Means Clustering and Apriori Association Rule Mining</i></div>',
            unsafe_allow_html=True)

if df_raw is None:
    st.error(f"ERROR: {load_msg}")
    st.markdown("### Files needed in your repo root")
    st.code("app.py\nrequirements.txt\ndataset indigeous.xlsx")
    st.stop()

df = preprocess(df_raw)

with st.expander("Data Summary", expanded=False):
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
st.sidebar.header("Analysis Controls")

# Safe barangay filter
if 'barangay' in df.columns:
    barangay_options = sorted(df['barangay'].dropna().unique().tolist())
else:
    barangay_options = []

barangay_filter = st.sidebar.multiselect(
    "Filter by Barangay:",
    options=barangay_options,
    default=barangay_options
)

# Safe tribe filter
if 'tribe' in df.columns:
    tribe_options = sorted(df['tribe'].dropna().unique().tolist())
else:
    tribe_options = []

tribe_filter = st.sidebar.multiselect(
    "Filter by Tribe:",
    options=tribe_options,
    default=tribe_options
)

k_clusters = st.sidebar.slider("K-Means: number of clusters", 2, 6, 3)
min_support = st.sidebar.slider("Apriori: min support", 0.05, 0.50, 0.15, 0.05)
min_confidence = st.sidebar.slider("Apriori: min confidence", 0.3, 1.0, 0.6, 0.05)

# Apply filters safely
df_f = df.copy()
if barangay_filter and 'barangay' in df_f.columns:
    df_f = df_f[df_f['barangay'].isin(barangay_filter)]
if tribe_filter and 'tribe' in df_f.columns:
    df_f = df_f[df_f['tribe'].isin(tribe_filter)]

if len(df_f) < 20:
    st.warning("Too few records after filtering. Adjust the sidebar filters.")
    st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "EDA",
    "K-Means Clustering",
    "Apriori Rules",
    "Recommendations"
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("Executive Overview")

    n_respondents = len(df_f)
    n_barangays = df_f['barangay'].nunique() if 'barangay' in df_f.columns else 0
    n_tribes = df_f['tribe'].nunique() if 'tribe' in df_f.columns else 0
    n_services = df_f['service_type'].nunique() if 'service_type' in df_f.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Respondents</p><h2>{n_respondents:,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Barangays</p><h2>{n_barangays}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Tribes</p><h2>{n_tribes}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Service Types</p><h2>{n_services}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Respondents per Barangay")
    if 'barangay' in df_f.columns and n_barangays > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_f['barangay'].value_counts().plot(kind='bar', color='#1f4e79', edgecolor='black', ax=ax)
        ax.set_ylabel("Respondents")
        ax.set_title("Respondents per Barangay")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Indigenous Group Distribution")
    if 'tribe' in df_f.columns and n_tribes > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = df_f['tribe'].value_counts()
            ax.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
            ax.set_title("Tribe Distribution")
            st.pyplot(fig)
            plt.close()
        with col2:
            st.dataframe(
                df_f['tribe'].value_counts().rename('Count').to_frame().assign(
                    Percentage=lambda d: (d['Count'] / d['Count'].sum() * 100).round(2)
                ),
                use_container_width=True
            )


# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.header("Exploratory Data Analysis")

    st.subheader("Socio-Demographic Profile")

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
            vc = df_f['gender'].dropna().value_counts()
            if len(vc) > 0:
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
            vc = df_f['education_level'].dropna().value_counts().head(8)
            if len(vc) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                vc.sort_values().plot(kind='barh', color='#3498db', edgecolor='black', ax=ax)
                ax.set_xlabel("Respondents")
                ax.set_title("Education Level")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    with col4:
        if 'employment_status' in df_f.columns:
            vc = df_f['employment_status'].dropna().value_counts().head(8)
            if len(vc) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                vc.sort_values().plot(kind='barh', color='#e67e22', edgecolor='black', ax=ax)
                ax.set_xlabel("Respondents")
                ax.set_title("Employment Status")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown("---")
    st.subheader("Household Income")
    if 'income_weekly' in df_f.columns:
        inc = df_f['income_weekly'].dropna()
        if len(inc) > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(inc, bins=15, color='#9b59b6', edgecolor='black')
                ax.set_xlabel("Weekly Income (PHP)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Weekly Household Income (mean = PHP {inc.mean():.0f})")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                st.markdown("**Income statistics:**")
                st.dataframe(inc.describe().round(2).rename('Income (PHP/week)').to_frame(),
                             use_container_width=True)

    st.markdown("---")
    st.subheader("Barangay-level Service Access (CBMS Institutional)")
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
    st.subheader("Barriers to Service Access")

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
    st.header("K-Means Clustering: Service Access Patterns")
    st.caption("Groups households by similar socio-economic profile and access characteristics.")

    feature_frame = pd.DataFrame(index=df_f.index)

    if 'age' in df_f.columns:
        feature_frame['age'] = df_f['age']
    if 'household_size' in df_f.columns:
        feature_frame['household_size'] = df_f['household_size']
    if 'income_weekly' in df_f.columns:
        feature_frame['income_weekly'] = df_f['income_weekly']
    if 'time_spent_hours' in df_f.columns:
        feature_frame['time_spent_hours'] = df_f['time_spent_hours']

    cat_features = ['gender', 'education_level', 'employment_status',
                    'access_transportation', 'access_electricity', 'access_internet',
                    'service_frequency_clean', 'civil_status']

    for col in cat_features:
        if col in df_f.columns:
            le = LabelEncoder()
            vals = df_f[col].apply(safe_str)
            feature_frame[col + '_enc'] = le.fit_transform(vals)

    feature_frame = feature_frame.dropna()

    if len(feature_frame) < k_clusters * 2:
        st.warning("Not enough complete records for clustering after filtering.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_frame)

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
            f"Interpretation: Silhouette closer to +1.0 means better-separated clusters. "
            f"Score of {sil:.4f} suggests "
            f"{'excellent' if sil > 0.7 else 'acceptable' if sil > 0.5 else 'weak' if sil > 0.25 else 'poor'} "
            f"cluster quality. Lower DBI (closer to 0) means better separation."
        )

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

        st.markdown("---")
        st.subheader("Elbow Method (WCSS)")
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
        ax.set_title("Elbow Method: Within-Cluster Sum of Squares")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Cluster Profiles")
        cluster_df = feature_frame.copy()
        cluster_df['_cluster'] = labels
        cluster_df['_barangay'] = df_f.loc[feature_frame.index, 'barangay'].apply(safe_str).values \
            if 'barangay' in df_f.columns else 'Unknown'
        if 'tribe' in df_f.columns:
            cluster_df['_tribe'] = df_f.loc[feature_frame.index, 'tribe'].apply(safe_str).values

        for c in sorted(cluster_df['_cluster'].unique()):
            sub = cluster_df[cluster_df['_cluster'] == c]
            st.markdown(f"### Cluster {c} - {len(sub)} respondents ({len(sub)/len(cluster_df)*100:.1f}%)")

            m1, m2, m3 = st.columns(3)
            if 'age' in sub.columns:
                m1.metric("Mean Age", f"{sub['age'].mean():.1f}")
            if 'income_weekly' in sub.columns:
                m2.metric("Mean Weekly Income", f"PHP {sub['income_weekly'].mean():.0f}")
            if 'household_size' in sub.columns:
                m3.metric("Mean HH Size", f"{sub['household_size'].mean():.1f}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Barangays in this Cluster:**")
                st.dataframe(sub['_barangay'].value_counts().head(5).rename('Count'),
                             use_container_width=True)
            with col2:
                if '_tribe' in sub.columns and sub['_tribe'].notna().any():
                    st.markdown("**Top Tribes in this Cluster:**")
                    st.dataframe(sub['_tribe'].value_counts().head(5).rename('Count'),
                                 use_container_width=True)

            st.markdown("---")

        st.markdown("### Cluster Interpretation")
        st.caption("Based on Chapter 3: Cluster 1 = High Access, Cluster 2 = Moderate, Cluster 3 = Low")
        for c in sorted(cluster_df['_cluster'].unique()):
            sub = cluster_df[cluster_df['_cluster'] == c]
            income_mean = sub['income_weekly'].mean() if 'income_weekly' in sub.columns else 0
            if pd.isna(income_mean):
                income_mean = 0
            if income_mean > 1000:
                label = "High Access (Higher income, more resources)"
            elif income_mean > 700:
                label = "Moderate Access"
            else:
                label = "Low Access (Lower income, more barriers)"
            st.markdown(f"- Cluster {c} -> **{label}** (mean weekly income = PHP {income_mean:.0f})")


# ------------------------------------------------------------
# TAB 4: APRIORI
# ------------------------------------------------------------
with tab4:
    st.header("Apriori Association Rule Mining")
    st.caption("Discovers patterns like: if respondent is in tribe X and has barrier Y, they access service Z.")

    txn_cols = ['barangay', 'tribe', 'service_type', 'service_frequency_clean',
                'mode_of_access', 'financial_barrier', 'distance_barrier',
                'language_barrier', 'info_barrier', 'employment_status']

    available_txn = [c for c in txn_cols if c in df_f.columns]

    if not available_txn:
        st.warning("No suitable columns found for transaction encoding.")
    else:
        # Build transactions safely
        transactions = []
        for idx, row in df_f[available_txn].iterrows():
            txn = []
            for col in available_txn:
                val = row[col]
                val_str = safe_str(val)
                if val_str == '' or val_str.lower() in ('nan', 'none', 'nat', 'null', '<na>'):
                    continue
                txn.append(f"{col}={val_str}")
            if len(txn) >= 2:
                transactions.append(txn)

        if len(transactions) < 20:
            st.warning(f"Not enough transactions after filtering. Only {len(transactions)} valid rows.")
        else:
            try:
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                basket = pd.DataFrame(te_ary, columns=te.columns_)
            except Exception as e:
                st.error(f"Transaction encoding failed: {e}")
                st.stop()

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

                st.subheader("Top 15 Frequent Itemsets")
                top_itemsets = frequent.sort_values('support', ascending=False).head(15).copy()
                top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(
                    lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                )
                st.dataframe(top_itemsets, use_container_width=True)

                try:
                    rules = association_rules(frequent, metric='confidence', min_threshold=min_confidence)
                    rules = rules.sort_values('lift', ascending=False)
                except Exception as e:
                    rules = pd.DataFrame()
                    st.warning(f"Rule generation error: {e}")

                if not rules.empty:
                    st.markdown("---")
                    st.subheader(f"Top 15 Association Rules (out of {len(rules)})")

                    display = rules.head(15).copy()
                    display['antecedents'] = display['antecedents'].apply(
                        lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                    )
                    display['consequents'] = display['consequents'].apply(
                        lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                    )

                    st.dataframe(
                        display[['antecedents', 'consequents', 'support',
                                 'confidence', 'lift']].round(4),
                        use_container_width=True
                    )

                    st.caption(
                        "Support = how often the rule appears. "
                        "Confidence = reliability. "
                        "Lift > 1 = positive association (better than random)."
                    )

                    dl_rules = rules.copy()
                    dl_rules['antecedents'] = dl_rules['antecedents'].apply(
                        lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                    )
                    dl_rules['consequents'] = dl_rules['consequents'].apply(
                        lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                    )
                    st.download_button(
                        "Download Association Rules CSV",
                        dl_rules.to_csv(index=False).encode('utf-8'),
                        "apriori_rules.csv", "text/csv"
                    )
                else:
                    st.info(f"No rules found with confidence >= {min_confidence}. Lower the threshold.")


# ------------------------------------------------------------
# TAB 5: RECOMMENDATIONS
# ------------------------------------------------------------
with tab5:
    st.header("Recommendations and Interpretation")

    total_respondents = len(df_f)
    total_barangays = df_f['barangay'].nunique() if 'barangay' in df_f.columns else 0

    st.markdown("### Analysis Summary")
    st.markdown(
        f"Based on **{total_respondents:,}** indigenous respondents across "
        f"**{total_barangays}** barangays in Bunawan, Agusan del Sur, using "
        f"**K-Means Clustering** and **Apriori Association Rule Mining**."
    )

    st.markdown("#### 1. For the Local Government Unit (LGU) of Bunawan")
    st.markdown(
        "- **Prioritize low-access clusters** identified in the K-Means tab. "
        "They have the lowest income and most barriers to accessing government services.\n"
        "- **Deploy mobile service units** to remote barangays where distance and "
        "transportation barriers are most frequent.\n"
        "- **Allocate social assistance budgets** proportionally to high-need clusters, "
        "using CBMS data as evidence."
    )

    st.markdown("#### 2. For the National Commission on Indigenous Peoples (NCIP)")
    st.markdown(
        "- Use cluster profiles to design **culturally appropriate interventions** that "
        "respect the distinct needs of Manobo, Banwaon, and Talaandig households.\n"
        "- Align program delivery with the **Indigenous Peoples Rights Act (IPRA)** and "
        "Free, Prior, and Informed Consent (FPIC) principles."
    )

    st.markdown("#### 3. For Government Agencies (DSWD, DOLE, DA, DTI)")
    st.markdown(
        "- **Coordinate** program delivery to avoid duplicating benefits in high-access "
        "clusters while leaving low-access clusters underserved.\n"
        "- **Translate program materials** into local languages to reduce language barriers."
    )

    st.markdown("#### 4. For Indigenous Communities")
    st.markdown(
        "- **Engage** with barangay officials for a more transparent picture of available "
        "programs and how to access them.\n"
        "- **Participate in community consultations** that shape locally responsive policies."
    )

    st.markdown("#### 5. For Future Researchers")
    st.markdown(
        "- **Extend** with Random Forest, XGBoost, or Neural Networks for predictive modeling.\n"
        "- **Add geospatial analysis** (GIS) to visualize service access across barangays.\n"
        "- **Longitudinal studies** to track changes in service access over time (2022-2026).\n"
        "- **Qualitative interviews** to complement the quantitative clusters with lived "
        "experiences of the indigenous community."
    )

    st.markdown("#### 6. Limitations")
    st.markdown(
        "- Only **reported** respondents are analyzed. The actual population is larger.\n"
        "- Cluster results depend on the **22 variables** collected; missing variables "
        "may hide barriers.\n"
        "- Cross-sectional design limits causal inference.\n"
        "- Findings specific to Bunawan; may not generalize to other municipalities."
    )

    st.markdown("---")
    st.subheader("Download Results")
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
            "Download Summary CSV",
            summary.to_csv(index=False).encode('utf-8'),
            "indigenous_access_summary.csv",
            "text/csv"
        )
    except Exception as e:
        st.warning(f"Could not generate summary CSV: {e}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Capstone Dashboard | Analysis of Service Access Patterns Among Indigenous Communities "
    "in Bunawan, Agusan del Sur Using Data Mining | "
    "Falcasantos, K.M.P. and Adrales, K.J. | Agusan del Sur State University | 2026"
)
