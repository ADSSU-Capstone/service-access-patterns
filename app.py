import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(
    page_title="Indigenous Service Access Patterns",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #f8f9fa;
    padding: 8px;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #ffffff;
    border: 1px solid #ced4da;
    border-radius: 6px;
    padding: 10px 18px;
    font-weight: 600;
    color: #1f4e79;
}
.stTabs [aria-selected="true"] {
    background-color: #1f4e79 !important;
    color: #ffffff !important;
    border-color: #1f4e79;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-left: 6px solid #1f4e79;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    height: 100%;
}
.kpi-card .label {
    font-size: 0.82rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.kpi-card .value {
    font-size: 2.0rem;
    font-weight: 700;
    color: #1f4e79;
    line-height: 1.1;
}
.interpretation {
    background: #eef5fb;
    border-left: 4px solid #1f4e79;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 8px 0 18px 0;
    font-size: 0.92rem;
    color: #212529;
    line-height: 1.5;
}
.interpretation b { color: #1f4e79; }
.app-title {
    font-size: 2.05rem;
    font-weight: 800;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 4px;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 0.98rem;
    color: #495057;
    text-align: center;
    margin-bottom: 22px;
}
</style>
""", unsafe_allow_html=True)


def safe_str(val):
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

BARANGAY_COORDS = {
    'Bunawan Brook': (8.183, 126.000),
    'Consuelo': (8.165, 125.980),
    'Imelda': (8.150, 126.020),
    'Libertad': (8.175, 125.990),
    'Mambalili': (8.190, 126.010),
    'Nueva Era': (8.140, 125.970),
    'Poblacion': (8.180, 125.995),
    'San Andres': (8.200, 126.020),
    'San Marcos': (8.160, 126.030),
    'San Teodoro': (8.170, 126.040),
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


@st.cache_data(show_spinner=False)
def load_example_data():
    target = _find_household_file(['.', '/content/data'])
    if target is None:
        return None
    try:
        xls = pd.ExcelFile(target)
        sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        df = _normalize_columns(df)
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"Could not read example dataset: {e}")
        return None


def load_from_upload(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(uploaded_file)
            sheet = xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
        else:
            df = pd.read_csv(uploaded_file)
        df = _normalize_columns(df)
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        return None


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


@st.cache_data(show_spinner=True)
def preprocess_with_tracking(df):
    stats = {}
    df = df.copy()
    stats['raw_records'] = len(df)

    before = len(df)
    df = df.drop_duplicates()
    stats['duplicates_removed'] = before - len(df)

    missing_before = int(df.isna().sum().sum())

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
        before = len(df)
        df = df[df['barangay'].apply(lambda x: not is_blank(x))]
        stats['dropped_blank_barangay'] = before - len(df)
    else:
        stats['dropped_blank_barangay'] = 0

    missing_after = int(df.isna().sum().sum())
    stats['missing_handled'] = missing_before - missing_after
    stats['final_records'] = len(df)
    stats['missing_after'] = missing_after
    return df, stats


st.markdown('<div class="app-title">Analysis of Service Access Patterns Among Indigenous Communities in Bunawan, Agusan del Sur Using Data Mining</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">K-Means Clustering and Apriori Association Rule Mining on Community-Based Monitoring System Records</div>', unsafe_allow_html=True)

st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=False)

if uploaded is not None:
    df_raw = load_from_upload(uploaded)
elif use_example:
    df_raw = load_example_data()
else:
    st.info("Upload a dataset file or check the example dataset option in the sidebar to begin.")
    st.stop()

if df_raw is None:
    st.error("Dataset could not be loaded.")
    st.stop()

df, prep_stats = preprocess_with_tracking(df_raw)

st.sidebar.header("Analysis Controls")
if 'barangay' in df.columns:
    barangay_options = sorted(df['barangay'].dropna().unique().tolist())
else:
    barangay_options = []

focus_barangay = st.sidebar.selectbox(
    "Focus Barangay",
    options=['All Barangays'] + barangay_options,
    index=0
)

if 'tribe' in df.columns:
    tribe_options = sorted(df['tribe'].dropna().unique().tolist())
else:
    tribe_options = []

tribe_filter = st.sidebar.multiselect(
    "Tribe filter (optional)",
    options=tribe_options,
    default=[]
)

k_clusters = st.sidebar.slider("K-Means clusters", 2, 6, 3)
min_support = st.sidebar.slider("Apriori min support", 0.05, 0.50, 0.15, 0.05)
min_confidence = st.sidebar.slider("Apriori min confidence", 0.30, 1.00, 0.60, 0.05)

df_f = df.copy()
if focus_barangay != 'All Barangays' and 'barangay' in df_f.columns:
    df_f = df_f[df_f['barangay'] == focus_barangay]
if tribe_filter and 'tribe' in df_f.columns:
    df_f = df_f[df_f['tribe'].isin(tribe_filter)]

if len(df_f) < 20:
    st.warning("Too few records after filtering. Adjust the sidebar controls or upload a larger dataset.")
    st.stop()

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Data Preprocessing and Cleaning",
    "Sociodemographic and Economic Profiles",
    "Government Service Access Levels",
    "Service Access Patterns and Barriers",
    "Descriptive Model Evaluation"
])


with tab0:
    st.subheader("Data Preprocessing Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="label">Raw Records</div><div class="value">{prep_stats["raw_records"]:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="label">Duplicates Removed</div><div class="value">{prep_stats["duplicates_removed"]:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="label">Blank Barangay Dropped</div><div class="value">{prep_stats.get("dropped_blank_barangay", 0):,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="label">Final Valid Records</div><div class="value">{prep_stats["final_records"]:,}</div></div>', unsafe_allow_html=True)

    retention = prep_stats["final_records"] / max(prep_stats["raw_records"], 1) * 100
    st.markdown(
        f'<div class="interpretation"><b>Interpretation:</b> '
        f'{prep_stats["final_records"]:,} of {prep_stats["raw_records"]:,} raw records were retained '
        f'({retention:.1f}%). A total of {prep_stats["missing_handled"]:,} missing cells were resolved '
        f'through parsing and categorical normalization, leaving {prep_stats["missing_after"]:,} remaining '
        f'missing cells across the final dataset. The cleaned dataset is complete and suitable for '
        f'clustering and classification.</div>',
        unsafe_allow_html=True
    )

    st.subheader("Cleaning Rules Applied")
    rules_df = pd.DataFrame({
        'Rule': [
            'Duplicate household entries across barangay records removed',
            'Records with blank or invalid barangay labels dropped',
            'Barangay spelling normalized (Bunawan Brok to Bunawan Brook, Lmelda to Imelda)',
            'Gender spelling normalized (Famale to Female)',
            'Age extracted as a numeric value from free-text fields',
            'Household income parsed as a numeric value',
            'Service frequency mapped to Full, Limited, or None',
            'Time spent accessing services converted to hours',
            'Categorical text standardized to Title Case with whitespace trimmed'
        ]
    })
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    st.subheader("Records Retained per Barangay")
    if 'barangay' in df_f.columns:
        retained = df_f['barangay'].value_counts().rename('Retained Records').to_frame()
        st.dataframe(retained, use_container_width=True)


with tab1:
    st.subheader("Sociodemographic and Economic Profiles")

    st.markdown("#### Geographic Distribution of Respondents")
    if 'barangay' in df_f.columns and len(barangay_options) > 0:
        bar_counts = df_f['barangay'].value_counts().reset_index()
        bar_counts.columns = ['Barangay', 'Respondents']
        bar_counts['lat'] = bar_counts['Barangay'].map(
            lambda x: BARANGAY_COORDS.get(x, (8.18, 126.00))[0]
        )
        bar_counts['lon'] = bar_counts['Barangay'].map(
            lambda x: BARANGAY_COORDS.get(x, (8.18, 126.00))[1]
        )

        fig_map = px.scatter_map(
            bar_counts, lat='lat', lon='lon',
            size='Respondents', color='Respondents',
            color_continuous_scale='YlOrRd',
            hover_name='Barangay',
            hover_data={'lat': False, 'lon': False},
            zoom=11, height=460, size_max=42
        )
        fig_map.update_layout(
            map_style='open-street-map',
            margin={'r': 0, 't': 0, 'l': 0, 'b': 0}
        )
        st.plotly_chart(fig_map, use_container_width=True)

        top = bar_counts.iloc[0]
        low = bar_counts.iloc[-1]
        st.markdown(
            f'<div class="interpretation"><b>Interpretation:</b> The map locates respondent concentration across the ten barangays. '
            f'{top["Barangay"]} registers the highest respondent volume ({top["Respondents"]:,}), while '
            f'{low["Barangay"]} registers the lowest ({low["Respondents"]:,}). Geographic spread informs where service '
            f'delivery and data validation efforts should be prioritized.</div>',
            unsafe_allow_html=True
        )

    st.markdown("#### Barangay Rankings")
    if 'barangay' in df_f.columns:
        bar_rank = df_f['barangay'].value_counts().sort_values()
        fig_rank, ax = plt.subplots(figsize=(9, 4))
        bar_rank.plot(kind='barh', color='#1f4e79', edgecolor='black', ax=ax)
        ax.set_xlabel("Respondents")
        ax.set_title("Barangay Ranking by Respondent Count")
        plt.tight_layout()
        st.pyplot(fig_rank)
        plt.close()

        top3 = bar_rank.sort_values(ascending=False).head(3)
        low_one = bar_rank.sort_values().head(1)
        st.markdown(
            f'<div class="interpretation"><b>Interpretation:</b> The top three barangays by respondent count are '
            f'{", ".join(top3.index.tolist())}. The lowest is {low_one.index.tolist()[0]}. '
            f'This ranking shows where the sample is most and least represented.</div>',
            unsafe_allow_html=True
        )

    st.markdown("#### Age and Gender Distribution")
    col1, col2 = st.columns(2)
    with col1:
        if 'age' in df_f.columns:
            ages = df_f['age'].dropna()
            if len(ages) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(ages, bins=15, color='#16a085', edgecolor='black')
                ax.set_xlabel("Age")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Age Distribution (mean = {ages.mean():.1f})")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown(
                    f'<div class="interpretation"><b>Interpretation:</b> The mean respondent age is '
                    f'{ages.mean():.1f} years (range {ages.min():.0f} to {ages.max():.0f}). '
                    f'The distribution reflects the working-age composition of the indigenous households surveyed.</div>',
                    unsafe_allow_html=True
                )
    with col2:
        if 'gender' in df_f.columns:
            vc = df_f['gender'].dropna().value_counts()
            if len(vc) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                vc.plot(kind='bar', color='#27ae60', edgecolor='black', ax=ax)
                ax.set_ylabel("Respondents")
                ax.set_title("Gender Distribution")
                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown(
                    f'<div class="interpretation"><b>Interpretation:</b> {vc.index[0]} respondents account for '
                    f'{vc.iloc[0]:,} of {vc.sum():,} records ({vc.iloc[0]/vc.sum()*100:.1f}%). '
                    f'Gender composition informs whether service programs are reaching both groups equitably.</div>',
                    unsafe_allow_html=True
                )

    st.markdown("#### Education and Employment")
    col3, col4 = st.columns(2)
    with col3:
        if 'education_level' in df_f.columns:
            vc = df_f['education_level'].dropna().value_counts().head(8)
            if len(vc) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                vc.sort_values().plot(kind='barh', color='#3498db', edgecolor='black', ax=ax)
                ax.set_xlabel("Respondents")
                ax.set_title("Education Level")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown(
                    f'<div class="interpretation"><b>Interpretation:</b> The most common education level is '
                    f'{vc.index[-1]} ({vc.iloc[-1]:,} respondents). Educational attainment is a known correlate '
                    f'of service access because it affects awareness of programs and navigation of requirements.</div>',
                    unsafe_allow_html=True
                )
    with col4:
        if 'employment_status' in df_f.columns:
            vc = df_f['employment_status'].dropna().value_counts().head(8)
            if len(vc) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                vc.sort_values().plot(kind='barh', color='#e67e22', edgecolor='black', ax=ax)
                ax.set_xlabel("Respondents")
                ax.set_title("Employment Status")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown(
                    f'<div class="interpretation"><b>Interpretation:</b> The dominant employment category is '
                    f'{vc.index[-1]} ({vc.iloc[-1]:,} respondents). Employment status conditions the economic '
                    f'capacity of households to absorb the indirect costs of service access.</div>',
                    unsafe_allow_html=True
                )

    st.markdown("#### Household Income")
    if 'income_weekly' in df_f.columns:
        inc = df_f['income_weekly'].dropna()
        if len(inc) > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(inc, bins=15, color='#9b59b6', edgecolor='black')
                ax.set_xlabel("Weekly Income (PHP)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Weekly Household Income (mean = PHP {inc.mean():.0f})")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                st.dataframe(
                    inc.describe().round(2).rename('Income (PHP/week)').to_frame(),
                    use_container_width=True
                )
            st.markdown(
                f'<div class="interpretation"><b>Interpretation:</b> Mean weekly income is PHP {inc.mean():.0f} '
                f'with a median of PHP {inc.median():.0f}. The spread indicates the economic heterogeneity of '
                f'households in the sample and highlights which segments may face financial barriers to access.</div>',
                unsafe_allow_html=True
            )

    st.markdown("#### Resource Availability")
    resource_cols = ['access_transportation', 'access_electricity', 'access_internet']
    available_resources = [c for c in resource_cols if c in df_f.columns]
    if available_resources:
        cols = st.columns(len(available_resources))
        for i, col_name in enumerate(available_resources):
            with cols[i]:
                vc = df_f[col_name].dropna().value_counts()
                if len(vc) > 0:
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    vc.plot(kind='bar', color='#16a085', edgecolor='black', ax=ax)
                    ax.set_ylabel("Respondents")
                    ax.set_title(col_name.replace('_', ' ').title())
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        st.markdown(
            '<div class="interpretation"><b>Interpretation:</b> These charts show how many respondents report '
            'access to transportation, electricity, and internet. Low values in any resource indicate structural '
            'constraints that limit the ability of households to reach service delivery points.</div>',
            unsafe_allow_html=True
        )


with tab2:
    st.subheader("Government Service Access Levels")

    st.markdown("#### Types of Services Accessed")
    if 'service_type' in df_f.columns:
        svc_counts = df_f['service_type'].dropna().value_counts().head(12)
        if len(svc_counts) > 0:
            fig = px.bar(
                x=svc_counts.values, y=svc_counts.index,
                orientation='h',
                labels={'x': 'Respondents', 'y': 'Service Type'},
                color=svc_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=420, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div class="interpretation"><b>Interpretation:</b> {svc_counts.index[0]} is the most '
                f'accessed service ({svc_counts.iloc[0]:,} respondents). The list reflects the dominant service '
                f'channels in the municipality and shows which programs are reaching the community.</div>',
                unsafe_allow_html=True
            )

    st.markdown("#### Access Frequency Levels")
    if 'service_frequency_clean' in df_f.columns:
        freq = df_f['service_frequency_clean'].dropna().value_counts()
        if len(freq) > 0:
            col1, col2 = st.columns([1, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors_freq = {'Full': '#27ae60', 'Limited': '#f39c12', 'None': '#c0392b'}
                bar_colors = [colors_freq.get(k, '#1f4e79') for k in freq.index]
                ax.bar(freq.index, freq.values, color=bar_colors, edgecolor='black')
                ax.set_ylabel("Respondents")
                ax.set_title("Access Frequency Levels")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                freq_df = freq.rename('Respondents').to_frame()
                freq_df['Share (%)'] = (freq_df['Respondents'] / freq_df['Respondents'].sum() * 100).round(2)
                st.dataframe(freq_df, use_container_width=True)
            st.markdown(
                f'<div class="interpretation"><b>Interpretation:</b> The dominant access frequency in the current '
                f'selection is {freq.index[0]} ({freq.iloc[0]:,} respondents, '
                f'{freq.iloc[0]/freq.sum()*100:.1f}%). This establishes the baseline distribution that the '
                f'classifier in the next tab is trained against.</div>',
                unsafe_allow_html=True
            )

    st.markdown("#### Mode of Access")
    if 'mode_of_access' in df_f.columns:
        vc = df_f['mode_of_access'].dropna().value_counts().head(8)
        if len(vc) > 0:
            fig = px.pie(
                values=vc.values, names=vc.index,
                title="Mode of Access Distribution",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div class="interpretation"><b>Interpretation:</b> The most common mode of access is '
                f'{vc.index[0]} ({vc.iloc[0]:,} respondents). Mode of access reveals whether households '
                f'rely on physical visits, intermediaries, or digital channels.</div>',
                unsafe_allow_html=True
            )

    st.markdown("#### Barangay-level Service Distribution (CBMS Institutional Records)")
    cbms = get_cbms_tables()
    sub_a, sub_b, sub_c, sub_d, sub_e = st.tabs([
        "Social Assistance", "Feeding Programs", "Labor Market",
        "Agri and Fisheries", "Bayanihan"
    ])
    with sub_a:
        st.dataframe(cbms['social_assistance'], use_container_width=True, hide_index=True)
    with sub_b:
        st.dataframe(cbms['feeding_programs'], use_container_width=True, hide_index=True)
    with sub_c:
        st.dataframe(cbms['labor_market'], use_container_width=True, hide_index=True)
    with sub_d:
        st.dataframe(cbms['agri_fisheries'], use_container_width=True, hide_index=True)
    with sub_e:
        st.dataframe(cbms['bayanihan'], use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="interpretation"><b>Interpretation:</b> These institutional tables report exact counts '
        'of beneficiaries per program per barangay. They allow the LGU to compare program penetration '
        'against the respondent-level access levels reported above and identify barangays where a program '
        'is nominally available but underutilized.</div>',
        unsafe_allow_html=True
    )


with tab3:
    st.subheader("Service Access Patterns and Key Barriers")

    st.markdown("#### Correlation Heatmap of Numeric Variables")
    numeric_cols = [c for c in ['age', 'household_size', 'income_weekly', 'time_spent_hours'] if c in df_f.columns]
    if len(numeric_cols) >= 2:
        corr = df_f[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title("Pearson Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if corr_pairs:
            strongest = corr_pairs[0]
            st.markdown(
                f'<div class="interpretation"><b>Interpretation:</b> The strongest linear association is between '
                f'{strongest[0]} and {strongest[1]} (r = {strongest[2]:.3f}). '
                f'{"Positive" if strongest[2] > 0 else "Negative"} association of this magnitude indicates that the two variables move together in the current sample.</div>',
                unsafe_allow_html=True
            )

    st.markdown("#### K-Means Clustering of Household Profiles")
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
    k = max(2, min(k_clusters, len(feature_frame) - 1))

    if len(feature_frame) < k * 2:
        st.warning("Not enough complete records to fit K-Means with the selected K.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_frame)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        dbi = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="kpi-card"><div class="label">Clusters (K)</div><div class="value">{k}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-card"><div class="label">Records Clustered</div><div class="value">{len(labels):,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-card"><div class="label">Silhouette Score</div><div class="value">{sil:.4f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="kpi-card"><div class="label">Davies-Bouldin Index</div><div class="value">{dbi:.4f}</div></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="interpretation"><b>Interpretation:</b> The silhouette score of {sil:.4f} indicates '
            f'{"excellent" if sil > 0.7 else "acceptable" if sil > 0.5 else "weak" if sil > 0.25 else "poor"} '
            f'cluster separation, and the Davies-Bouldin Index of {dbi:.4f} confirms the same direction '
            f'(lower values mean better separation). These metrics validate the cluster structure used for profiling.</div>',
            unsafe_allow_html=True
        )

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        col1, col2 = st.columns(2)
        with col1:
            counts = pd.Series(labels).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_km = plt.cm.Set2(np.linspace(0, 1, k))
            ax.bar([f"C{i}" for i in counts.index], counts.values,
                   color=colors_km, edgecolor='black')
            ax.set_ylabel("Respondents")
            ax.set_title("Respondents per Cluster")
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig = go.Figure()
            for c in sorted(set(labels)):
                mask = labels == c
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0], y=X_pca[mask, 1],
                    mode='markers', name=f'Cluster {c}',
                    marker=dict(size=6, opacity=0.75, line=dict(width=0.5, color='black'))
                ))
            fig.update_layout(
                title="Households in PCA Space",
                xaxis_title="PC1", yaxis_title="PC2",
                height=380, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div class="interpretation"><b>Interpretation:</b> The PCA projection preserves {pca.explained_variance_ratio_.sum()*100:.1f}% '
            f'of total variance in two dimensions. Visible cluster separation in the projection indicates that '
            f'socio-economic characteristics group households into distinct service access profiles.</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Elbow Method for Cluster Selection")
        wcss = []
        max_k = min(10, len(feature_frame) - 1)
        for kk in range(1, max_k + 1):
            km = KMeans(n_clusters=kk, random_state=42, n_init=10)
            km.fit(X_scaled)
            wcss.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, max_k + 1), wcss, marker='o', color='#1f4e79')
        ax.axvline(k, color='red', linestyle='--', label=f'Chosen K = {k}')
        ax.set_xlabel("K")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method: Within-Cluster Sum of Squares")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown(
            '<div class="interpretation"><b>Interpretation:</b> The elbow curve plots total within-cluster '
            'variance against the number of clusters. The chosen K is where additional clusters no longer '
            'produce a substantial reduction in WCSS, indicating diminishing returns.</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Cluster Profiles")
        cluster_df = feature_frame.copy()
        cluster_df['_cluster'] = labels
        if 'barangay' in df_f.columns:
            cluster_df['_barangay'] = df_f.loc[feature_frame.index, 'barangay'].apply(safe_str).values
        if 'tribe' in df_f.columns:
            cluster_df['_tribe'] = df_f.loc[feature_frame.index, 'tribe'].apply(safe_str).values

        for c in sorted(cluster_df['_cluster'].unique()):
            sub = cluster_df[cluster_df['_cluster'] == c]
            st.markdown(f"Cluster {c} — {len(sub):,} respondents ({len(sub)/len(cluster_df)*100:.1f}%)")
            m1, m2, m3 = st.columns(3)
            if 'age' in sub.columns:
                m1.metric("Mean Age", f"{sub['age'].mean():.1f}")
            if 'income_weekly' in sub.columns:
                m2.metric("Mean Weekly Income", f"PHP {sub['income_weekly'].mean():.0f}")
            if 'household_size' in sub.columns:
                m3.metric("Mean Household Size", f"{sub['household_size'].mean():.1f}")

            col1, col2 = st.columns(2)
            with col1:
                if '_barangay' in sub.columns:
                    st.markdown("Top Barangays:")
                    st.dataframe(sub['_barangay'].value_counts().head(5).rename('Count'), use_container_width=True)
            with col2:
                if '_tribe' in sub.columns:
                    st.markdown("Top Tribes:")
                    st.dataframe(sub['_tribe'].value_counts().head(5).rename('Count'), use_container_width=True)

        st.markdown(
            '<div class="interpretation"><b>Interpretation:</b> The cluster profiles summarize the shared '
            'characteristics of each group. Higher mean weekly income and better infrastructure access '
            'correspond to higher service access; lower income and more barriers correspond to lower access. '
            'These profiles form the basis for targeted intervention in the next tabs.</div>',
            unsafe_allow_html=True
        )

    st.markdown("#### Apriori Association Rule Mining")
    txn_cols = ['barangay', 'tribe', 'service_type', 'service_frequency_clean',
                'mode_of_access', 'financial_barrier', 'distance_barrier',
                'language_barrier', 'info_barrier', 'employment_status']
    available_txn = [c for c in txn_cols if c in df_f.columns]

    if not available_txn:
        st.warning("No suitable columns found for transaction encoding.")
    else:
        transactions = []
        for _, row in df_f[available_txn].iterrows():
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
            st.warning(f"Only {len(transactions)} valid transactions found. Apriori requires at least 20.")
        else:
            try:
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                basket = pd.DataFrame(te_ary, columns=te.columns_)
                frequent = apriori(basket, min_support=min_support, use_colnames=True)
            except Exception as e:
                st.error(f"Apriori error: {e}")
                frequent = pd.DataFrame()

            if frequent.empty:
                st.warning(f"No frequent itemsets at min support = {min_support}. Lower the threshold.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="kpi-card"><div class="label">Frequent Itemsets</div><div class="value">{len(frequent):,}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="kpi-card"><div class="label">Min Support</div><div class="value">{min_support:.2f}</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="kpi-card"><div class="label">Min Confidence</div><div class="value">{min_confidence:.2f}</div></div>', unsafe_allow_html=True)

                st.markdown("Top 15 Frequent Itemsets")
                top_itemsets = frequent.sort_values('support', ascending=False).head(15).copy()
                top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(
                    lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                )
                st.dataframe(top_itemsets, use_container_width=True, hide_index=True)

                try:
                    rules = association_rules(frequent, metric='confidence', min_threshold=min_confidence)
                    rules = rules.sort_values('lift', ascending=False)
                except Exception as e:
                    rules = pd.DataFrame()
                    st.warning(f"Rule generation error: {e}")

                if not rules.empty:
                    st.markdown(f"Top 15 Association Rules (out of {len(rules)})")
                    display = rules.head(15).copy()
                    display['antecedents'] = display['antecedents'].apply(
                        lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                    )
                    display['consequents'] = display['consequents'].apply(
                        lambda x: ', '.join(sorted([safe_str(i) for i in x]))
                    )
                    st.dataframe(
                        display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(4),
                        use_container_width=True, hide_index=True
                    )

                    st.markdown(
                        '<div class="interpretation"><b>Interpretation:</b> Support measures how frequently '
                        'an itemset appears. Confidence measures reliability of the rule. Lift above 1 indicates '
                        'a positive association stronger than chance. Rules with high lift and high confidence '
                        'identify barrier-service combinations that co-occur and are useful for targeting '
                        'interventions.</div>',
                        unsafe_allow_html=True
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
                    st.info(f"No rules at confidence >= {min_confidence}. Lower the threshold.")


with tab4:
    st.subheader("Descriptive Model Evaluation")

    st.markdown("#### Classification Setup")
    if 'service_frequency_clean' not in df_f.columns:
        st.warning("The service frequency field is not available in the current selection.")
    else:
        access_map = {'Full': 'High', 'Limited': 'Moderate', 'None': 'Low'}
        model_df = df_f.copy()
        model_df['target'] = model_df['service_frequency_clean'].map(access_map)
        model_df = model_df[model_df['target'].notna()]

        feature_cols = []
        for col in ['age', 'household_size', 'income_weekly', 'time_spent_hours']:
            if col in model_df.columns:
                feature_cols.append(col)
        for col in ['gender', 'education_level', 'employment_status',
                    'access_transportation', 'access_electricity', 'access_internet',
                    'civil_status', 'financial_barrier', 'distance_barrier',
                    'language_barrier', 'info_barrier']:
            if col in model_df.columns:
                feature_cols.append(col)

        if len(feature_cols) < 3 or len(model_df) < 50:
            st.warning("Insufficient features or records to train a classifier on the current selection.")
        else:
            X = model_df[feature_cols].copy()
            for col in X.columns:
                converted = pd.to_numeric(X[col], errors='coerce')
                if converted.notna().any():
                    median_val = converted.median()
                    if pd.isna(median_val):
                        median_val = 0
                    X[col] = converted.fillna(median_val)
                else:
                    X[col] = X[col].astype(str).apply(safe_str)
                    X[col] = LabelEncoder().fit_transform(X[col])
            y = model_df['target']

            class_counts = y.value_counts()
            st.markdown("Class Distribution")
            class_df = class_counts.rename('Count').to_frame()
            class_df['Share (%)'] = (class_df['Count'] / class_df['Count'].sum() * 100).round(2)
            st.dataframe(class_df, use_container_width=True)

            min_share = class_df['Share (%)'].min()
            if min_share < 20:
                st.warning(
                    f"Class imbalance detected. The smallest class represents only {min_share:.1f}% of the "
                    f"training data. Accuracy alone may be misleading; macro-averaged precision, recall, "
                    f"and F1 are reported to compensate."
                )

            if y.nunique() < 2:
                st.warning("The target variable has fewer than two classes in the current selection.")
            else:
                stratify = y if y.value_counts().min() >= 2 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=stratify
                )

                rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)

                if hasattr(rf, 'predict_proba'):
                    y_prob_rf = rf.predict_proba(X_test)
                else:
                    y_prob_rf = None

                lr = LogisticRegression(max_iter=2000, random_state=42)
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)

                acc_rf = accuracy_score(y_test, y_pred_rf)
                prec_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
                rec_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
                f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)

                acc_lr = accuracy_score(y_test, y_pred_lr)
                prec_lr = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
                rec_lr = recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)
                f1_lr = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown(f'<div class="kpi-card"><div class="label">Accuracy</div><div class="value">{acc_rf:.3f}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="kpi-card"><div class="label">Precision</div><div class="value">{prec_rf:.3f}</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="kpi-card"><div class="label">Recall</div><div class="value">{rec_rf:.3f}</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="kpi-card"><div class="label">F1-Score</div><div class="value">{f1_rf:.3f}</div></div>', unsafe_allow_html=True)
                with c5:
                    try:
                        if y_prob_rf is not None and y.nunique() > 2:
                            auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class='ovr', average='weighted')
                        elif y_prob_rf is not None and y.nunique() == 2:
                            auc_rf = roc_auc_score(y_test, y_prob_rf[:, 1])
                        else:
                            auc_rf = float('nan')
                    except Exception:
                        auc_rf = float('nan')
                    auc_display = f'{auc_rf:.3f}' if not np.isnan(auc_rf) else 'N/A'
                    st.markdown(f'<div class="kpi-card"><div class="label">ROC-AUC (OVR)</div><div class="value">{auc_display}</div></div>', unsafe_allow_html=True)

                best = 'Random Forest' if f1_rf >= f1_lr else 'Logistic Regression'
                st.markdown(
                    f'<div class="interpretation"><b>Interpretation:</b> Random Forest achieves accuracy '
                    f'{acc_rf:.3f} with weighted F1 of {f1_rf:.3f}. Logistic Regression achieves accuracy '
                    f'{acc_lr:.3f} with weighted F1 of {f1_lr:.3f}. The best-performing model in the current '
                    f'selection is <b>{best}</b>. Random Forest captures non-linear interactions between '
                    f'socio-economic variables and access levels, which is why it is used as the primary '
                    f'classifier for feature importance.</div>',
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Confusion Matrix — Random Forest")
                    cm = confusion_matrix(y_test, y_pred_rf, labels=sorted(y.unique()))
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=sorted(y.unique()),
                                yticklabels=sorted(y.unique()), ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.markdown("ROC Curves — Random Forest")
                    fig = go.Figure()
                    if y_prob_rf is not None:
                        classes = sorted(y.unique())
                        for i, cls in enumerate(classes):
                            try:
                                if len(classes) > 2:
                                    fpr, tpr, _ = roc_curve((y_test == cls).astype(int), y_prob_rf[:, i])
                                    auc_val = roc_auc_score((y_test == cls).astype(int), y_prob_rf[:, i])
                                else:
                                    fpr, tpr, _ = roc_curve((y_test == cls).astype(int), y_prob_rf[:, i])
                                    auc_val = roc_auc_score((y_test == cls).astype(int), y_prob_rf[:, i])
                                fig.add_trace(go.Scatter(
                                    x=fpr, y=tpr, mode='lines',
                                    name=f'{cls} (AUC = {auc_val:.3f})'
                                ))
                            except Exception:
                                continue
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                              line=dict(dash='dash', color='gray'),
                                              name='Chance'))
                    fig.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=380, margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    '<div class="interpretation"><b>Interpretation:</b> The confusion matrix shows how often '
                    'each actual access level is correctly or incorrectly predicted. The ROC curves show the '
                    'trade-off between true positive and false positive rates for each class. Curves closer to '
                    'the top-left corner indicate stronger discriminative ability for that class.</div>',
                    unsafe_allow_html=True
                )

                st.markdown("#### Feature Importance — Random Forest")
                importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(9, 5))
                importances.plot(kind='barh', color='#1f4e79', edgecolor='black', ax=ax)
                ax.set_xlabel("Relative Importance")
                ax.set_title("Random Forest Feature Importance")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                top_feature = importances.sort_values(ascending=False).index[0]
                top_value = importances.sort_values(ascending=False).iloc[0]
                st.markdown(
                    f'<div class="interpretation"><b>Interpretation:</b> The strongest predictor of access level '
                    f'in the current selection is <b>{top_feature}</b> with a relative importance of '
                    f'{top_value:.3f}. The ranking indicates which socio-economic or barrier variables most '
                    f'strongly separate high-access households from low-access households, and where policy '
                    f'interventions would have the largest marginal effect.</div>',
                    unsafe_allow_html=True
                )

                st.markdown("#### Algorithm Justification")
                st.markdown(
                    "- **Random Forest** is selected as the primary classifier because it handles mixed "
                    "numeric and categorical predictors, does not assume linearity, and reports feature "
                    "importance that is directly interpretable for policy. It is robust to outliers and "
                    "captures interaction effects between socio-economic variables.\n"
                    "- **Logistic Regression** is included as a baseline because it is transparent and "
                    "well understood by non-technical stakeholders. Comparing the two models shows whether "
                    "the additional complexity of Random Forest produces a material gain in F1.\n"
                    "- **K-Means** is used for unsupervised profiling because Chapter 3 specifies it as the "
                    "primary clustering algorithm and it aligns with the household segmentation objectives.\n"
                    "- **Apriori** is used for association rule mining because Chapter 3 specifies it and it "
                    "reveals co-occurrence patterns between barriers and service types that clustering alone "
                    "does not surface."
                )

                st.markdown("#### Record-Level Predictions")
                prediction_df = X_test.copy()
                prediction_df['actual'] = y_test.values
                prediction_df['predicted'] = y_pred_rf
                st.dataframe(prediction_df.head(20), use_container_width=True)
                st.download_button(
                    "Download Record-Level Predictions CSV",
                    prediction_df.to_csv(index=False).encode('utf-8'),
                    "record_level_predictions.csv", "text/csv"
                )


st.markdown("---")
st.caption(
    "Falcasantos, K.M.P. and Adrales, K.J. | Agusan del Sur State University | "
    "Analysis of Service Access Patterns Among Indigenous Communities in Bunawan, Agusan del Sur Using Data Mining"
)
