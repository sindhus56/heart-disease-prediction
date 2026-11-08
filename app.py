"""
HeartPredict - AI-Powered Heart Disease Prediction System
Developed by Sindhu S

Streamlit front-end for the Heart Disease Prediction notebook.
The models are trained on start-up from values.csv + labels.csv using the same
preprocessing / split / hyper-parameters as the notebook, so every number shown
in the app matches the notebook.

Requires: streamlit >= 1.40
"""

import base64
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="HeartPredict | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BLUE = "#1f6feb"
DARK = "#0b3d91"
LIGHT = "#eaf3ff"
GREEN = "#1fa35c"
RED = "#d64545"
APP_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Column names / label maps (from the notebook's dataset)
# --------------------------------------------------------------------------
COL_SLOPE = "slope_of_peak_exercise_st_segment"
COL_THAL = "thal"
COL_BP = "resting_blood_pressure"
COL_CP = "chest_pain_type"
COL_VESSELS = "num_major_vessels"
COL_FBS = "fasting_blood_sugar_gt_120_mg_per_dl"
COL_EKG = "resting_ekg_results"
COL_CHOL = "serum_cholesterol_mg_per_dl"
COL_OLDPEAK = "oldpeak_eq_st_depression"
COL_SEX = "sex"
COL_AGE = "age"
COL_HR = "max_heart_rate_achieved"
COL_ANGINA = "exercise_induced_angina"
COL_TARGET = "heart_disease_present"

NUMERIC_COLS = [COL_AGE, COL_BP, COL_CHOL, COL_OLDPEAK, COL_HR]

PRETTY = {
    COL_SLOPE: "Slope of Peak Exercise ST",
    COL_THAL: "Thal",
    COL_BP: "Resting Blood Pressure",
    COL_CP: "Chest Pain Type",
    COL_VESSELS: "Number of Major Vessels",
    COL_FBS: "Fasting Blood Sugar > 120",
    COL_EKG: "Resting ECG (EKG) Result",
    COL_CHOL: "Serum Cholesterol",
    COL_OLDPEAK: "Oldpeak (ST Depression)",
    COL_SEX: "Sex",
    COL_AGE: "Age",
    COL_HR: "Max Heart Rate Achieved",
    COL_ANGINA: "Exercise Induced Angina",
    "thal_normal": "Thal: Normal",
    "thal_reversible_defect": "Thal: Reversible Defect",
}

SEX_MAP = {"Female": 0, "Male": 1}
CP_MAP = {
    "1 - Typical Angina": 1,
    "2 - Atypical Angina": 2,
    "3 - Non-anginal Pain": 3,
    "4 - Asymptomatic": 4,
}
FBS_MAP = {"No (≤ 120 mg/dL)": 0, "Yes (> 120 mg/dL)": 1}
EKG_MAP = {
    "0 - Normal": 0,
    "1 - ST-T Wave Abnormality": 1,
    "2 - Left Ventricular Hypertrophy": 2,
}
ANGINA_MAP = {"No": 0, "Yes": 1}
SLOPE_MAP = {"1 - Upsloping": 1, "2 - Flat": 2, "3 - Downsloping": 3}
VESSELS_MAP = {"0": 0, "1": 1, "2": 2, "3": 3}
THAL_MAP = {
    "Normal": "normal",
    "Fixed Defect": "fixed_defect",
    "Reversible Defect": "reversible_defect",
}
MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM",
    "Naive Bayes",
]

# --------------------------------------------------------------------------
# Inline SVG icons (consistent line icons, like the design)
# --------------------------------------------------------------------------
_HEART = "M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"
ICONS = {
    "heart": f'<path d="{_HEART}"/>',
    "heart-pulse": f'<path d="{_HEART}"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "bars": '<path d="M3 3v18h18"/><path d="M18 17V9M13 17V5M8 17v-3"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6M10 22h4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5M12 15V3"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6M18 9h1.5a2.5 2.5 0 0 0 0-5H18M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-1.97 1.5C7.85 18.75 7 20.24 7 22M14 14.66V17c0 .55.47.98 1.97 1.5C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "warning": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "x": '<path d="M18 6 6 18M6 6l12 12"/>',
    "clipboard": '<rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>',
    "file": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><path d="M14 2v6h6M9 15l2 2 4-4"/>',
}


def icon(name: str, size: int = 22, color: str = BLUE, stroke: float = 2) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0">{ICONS[name]}</svg>'
    )


def logo_mark(size: int = 38) -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;width:{size}px;'
        f"height:{size}px;border-radius:12px;background:linear-gradient(135deg,#3b8cff,#0b4fd1);"
        f'box-shadow:0 4px 12px rgba(31,111,235,.35)">{icon("heart-pulse", int(size*0.6), "#fff", 2.2)}</span>'
    )


def _hero_data_uri() -> str:
    """Hero background image (assets/hero.webp) embedded as a data-URI."""
    for name, mime in (("hero.webp", "image/webp"), ("hero.png", "image/png"), ("hero.jpg", "image/jpeg")):
        path = APP_DIR / "assets" / name
        if path.exists():
            return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return ""


HERO_URL = _hero_data_uri()
HERO_IMG_LAYER = f'url("{HERO_URL}") right center / auto 100% no-repeat,' if HERO_URL else ""

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
_BG_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='420' viewBox='0 0 1600 420'>"
    "<defs><linearGradient id='w' x1='0' y1='0' x2='1' y2='0'>"
    "<stop offset='0' stop-color='#a9d2ff' stop-opacity='.55'/><stop offset='1' stop-color='#d9ecff' stop-opacity='.1'/>"
    "</linearGradient></defs>"
    "<path d='M0 190 C260 120 420 280 760 200 C1100 120 1300 260 1600 170 L1600 0 L0 0 Z' fill='url(#w)'/>"
    "<polyline points='40,120 140,120 165,70 195,170 225,40 255,120 360,120 600,120 625,95 650,135 675,120 760,120' "
    "fill='none' stroke='#8ec2ff' stroke-opacity='.65' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>"
    "<polyline points='1020,90 1120,90 1140,60 1165,115 1190,90 1330,90 1350,70 1375,105 1400,90 1590,90' "
    "fill='none' stroke='#8ec2ff' stroke-opacity='.5' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>"
    "<g fill='#8ec2ff' fill-opacity='.45'><rect x='430' y='30' width='12' height='38' rx='3'/>"
    "<rect x='417' y='43' width='38' height='12' rx='3'/>"
    "<rect x='1250' y='160' width='10' height='30' rx='3'/><rect x='1240' y='170' width='30' height='10' rx='3'/></g>"
    "</svg>"
)
BG_URL = "data:image/svg+xml;utf8," + quote(_BG_SVG)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp button,
.stApp input, .stApp textarea, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stMarkdownContainer"] {{ font-family: 'Inter', sans-serif; }}
/* keep Streamlit's icon font intact (expander arrows etc.) */
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded, .stApp .material-icons {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}
.stApp {{
    background: url("{BG_URL}") no-repeat top center / 100% auto,
                linear-gradient(180deg, #d4e8ff 0%, #e8f3ff 30%, #f5faff 100%);
}}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container, [data-testid="stMainBlockContainer"] {{
    padding-top: 3rem !important; padding-bottom: 1rem !important; max-width: 1150px;
}}
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] {{ padding-bottom: 0 !important; }}

/* ---------- navbar ---------- */
.st-key-navbar {{
    background: rgba(255,255,255,.92); border: 1px solid #d6e6fb; border-radius: 18px;
    padding: .55rem 1.1rem; box-shadow: 0 6px 22px rgba(31,111,235,.12); margin-bottom: -.6rem;
    align-items: center;
}}
.st-key-navbar [data-testid="stHorizontalBlock"] {{ align-items: center; gap: .3rem; }}
.st-key-navbar .stButton > button {{
    border: none; background: transparent; color: {DARK}; font-weight: 600; font-size: .9rem;
    padding: .35rem .5rem; border-radius: 999px; box-shadow: none; min-height: 0; width: 100%;
}}
.st-key-navbar .stButton > button:hover {{ background: {LIGHT}; color: {BLUE}; }}
.st-key-navbar .stButton > button[kind="primary"],
.st-key-navbar .stButton > button[kind="primaryFormSubmit"] {{ background: {BLUE}; color: #fff; }}
.st-key-navbar .stButton > button[kind="primary"] p {{ color: #fff; }}
.logo-text {{
    font-family: 'Playfair Display', serif !important; font-weight: 800; font-size: 1.55rem;
    color: {DARK}; margin-left: .55rem; vertical-align: middle;
}}
.logo-text span {{ color: {BLUE}; }}

/* ---------- cards ---------- */
.card, div[class*="st-key-card"] {{
    background: rgba(255,255,255,.9); border: 1px solid #d6e6fb; border-radius: 18px;
    padding: 1.1rem 1.4rem; box-shadow: 0 6px 22px rgba(31,111,235,.10); margin-bottom: 1rem;
}}
.card h3 {{ font-family: 'Playfair Display', serif !important; color: {DARK}; margin: 0 0 .5rem; font-size: 1.25rem; }}
.card p, .card li {{ color: #31475f; line-height: 1.6; font-size: .93rem; }}
.card ul, .card ol {{ padding-left: 1.2rem; margin: 0; }}
div.st-key-card_hero[class*="st-key"] {{
    background: {HERO_IMG_LAYER}
                linear-gradient(90deg, #eef6ff 0%, #d5e9fd 55%, #c3e1fd 100%);
    border: none; border-radius: 24px; padding: 2.4rem 2.8rem; min-height: 420px;
    justify-content: center; overflow: hidden;
    box-shadow: 0 10px 30px rgba(31,111,235,.14);
}}
.hero-title {{
    font-family: 'Playfair Display', serif !important; font-size: 3.6rem; font-weight: 800;
    color: #0a2a66; line-height: 1.08; margin-bottom: 1rem;
}}
.hero-title span {{ color: {BLUE}; }}
.hero-text {{ color: #1d3557; font-size: 1.12rem; line-height: 1.65; margin-bottom: 1.4rem; }}
.hero-text b {{ color: {DARK}; font-size: 1.2rem; }}
.page-title {{
    font-family: 'Playfair Display', serif !important; color: {DARK}; font-size: 2rem; font-weight: 700;
    text-align: center; display: flex; gap: .6rem; justify-content: center; align-items: center;
}}
.page-sub {{ text-align: center; color: #5b7088; margin: .1rem 0 1.2rem; font-size: .92rem; }}
.card-title {{
    font-family: 'Playfair Display', serif !important; color: {DARK}; font-weight: 700; font-size: 1.1rem;
    display: flex; gap: .5rem; align-items: center; margin-bottom: .5rem;
}}
.section-head {{
    color: {DARK}; font-weight: 700; font-size: 1.02rem; display: flex; gap: .55rem; align-items: center;
    margin-bottom: .6rem;
}}
.section-head .ic {{
    width: 30px; height: 30px; border-radius: 999px; background: {LIGHT}; display: inline-flex;
    align-items: center; justify-content: center;
}}
.tile {{
    background: rgba(255,255,255,.92); border: 1px solid #d6e6fb; border-radius: 16px;
    padding: 1rem .6rem; text-align: center; box-shadow: 0 4px 14px rgba(31,111,235,.08); height: 100%;
}}
.tile b {{ color: {DARK}; display: block; margin-top: .45rem; font-size: .92rem; }}
.tile span {{ color: #5b7088; font-size: .8rem; }}
.stat {{
    background: rgba(255,255,255,.92); border: 1px solid #d6e6fb; border-radius: 16px;
    padding: .9rem 1rem; box-shadow: 0 4px 14px rgba(31,111,235,.08);
    display: flex; align-items: center; gap: .8rem; justify-content: center;
}}
.stat.col {{ flex-direction: column; gap: .15rem; }}
.stat .l {{ color: #5b7088; font-size: .82rem; font-weight: 500; }}
.stat .v {{ font-size: 1.35rem; font-weight: 700; color: {DARK}; }}
.stat.col .v {{ color: {BLUE}; font-size: 1.7rem; }}
.stat .ic {{
    width: 46px; height: 46px; border-radius: 14px; background: {LIGHT};
    display: flex; align-items: center; justify-content: center;
}}

/* result */
.result-banner {{
    display: flex; align-items: center; gap: 1.4rem; justify-content: center;
    border-radius: 22px; padding: 1.6rem 2rem; margin: 0 auto 1.2rem; max-width: 760px;
    box-shadow: 0 8px 26px rgba(31,111,235,.12); border: 1px solid #d6e6fb;
}}
.result-banner.neg {{ background: linear-gradient(135deg,#ffffff,#eafaf1); }}
.result-banner.pos {{ background: linear-gradient(135deg,#ffffff,#fdeeee); border-color:#f3c4c4; }}
.result-banner .badge {{
    width: 78px; height: 78px; border-radius: 999px; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}}
.result-banner h2 {{ font-family: 'Playfair Display', serif !important; margin: 0; font-size: 2rem; }}
.result-banner p {{ margin: .2rem 0 0; color: #31475f; }}
.card table.kv, .card table.kv th, .card table.kv td {{ border: none !important; background: transparent !important; }}
.kv {{ width: 100%; border-collapse: collapse; }}
[data-testid="stExpander"] {{
    background: rgba(255,255,255,.92); border: 1px solid #d6e6fb !important; border-radius: 14px;
    box-shadow: 0 4px 14px rgba(31,111,235,.08); margin-bottom: 1rem;
}}
[data-testid="stExpander"] summary p {{ font-weight: 600; color: {DARK}; }}
.kv td {{ padding: .38rem .2rem; color: #31475f; font-size: .92rem; border-bottom: 1px dashed #e0ecfb !important; }}
.kv td:first-child {{ width: 52%; font-weight: 500; }}
.kv td:nth-child(2) {{ width: 6%; color: #8aa0b8; }}
.kv td:last-child {{ color: {DARK}; font-weight: 600; }}
.eqcard {{ height: 320px; box-sizing: border-box; overflow: hidden; }}
.minis {{ display: flex; gap: .8rem; margin: .7rem 0 .4rem; }}
.mini {{ background: {LIGHT}; border-radius: 12px; padding: .45rem .9rem; font-size: .78rem; color: #5b7088; flex: 1; }}
.mini b {{ display: block; font-size: .95rem; color: {DARK}; margin-top: .1rem; }}
.probbar {{ height: 12px; border-radius: 999px; background: #e3eefc; overflow: hidden; margin: .5rem 0 .2rem; }}
.probbar div {{ height: 100%; border-radius: 999px; }}
.note {{
    background: #e8f1ff; border-radius: 14px; padding: .75rem 1.1rem; color: #31475f; font-size: .88rem;
    border: 1px solid #cfe1fb; display: flex; gap: .7rem; align-items: center; max-width: 760px; margin: .6rem auto 0;
}}
.best {{
    background: linear-gradient(90deg, #fff1dc, #fff8ec); border: 1px solid #f5d9a8; border-radius: 16px;
    padding: .8rem 1.3rem; color: #6b4a00; display: flex; gap: .9rem; align-items: center;
}}
.best b {{ color: #8a5a00; font-size: 1.02rem; }}
.takeaway {{ display: flex; gap: .55rem; align-items: flex-start; margin-bottom: .55rem; color: #31475f; font-size: .9rem; line-height: 1.45; }}
.steps li {{ margin-bottom: .55rem; list-style: none; display: flex; gap: .7rem; align-items: center; }}
.steps .n {{
    width: 28px; height: 28px; border-radius: 999px; border: 2px solid {BLUE}; color: {BLUE};
    display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; flex-shrink: 0;
}}
.grid {{ display: grid; gap: 1rem; margin-bottom: 1rem; }}
.grid.c4 {{ grid-template-columns: repeat(4, 1fr); }}
.grid.c3 {{ grid-template-columns: repeat(3, 1fr); }}
.grid .card {{ margin-bottom: 0; }}
.grid .card h4 {{ font-family: 'Playfair Display', serif !important; color: {DARK}; margin: 0 0 .4rem; font-size: 1rem; }}
.techs {{ display: flex; justify-content: space-around; flex-wrap: wrap; gap: .6rem; }}
.tech {{ text-align: center; font-size: .76rem; color: #31475f; width: 64px; }}
.tech .b {{
    width: 40px; height: 40px; border-radius: 12px; margin: 0 auto .25rem; display: flex; align-items: center;
    justify-content: center; font-weight: 800; color: #fff; font-size: 1rem;
}}
.footer-pill {{
    background: rgba(255,255,255,.9); border: 1px solid #d6e6fb; border-radius: 999px; padding: .7rem 1.6rem;
    text-align: center; color: {DARK}; font-weight: 600; width: fit-content; margin: 1.4rem auto .3rem;
    display: flex; gap: .6rem; align-items: center; box-shadow: 0 4px 14px rgba(31,111,235,.08);
}}
.tagline {{ text-align: center; color: {DARK}; font-family: 'Playfair Display', serif !important; font-style: italic; margin-bottom: 1rem; }}

/* ---------- form ---------- */
[data-testid="stForm"] {{ border: none; padding: 0; background: transparent; }}
[data-testid="stWidgetLabel"] p {{ font-size: .84rem; font-weight: 600; color: #1d3557; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"],
[data-testid="stNumberInputContainer"] {{
    background: #fff !important; border-radius: 10px !important;
}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] {{ border: 1px solid #d6e6fb !important; }}
div[data-baseweb="input"] input {{ background: #fff !important; }}
/* Dropdowns: force every layer of the select control to white (works across Streamlit versions) */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stSelectbox"] div[role="combobox"] * {{
    background-color: #ffffff !important; background-image: none !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    border: 1px solid #d6e6fb !important; border-radius: 10px !important;
}}
[data-testid="stSelectbox"] input {{ color: #1d3557 !important; }}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
    border-radius: 999px; font-weight: 600; padding: .55rem 1.7rem; border: 1.6px solid {BLUE};
    color: {BLUE}; background: #fff;
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
.stDownloadButton > button[kind="primary"], .stDownloadButton > button[kind="primaryDownload"] {{
    background: linear-gradient(90deg, #2f7bff, #1457d8) !important; color: #fff !important; border: none !important;
    box-shadow: 0 6px 16px rgba(31,111,235,.35);
}}
.stButton > button[kind="primary"] p, [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] p,
[data-testid="stFormSubmitButton"] > button[kind="primary"] p, .stDownloadButton > button p {{ color: inherit; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {DARK}; }}

@media (max-width: 800px) {{
    .grid.c4, .grid.c3 {{ grid-template-columns: 1fr; }}
    .eqcard {{ height: auto; overflow: visible; }}
    .hero-title {{ font-size: 2.2rem; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data + model (same steps as the notebook)
# --------------------------------------------------------------------------
def _find_file(name: str):
    for folder in (APP_DIR, APP_DIR / "data"):
        if (folder / name).exists():
            return folder / name
    return None


@st.cache_resource(show_spinner="Training models… (first load only)")
def load_everything():
    v_path, l_path = _find_file("values.csv"), _find_file("labels.csv")
    if v_path is None or l_path is None:
        return None

    values = pd.read_csv(v_path)
    labels = pd.read_csv(l_path)
    df = pd.merge(values, labels, on="patient_id")
    data = df.drop("patient_id", axis=1)

    X = data.drop(COL_TARGET, axis=1)
    y = data[COL_TARGET]
    X = pd.get_dummies(X, columns=[COL_THAL], drop_first=True)
    for c in ["thal_normal", "thal_reversible_defect"]:
        X[c] = X[c].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
        "Random Forest": (RandomForestClassifier(random_state=42), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True),
        "SVM": (SVC(random_state=42), True),
        "Naive Bayes": (GaussianNB(), True),
    }

    rows, cms = [], {}
    for name in MODEL_ORDER:
        model, scaled = models[name]
        model.fit(Xtr_s if scaled else X_train, y_train)
        pred = model.predict(Xte_s if scaled else X_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1 Score": f1_score(y_test, pred),
            }
        )
        cms[name] = confusion_matrix(y_test, pred)

    rf = models["Random Forest"][0]
    importance = (
        pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "df": df,
        "rf": rf,
        "columns": list(X.columns),
        "comparison": pd.DataFrame(rows),
        "cms": cms,
        "importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def go_to(page: str):
    st.session_state["page"] = page


def style_fig(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#31475f"),
    )
    fig.update_xaxes(gridcolor="#e6effb")
    fig.update_yaxes(gridcolor="#e6effb")
    return fig


def page_title(ic: str, title: str, sub: str):
    st.markdown(
        f'<div class="page-title">{icon(ic, 30)} {title}</div><div class="page-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def section_head(ic: str, title: str) -> str:
    return f'<div class="section-head"><span class="ic">{icon(ic, 17)}</span>{title}</div>'


def card_title(ic: str, title: str) -> str:
    return f'<div class="card-title">{icon(ic, 21)}{title}</div>'


def stat_row(ic: str, label: str, value: str) -> str:
    return (
        f'<div class="stat"><div class="ic">{icon(ic, 24)}</div>'
        f'<div><div class="l">{label}</div><div class="v">{value}</div></div></div>'
    )


def stat_col(ic: str, label: str, value: str) -> str:
    return (
        f'<div class="stat col">{icon(ic, 26)}<div class="v">{value}</div>'
        f'<div class="l">{label}</div></div>'
    )


def footer():
    st.markdown(
        '<div class="tagline" style="font-size:1.25rem;margin:.6rem 0 0;padding-bottom:.2rem">'
        'Your health Our priority <span style="color:#e0245e;font-style:normal">❤️</span></div>',
        unsafe_allow_html=True,
    )


def build_input_row(p: dict, columns: list) -> pd.DataFrame:
    row = {
        COL_SLOPE: p["slope"],
        COL_BP: p["bp"],
        COL_CP: p["cp"],
        COL_VESSELS: p["vessels"],
        COL_FBS: p["fbs"],
        COL_EKG: p["ekg"],
        COL_CHOL: p["chol"],
        COL_OLDPEAK: p["oldpeak"],
        COL_SEX: p["sex"],
        COL_AGE: p["age"],
        COL_HR: p["hr"],
        COL_ANGINA: p["angina"],
        "thal_normal": int(p["thal"] == "normal"),
        "thal_reversible_defect": int(p["thal"] == "reversible_defect"),
    }
    return pd.DataFrame([row])[columns]


def _pdf_safe(text) -> str:
    """The built-in PDF font is Latin-1 only; swap common symbols for plain equivalents."""
    text = str(text).replace("≤", "<=").replace("≥", ">=").replace("–", "-").replace("—", "-")
    return text.encode("latin-1", "replace").decode("latin-1")


def make_pdf(summary: dict, prob: float, pred: int) -> bytes:
    from fpdf import FPDF  # fpdf2

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(11, 61, 145)
    pdf.cell(0, 12, "HeartPredict - Prediction Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 7, f"Generated on {datetime.now():%d %b %Y, %H:%M}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    if pred == 1:
        pdf.set_text_color(214, 69, 69)
        pdf.cell(0, 9, "Result: Heart disease may be present", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_text_color(31, 163, 92)
        pdf.cell(0, 9, "Result: Heart disease not predicted", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(
        0, 8,
        f"Model: Random Forest | Estimated probability of heart disease: {prob*100:.1f}%",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(11, 61, 145)
    pdf.cell(0, 8, "Patient Input Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    for k, v in summary.items():
        pdf.cell(80, 7, _pdf_safe(k), border="B")
        pdf.cell(0, 7, _pdf_safe(v), border="B", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This prediction is generated by a machine learning model trained on a small "
        "academic dataset (180 records). It is for informational purposes only and is not a "
        "substitute for professional medical advice, diagnosis or treatment.",
    )
    return bytes(pdf.output())


# --------------------------------------------------------------------------
# Navbar
# --------------------------------------------------------------------------
PAGES = ["Home", "Prediction", "Performance", "Insights", "About"]
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
page = st.session_state["page"]

with st.container(key="navbar"):
    cols = st.columns([3.2, 0.9, 1.1, 1.3, 1.0, 0.9])
    cols[0].markdown(
        f'{logo_mark()}<span class="logo-text">Heart<span>Predict</span></span>',
        unsafe_allow_html=True,
    )
    for c, p in zip(cols[1:], PAGES):
        c.button(p, key=f"nav_{p}", type="primary" if p == page else "secondary",
                 on_click=go_to, args=(p,), use_container_width=True)

art = load_everything()
if art is None:
    st.error(
        "**values.csv** and **labels.csv** were not found. Place both files in the same folder "
        "as `app.py` (or in a `data/` sub-folder) and restart the app."
    )
    st.stop()

df = art["df"]
cmp_df = art["comparison"]
rf_row = cmp_df[cmp_df["Model"] == "Random Forest"].iloc[0]


# ==========================================================================
# 1. HOME
# ==========================================================================
def page_home():
    with st.container(key="card_hero"):
        left, _right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown(
                '<div class="hero-title">Heart Disease<br><span>Prediction</span></div>'
                '<div class="hero-text"><b>Take charge of your heart health!</b><br>'
                "Enter your heart details and get an<br>"
                "instant prediction using machine learning.</div>",
                unsafe_allow_html=True,
            )
            st.button("Get Started  →", type="primary", on_click=go_to, args=("Prediction",))

    tiles = [
        ("cpu", "ML-Powered Prediction", "Random Forest classifier"),
        ("shield", "Model Performance You Can Trust", f"{rf_row['Accuracy']*100:.1f}% test accuracy"),
        ("heart-pulse", "Early Detection for Better Care", "Flag higher-risk patients early"),
        ("bars", "Health Insights at Your Fingertips", "Explore the dataset"),
    ]
    html = '<div class="grid c4">' + "".join(
        f'<div class="tile">{icon(i, 32)}<b>{t}</b><span>{s}</span></div>' for i, t, s in tiles
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        '<div class="card" style="text-align:center"><h3>About HeartPredict</h3>'
        "<p>HeartPredict is a smart and reliable machine learning system that predicts heart disease "
        "from patient medical data — chest pain type, blood pressure, cholesterol, maximum heart rate, "
        "ST depression, thal result and more — using a <b>Random Forest</b> model selected after "
        "comparing six algorithms, helping you take proactive care of your heart.</p></div>",
        unsafe_allow_html=True,
    )


# ==========================================================================
# 2 + 3. PREDICTION
# ==========================================================================
FORM_KEYS = [
    "f_age", "f_sex", "f_cp", "f_bp", "f_chol", "f_fbs", "f_ekg",
    "f_hr", "f_angina", "f_oldpeak", "f_slope", "f_vessels", "f_thal",
]


SAMPLES = {
    "A": {  # high-risk example
        "f_age": 62, "f_sex": "Male", "f_cp": "4 - Asymptomatic", "f_bp": 150, "f_chol": 280,
        "f_fbs": "No (≤ 120 mg/dL)", "f_ekg": "2 - Left Ventricular Hypertrophy", "f_hr": 105,
        "f_angina": "Yes", "f_oldpeak": 3.0, "f_slope": "2 - Flat", "f_vessels": "2",
        "f_thal": "Reversible Defect",
    },
    "B": {  # low-risk example
        "f_age": 38, "f_sex": "Female", "f_cp": "2 - Atypical Angina", "f_bp": 115, "f_chol": 190,
        "f_fbs": "No (≤ 120 mg/dL)", "f_ekg": "0 - Normal", "f_hr": 175,
        "f_angina": "No", "f_oldpeak": 0.0, "f_slope": "1 - Upsloping", "f_vessels": "0",
        "f_thal": "Normal",
    },
}


def load_sample(which: str):
    """Callback: fill every form field with a saved sample patient."""
    st.session_state.update(SAMPLES[which])


def clear_form():
    for k in FORM_KEYS:
        st.session_state.pop(k, None)


def reset_prediction():
    st.session_state.pop("result", None)
    clear_form()


def page_prediction():
    if "result" in st.session_state:
        show_result(st.session_state["result"])
        return

    page_title("activity", "Patient Health Information",
               "Please fill in the details below to predict heart disease risk.")

    _, q1, q2, _ = st.columns([0.7, 1.6, 1.6, 0.7])
    q1.button("⚡ Sample A · High Risk", key="load_a", on_click=load_sample,
              args=("A",), use_container_width=True)
    q2.button("⚡ Sample B · Low Risk", key="load_b", on_click=load_sample,
              args=("B",), use_container_width=True)
    st.write("")

    with st.form("patient_form", border=False):
        with st.container(key="card_personal"):
            st.markdown(section_head("user", "Personal Information"), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            age = c1.number_input("Age *", 1, 120, value=None, placeholder="Enter age", key="f_age")
            sex = c2.selectbox("Sex *", list(SEX_MAP), index=None, placeholder="Select gender", key="f_sex")

        with st.container(key="card_heart"):
            st.markdown(section_head("heart", "Heart Examination Details"), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            cp = c1.selectbox("Chest Pain Type *", list(CP_MAP), index=None, placeholder="Select type", key="f_cp")
            bp = c2.number_input("Resting Blood Pressure *", 50, 250, value=None,
                                 placeholder="Enter value (mmHg)", key="f_bp")
            chol = c3.number_input("Cholesterol *", 50, 700, value=None,
                                   placeholder="Enter value (mg/dL)", key="f_chol")
            c1, c2, c3 = st.columns(3)
            fbs = c1.selectbox("Fasting Blood Sugar *", list(FBS_MAP), index=None,
                               placeholder="Select option", key="f_fbs")
            ekg = c2.selectbox("Resting ECG *", list(EKG_MAP), index=None,
                               placeholder="Select result", key="f_ekg")
            hr = c3.number_input("Maximum Heart Rate *", 50, 250, value=None,
                                 placeholder="Enter value (bpm)", key="f_hr")

        with st.container(key="card_exercise"):
            st.markdown(section_head("activity", "Exercise & ECG Details"), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            angina = c1.selectbox("Exercise Angina *", list(ANGINA_MAP), index=None,
                                  placeholder="Select option", key="f_angina")
            oldpeak = c2.number_input("Oldpeak *", 0.0, 10.0, value=None, step=0.1, format="%.1f",
                                      placeholder="Enter value", key="f_oldpeak")
            slope = c3.selectbox("Slope *", list(SLOPE_MAP), index=None,
                                 placeholder="Select slope", key="f_slope")
            c1, c2, _ = st.columns(3)
            vessels = c1.selectbox("Major Vessels *", list(VESSELS_MAP), index=None,
                                   placeholder="Select option", key="f_vessels")
            thal = c2.selectbox("Thal *", list(THAL_MAP), index=None,
                                placeholder="Select type", key="f_thal")

        _, b1, b2, _ = st.columns([1.3, 1.4, 1, 1.3])
        submitted = b1.form_submit_button("Predict Heart Disease  →", type="primary",
                                          use_container_width=True)
        b2.form_submit_button("⟳  Clear Form", on_click=clear_form, use_container_width=True)

    if submitted:
        required = [age, sex, cp, bp, chol, fbs, ekg, hr, angina, oldpeak, slope, vessels, thal]
        if any(v is None for v in required):
            st.warning("Please fill in all the fields marked with * before predicting.")
        else:
            params = {
                "age": int(age), "sex": SEX_MAP[sex], "cp": CP_MAP[cp], "bp": int(bp),
                "chol": int(chol), "fbs": FBS_MAP[fbs], "ekg": EKG_MAP[ekg], "hr": int(hr),
                "angina": ANGINA_MAP[angina], "oldpeak": float(oldpeak), "slope": SLOPE_MAP[slope],
                "vessels": VESSELS_MAP[vessels], "thal": THAL_MAP[thal],
            }
            x_row = build_input_row(params, art["columns"])
            pred = int(art["rf"].predict(x_row)[0])
            prob = float(art["rf"].predict_proba(x_row)[0][1])
            summary = {
                "Age": f"{age}",
                "Sex": sex,
                "Chest Pain Type": cp.split(" - ")[1],
                "Blood Pressure": f"{bp} mmHg",
                "Cholesterol": f"{chol} mg/dL",
                "Fasting Blood Sugar > 120": fbs,
                "Resting ECG": ekg.split(" - ")[1],
                "Max Heart Rate": f"{hr} bpm",
                "Exercise Angina": angina,
                "Oldpeak": f"{oldpeak:.1f}",
                "Slope": slope.split(" - ")[1],
                "Major Vessels": vessels,
                "Thal": thal,
            }
            st.session_state["result"] = {
                "pred": pred, "prob": prob, "summary": summary, "params": params,
            }
            st.rerun()

    footer()


def show_result(res: dict):
    pred, prob, summary, p = res["pred"], res["prob"], res["summary"], res["params"]

    page_title("file", "Prediction Result", "Random Forest model output for the entered patient data")

    if pred == 1:
        st.markdown(
            f'<div class="result-banner pos"><div class="badge" style="background:{RED}">'
            f'{icon("warning", 42, "#fff", 2.4)}</div><div>'
            f'<h2 style="color:{RED}">Heart Disease Predicted</h2>'
            f"<p>Based on the provided information, the model predicts that "
            f"<b>heart disease may be present</b>.</p></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-banner neg"><div class="badge" style="background:{GREEN}">'
            f'{icon("check", 44, "#fff", 3)}</div><div>'
            f'<h2 style="color:{GREEN}">Heart Disease Not Predicted</h2>'
            f"<p>Based on the provided information, the model predicts that "
            f"<b>heart disease is not present</b>.</p></div></div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns(2, gap="large")
    items = list(summary.items())
    with left:
        rows = "".join(f"<tr><td>{k}</td><td>:</td><td>{v}</td></tr>" for k, v in items[:5])
        st.markdown(
            f'<div class="card eqcard">{card_title("user", "Patient Input Summary")}'
            f'<table class="kv">{rows}</table>'
            f'<div style="color:#8aa0b8;font-size:.82rem;margin-top:.4rem">… and more</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("View all patient inputs"):
            allrows = "".join(f"<tr><td>{k}</td><td>:</td><td>{v}</td></tr>" for k, v in items)
            st.markdown(f'<table class="kv">{allrows}</table>', unsafe_allow_html=True)

    with right:
        color = RED if pred == 1 else GREEN
        st.markdown(
            f'<div class="card eqcard">{card_title("bulb", "Explanation")}'
            f"<p>The Random Forest model (test accuracy {rf_row['Accuracy']*100:.2f}%, recall "
            f"{rf_row['Recall']*100:.2f}%) analysed the key health indicators and estimated a "
            f"<b>{prob*100:.1f}%</b> probability of heart disease. Predictions above 50% are "
            f"classified as <i>heart disease present</i>.</p>"
            f'<div class="minis"><div class="mini">Prediction<b style="color:{color}">'
            f'{"Heart disease present" if pred == 1 else "Heart disease not present"}</b></div>'
            f'<div class="mini">Probability<b>{prob*100:.1f}%</b></div></div>'
            f'<div class="probbar"><div style="width:{prob*100:.1f}%;background:{color}"></div></div>'
            f'<div style="display:flex;justify-content:space-between;color:#8aa0b8;font-size:.78rem">'
            f"<span>0%</span><span>50%</span><span>100%</span></div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("Compare with dataset averages"):
            avg = df.groupby(COL_TARGET)[NUMERIC_COLS].mean()
            comp = pd.DataFrame(
                {
                    "Patient": [p["age"], p["bp"], p["chol"], p["oldpeak"], p["hr"]],
                    "Avg: No disease": avg.loc[0].values.round(1),
                    "Avg: Disease": avg.loc[1].values.round(1),
                },
                index=["Age", "Resting BP", "Cholesterol", "Oldpeak", "Max Heart Rate"],
            )
            st.dataframe(comp, use_container_width=True)

    if pred == 1:
        st.markdown(
            f'<div class="card">{card_title("heart-pulse", "Suggested Next Steps")}<ul>'
            "<li>Prioritise the patient for further medical evaluation.</li>"
            "<li>Arrange clinical assessment and tests as recommended by a qualified professional.</li>"
            "<li>Monitor blood pressure, cholesterol and heart rate regularly.</li>"
            "<li>Provide preventive-care and healthy-lifestyle guidance.</li></ul></div>",
            unsafe_allow_html=True,
        )

    _, b1, b2, _ = st.columns([1, 1.3, 1.3, 1])
    try:
        b1.download_button(
            "⬇  Download PDF Report", make_pdf(summary, prob, pred), "heartpredict_report.pdf",
            "application/pdf", type="primary", use_container_width=True,
        )
    except Exception:
        b1.caption("Install `fpdf2` to enable PDF export.")
    b2.button("Predict Another Patient  ⟳", on_click=reset_prediction, use_container_width=True)

    st.markdown(
        f'<div class="note">{icon("info", 24)}<span><b>Note:</b> This prediction is for informational '
        f"purposes only and is not a substitute for professional medical advice.</span></div>",
        unsafe_allow_html=True,
    )
    footer()


# ==========================================================================
# 4. PERFORMANCE
# ==========================================================================
def page_performance():
    page_title("bars", "Model Performance",
               f"Comparison of different ML algorithms · trained on {art['n_train']} records, "
               f"tested on {art['n_test']} unseen records")

    c = st.columns(4)
    for col, (ic, lab, key) in zip(
        c,
        [("target", "Accuracy", "Accuracy"), ("shield", "Precision", "Precision"),
         ("heart", "Recall", "Recall"), ("bars", "F1-Score", "F1 Score")],
    ):
        col.markdown(stat_col(ic, f"{lab} (Random Forest)", f"{rf_row[key]*100:.2f}%"),
                     unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        with st.container(key="card_algo"):
            h1, h2 = st.columns([1.4, 1])
            h1.markdown('<div class="card-title" style="margin-top:.4rem">Algorithm Comparison</div>',
                        unsafe_allow_html=True)
            metric = h2.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1 Score"],
                                  label_visibility="collapsed")
            colors = ["#0b5bd3" if m == "Random Forest" else "#4f9cff" for m in MODEL_ORDER]
            vals = [cmp_df.set_index("Model").loc[m, metric] for m in MODEL_ORDER]
            fig = go.Figure(go.Bar(
                x=MODEL_ORDER, y=vals, marker_color=colors,
                text=[f"{v:.2f}" for v in vals], textposition="outside",
            ))
            fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(style_fig(fig, 340), use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(key="card_cm"):
            chosen = st.selectbox("Confusion matrix for", MODEL_ORDER, index=2,
                                  label_visibility="collapsed")
            st.markdown(f'<div class="card-title">Confusion Matrix · {chosen}</div>', unsafe_allow_html=True)
            fig = px.imshow(
                art["cms"][chosen], text_auto=True, color_continuous_scale="Blues",
                x=["No Disease", "Disease"], y=["No Disease", "Disease"],
                labels=dict(x="Predicted", y="Actual"),
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(style_fig(fig, 300), use_container_width=True, config={"displayModeBar": False})

    b_l, b_r = st.columns([3, 1], gap="medium")
    b_l.markdown(
        f'<div class="best">{icon("trophy", 38, "#d99200")}<div><b>Best Model: Random Forest</b><br>'
        f'Joint-highest accuracy ({rf_row["Accuracy"]*100:.2f}%, tied with SVM and Naive Bayes) and '
        f'joint-highest recall ({rf_row["Recall"]*100:.2f}%, tied with Naive Bayes).</div></div>',
        unsafe_allow_html=True,
    )
    if "show_report" not in st.session_state:
        st.session_state["show_report"] = False
    b_r.button(
        "Hide Detailed Report ↑" if st.session_state["show_report"] else "View Detailed Report  →",
        type="primary", use_container_width=True,
        on_click=lambda: st.session_state.update(show_report=not st.session_state["show_report"]),
    )

    if st.session_state["show_report"]:
        with st.container(key="card_report"):
            st.markdown(card_title("clipboard", "Detailed Comparison"), unsafe_allow_html=True)
            st.dataframe(
                cmp_df.style.format({c: "{:.4f}" for c in cmp_df.columns if c != "Model"})
                .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#d6ebff"),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "With only 36 test records, one patient changes accuracy by about 2.8 percentage "
                "points, so small differences between models should be read with care."
            )
    footer()


# ==========================================================================
# 5. INSIGHTS
# ==========================================================================
def page_insights():
    page_title("pie", "Health Insights", "Explore key insights from our heart health dataset")

    c = st.columns(3)
    c[0].markdown(stat_row("database", "Dataset Size", f"{len(df)} Records"), unsafe_allow_html=True)
    c[1].markdown(stat_row("heart", "Heart Disease", f"{df[COL_TARGET].mean()*100:.1f}%"),
                  unsafe_allow_html=True)
    c[2].markdown(stat_row("users", "Avg Age", f"{df[COL_AGE].mean():.1f} Years"), unsafe_allow_html=True)
    st.write("")

    l, r = st.columns(2, gap="medium")
    with l:
        with st.container(key="card_age"):
            st.markdown('<div class="card-title">Age Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x=COL_AGE, nbins=12, color_discrete_sequence=["#4f9cff"],
                               labels={COL_AGE: "Age"})
            fig.update_traces(marker_line_color="#fff", marker_line_width=1.5)
            fig.update_layout(yaxis_title="Patients", bargap=0.05)
            st.plotly_chart(style_fig(fig, 280), use_container_width=True, config={"displayModeBar": False})
    with r:
        with st.container(key="card_chol"):
            st.markdown('<div class="card-title">Cholesterol Analysis</div>', unsafe_allow_html=True)
            band = pd.cut(df[COL_CHOL], [0, 199, 239, 10_000],
                          labels=["Low (<200)", "Medium (200–239)", "High (≥240)"])
            rate = df.groupby(band, observed=True)[COL_TARGET].mean() * 100
            counts = band.value_counts().reindex(rate.index)
            fig = go.Figure(go.Scatter(
                x=list(rate.index), y=rate.values, mode="lines+markers+text",
                text=[f"{v:.0f}%" for v in rate.values], textposition="top center",
                line=dict(color=BLUE, width=3), marker=dict(size=10, color=BLUE),
                customdata=counts.values, hovertemplate="%{x}<br>%{y:.1f}% (n=%{customdata})<extra></extra>",
            ))
            fig.update_yaxes(range=[0, 100], title="Heart disease rate (%)")
            st.plotly_chart(style_fig(fig, 280), use_container_width=True, config={"displayModeBar": False})

    l, r = st.columns([1.5, 1], gap="medium")
    with l:
        with st.container(key="card_fi"):
            st.markdown('<div class="card-title">Feature Importance (Top 10)</div>', unsafe_allow_html=True)
            top = art["importance"].head(10).iloc[::-1].copy()
            top["Feature"] = top["Feature"].map(PRETTY)
            fig = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"], orientation="h",
                                   marker_color=BLUE))
            st.plotly_chart(style_fig(fig, 360), use_container_width=True, config={"displayModeBar": False})
    with r:
        points = [
            "Patients with heart disease have a higher average ST depression (oldpeak) and a lower "
            "average maximum heart rate.",
            "Asymptomatic chest pain, reversible-defect thal, exercise-induced angina and more major "
            "vessels are more common among positive cases.",
            "Feature importance shows how the model behaves — it is not medical causation.",
            "ML can help flag patients for early evaluation.",
        ]
        items = "".join(f'<div class="takeaway">{icon("check", 20, GREEN, 3)}<span>{t}</span></div>' for t in points)
        st.markdown(f'<div class="card">{card_title("bulb", "Key Takeaways")}{items}</div>',
                    unsafe_allow_html=True)

    with st.expander("Explore more features by heart disease status"):
        plot = df.copy()
        plot["Status"] = plot[COL_TARGET].map({0: "No Heart Disease", 1: "Heart Disease"})
        cmap = {"No Heart Disease": "#9cc7ff", "Heart Disease": RED}
        t1, t2, t3 = st.tabs(["Numerical", "Categorical", "Correlation"])
        with t1:
            col = st.selectbox("Numerical feature", NUMERIC_COLS, format_func=lambda c: PRETTY[c])
            fig = px.box(plot, x="Status", y=col, color="Status", color_discrete_map=cmap, points="all")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 340), use_container_width=True, config={"displayModeBar": False})
        with t2:
            cats = [COL_THAL, COL_CP, COL_SLOPE, COL_VESSELS, COL_FBS, COL_EKG, COL_SEX, COL_ANGINA]
            col = st.selectbox("Categorical feature", cats, format_func=lambda c: PRETTY[c])
            ct = plot.groupby([col, "Status"]).size().reset_index(name="Patients")
            ct[col] = ct[col].astype(str)
            fig = px.bar(ct, x=col, y="Patients", color="Status", barmode="group", color_discrete_map=cmap)
            st.plotly_chart(style_fig(fig, 340), use_container_width=True, config={"displayModeBar": False})
        with t3:
            corr = df.select_dtypes("number").corr()
            corr.index = [PRETTY.get(i, i) for i in corr.index]
            corr.columns = [PRETTY.get(i, i) for i in corr.columns]
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(style_fig(fig, 520), use_container_width=True, config={"displayModeBar": False})
    footer()


# ==========================================================================
# 6. ABOUT
# ==========================================================================
def page_about():
    page_title("info", "About This Project", "Building a smarter future for heart care")

    l, r = st.columns(2, gap="medium")
    with l:
        st.markdown(
            f'<div class="card" style="min-height:250px;position:relative">'
            f'{card_title("clipboard", "Project Overview")}'
            "<p>The objective of this project is to analyse patient cardiovascular data and build "
            "machine learning models that predict the presence of heart disease. It covers data "
            "analysis, training and comparing multiple models, selecting the best model, giving "
            "suggestions to the hospital and documenting the challenges faced.</p>"
            "<p><b>Developed by:</b> Sindhu S</p>"
            f'<div style="position:absolute;right:18px;bottom:14px;opacity:.9">{icon("heart-pulse", 64, "#3b8cff", 1.6)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with r:
        steps = ["Load &amp; Explore Data", "Pre-process (encode, scale, 80:20 split)",
                 "Train 6 ML Models", "Evaluate Performance", "Make Predictions"]
        li = "".join(f'<li><span class="n">{i+1}</span>{s}</li>' for i, s in enumerate(steps))
        st.markdown(
            f'<div class="card" style="min-height:250px">{card_title("activity", "Machine Learning Workflow")}'
            f'<ul class="steps" style="padding:0">{li}</ul></div>',
            unsafe_allow_html=True,
        )

    l, r = st.columns(2, gap="medium")
    techs = [("Py", "Python", "#3776ab"), ("St", "Streamlit", "#ff4b4b"), ("sk", "scikit-learn", "#f7931e"),
             ("Pd", "Pandas", "#150458"), ("Np", "NumPy", "#4d77cf"), ("Pl", "Plotly", "#3f4f75")]
    tech_html = "".join(
        f'<div class="tech"><div class="b" style="background:{c}">{s}</div>{n}</div>' for s, n, c in techs
    )
    with l:
        st.markdown(
            f'<div class="card" style="min-height:190px">{card_title("cpu", "Technology Stack")}'
            f'<div class="techs">{tech_html}</div></div>',
            unsafe_allow_html=True,
        )
    with r:
        st.markdown(
            f'<div class="card" style="min-height:190px">{card_title("warning", "Limitations")}'
            "<p>This is a decision-support tool and not a substitute for professional medical advice. "
            "The dataset has only 180 patient records, so further validation on larger, representative "
            "clinical data would be required before any real-world deployment.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(section_head("target", "Challenges Faced &amp; Techniques Used"), unsafe_allow_html=True)
    ch = [
        ("Categorical Data", "The <code>thal</code> feature is categorical.",
         "One-hot encoding converted it into numeric binary columns."),
        ("Different Feature Scales", "Age, blood pressure, cholesterol and heart rate have very different ranges.",
         "StandardScaler for Logistic Regression, KNN, SVM and Naive Bayes."),
        ("Small Dataset", "Only 180 patient records.",
         "80:20 train-test split with stratification to keep class balance."),
        ("Potential Outliers", "Boxplots showed possible outliers.",
         "Kept them — unusual values in healthcare data may be genuine patient conditions."),
    ]
    st.markdown(
        '<div class="grid c4">' + "".join(
            f'<div class="card"><h4>{t}</h4><p><b>Challenge:</b> {a}</p><p><b>Technique:</b> {b}</p></div>'
            for t, a, b in ch) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(section_head("heart-pulse", "Suggestions to the Hospital"), unsafe_allow_html=True)
    sg = [
        ("Early Risk Identification", "Use the model to identify higher-risk patients and prioritise them for evaluation."),
        ("Detailed Medical Assessment", "High-risk patients should get clinical assessment and tests recommended by professionals."),
        ("Regular Monitoring", "Monitor blood pressure, cholesterol, heart rate and other clinical indicators."),
        ("Preventive Care", "Encourage preventive-care practices and healthy-lifestyle guidance."),
        ("Decision-Support Only", "Use predictions as a support tool, never as a replacement for doctors."),
        ("Continuous Evaluation", "Re-evaluate the model on new, representative data before any real-world use."),
    ]
    st.markdown(
        '<div class="grid c3">' + "".join(
            f'<div class="card"><h4>{i+1}. {t}</h4><p>{d}</p></div>' for i, (t, d) in enumerate(sg)
        ) + "</div>",
        unsafe_allow_html=True,
    )
    footer()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
{
    "Home": page_home,
    "Prediction": page_prediction,
    "Performance": page_performance,
    "Insights": page_insights,
    "About": page_about,
}[page]()

if page == "Home":
    footer()
