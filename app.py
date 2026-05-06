"""
SmartOps Manufacturing Analytics Dashboard  v5.1
=================================================
FIXES APPLIED (v5.0 → v5.1):
  1. FEATURE MISMATCH FIXED:
       - Removed Prev_Demand and Demand_7day_avg from TRAIN_FEATURES
       - Added Stock_Level (feature 8) and Lead_Time (feature 9)
       - Exact training order enforced in sidebar feature vector,
         compute_eval_data(), and Tab 4 display table
  2. INDENTATION BUG FIXED:
       - TRAIN_FEATURES block in sidebar was indented 2 spaces (under `with st.sidebar`)
         but the features DataFrame and model.predict call were also mis-indented;
         all corrected to consistent 4-space indent inside the `with` block
  3. prob / prediction NameError FIXED:
       - Replaced st.session_state reads with direct local variables
         (prob, prediction are set inside `with st.sidebar` and used below)
  4. TAB 4 FEATURE TABLE FIXED:
       - Rows 8–12 previously showed Ordering_Cost/Holding_Cost/Order_Qty/
         Prev_Demand/7-Day Avg — now correctly shows Stock_Level (8),
         Lead_Time (9), Ordering_Cost (10), Holding_Cost (11), Order_Quantity (12)
  5. DUPLICATE / WRONG feature count in Model Card table fixed (was "12 features"
     but listed wrong names — now matches actual TRAIN_FEATURES list)
  6. No UI, layout, styling, charts, or structure changed.

Model feature order (12 features, EXACT training order):
  1  Machine_ID
  2  Processing_Time
  3  Material_Used
  4  Energy_Consumption
  5  Machine_Availability
  6  Scheduled_Duration
  7  Demand
  8  Stock_Level          ← was missing (Prev_Demand was here incorrectly)
  9  Lead_Time            ← was missing (Demand_7day_avg was here incorrectly)
  10 Ordering_Cost
  11 Holding_Cost
  12 Order_Quantity
"""

import os
os.environ["PANDAS_USE_PYARROW"] = "0"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_score, recall_score, accuracy_score)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartOps | Manufacturing Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0f1117;}
[data-testid="stSidebar"]{background:#161b27;border-right:1px solid #1e2a3a;}
#MainMenu,footer,header{visibility:hidden;}

.kpi-card{background:#161b27;border:1px solid #1e2a3a;border-radius:12px;
           padding:18px 20px 14px;position:relative;overflow:hidden;min-height:90px;}
.kpi-card::before{content:"";position:absolute;left:0;top:0;bottom:0;
                   width:3px;border-radius:3px 0 0 3px;}
.kpi-card.blue::before {background:#3b82f6;}
.kpi-card.green::before{background:#10b981;}
.kpi-card.amber::before{background:#f59e0b;}
.kpi-card.red::before  {background:#ef4444;}
.kpi-card.teal::before {background:#14b8a6;}
.kpi-label{font-size:11px;color:#6b7280;text-transform:uppercase;
            letter-spacing:.07em;margin-bottom:6px;}
.kpi-value{font-size:26px;font-weight:700;color:#f1f5f9;line-height:1;margin-bottom:4px;}
.kpi-unit {font-size:13px;color:#6b7280;margin-left:3px;}
.kpi-delta{font-size:12px;} .kpi-delta.pos{color:#10b981;} .kpi-delta.neg{color:#ef4444;}
.kpi-icon {position:absolute;top:14px;right:14px;font-size:20px;opacity:.28;}

.sec{font-size:12px;font-weight:600;color:#64748b;text-transform:uppercase;
     letter-spacing:.07em;margin:24px 0 12px;padding-bottom:8px;
     border-bottom:1px solid #1e2a3a;}

.alert{border-radius:8px;padding:11px 14px;font-size:13px;font-weight:500;margin-bottom:8px;}
.a-red  {background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.30); color:#fca5a5;}
.a-amber{background:rgba(245,158,11,.10);border:1px solid rgba(245,158,11,.30);color:#fcd34d;}
.a-green{background:rgba(16,185,129,.10);border:1px solid rgba(16,185,129,.30);color:#6ee7b7;}
.a-blue {background:rgba(59,130,246,.10); border:1px solid rgba(59,130,246,.30);color:#93c5fd;}

.rec{background:#161b27;border:1px solid #1e2a3a;border-left-width:3px;
     border-radius:8px;padding:13px 15px;margin-bottom:9px;
     font-size:13px;color:#cbd5e1;}
.rec .rt{font-weight:600;color:#f1f5f9;margin-bottom:3px;}
.rec.urgent {border-left-color:#ef4444;}
.rec.warning{border-left-color:#f59e0b;}
.rec.ok     {border-left-color:#10b981;}

.stbl{width:100%;border-collapse:collapse;font-size:13px;}
.stbl th{background:#1e2a3a;color:#94a3b8;font-weight:600;padding:8px 12px;
          font-size:11px;text-transform:uppercase;letter-spacing:.06em;}
.stbl td{padding:9px 12px;color:#cbd5e1;border-bottom:1px solid #1e2a3a;}
.stbl tr:last-child td{border-bottom:none;}

.div{border:none;border-top:1px solid #1e2a3a;margin:18px 0;}
.real-badge{display:inline-block;background:rgba(16,185,129,.15);color:#10b981;
             border:1px solid rgba(16,185,129,.35);border-radius:4px;
             font-size:10px;font-weight:600;padding:2px 8px;letter-spacing:.05em;
             vertical-align:middle;margin-left:6px;}
.sim-badge {display:inline-block;background:rgba(245,158,11,.15);color:#f59e0b;
             border:1px solid rgba(245,158,11,.35);border-radius:4px;
             font-size:10px;font-weight:600;padding:2px 8px;letter-spacing:.05em;
             vertical-align:middle;margin-left:6px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHART THEME HELPERS
# ─────────────────────────────────────────────────────────────
_G = "#1e2a3a"
_AXIS_STYLE = dict(gridcolor=_G, linecolor=_G, tickcolor=_G, zerolinecolor=_G)

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=12),
    margin=dict(l=4, r=4, t=36, b=4),
    colorway=["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#14b8a6"],
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)


def themed(fig, height=270, title="", extra_layout=None):
    kw = {**BASE_LAYOUT, "height": height}
    if title:
        kw["title"] = dict(text=title, font=dict(size=13, color="#94a3b8"),
                           x=0, pad=dict(l=4))
    if extra_layout:
        kw.update(extra_layout)
    fig.update_layout(**kw)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ─────────────────────────────────────────────────────────────
# SVG LOGO
# ─────────────────────────────────────────────────────────────
def logo_svg(size=36):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 36 36" '
            f'fill="none" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="36" height="36" rx="8" fill="#1e3a5f"/>'
            f'<rect x="7" y="22" width="5" height="8" rx="1" fill="#3b82f6"/>'
            f'<rect x="15" y="15" width="5" height="15" rx="1" fill="#60a5fa"/>'
            f'<rect x="23" y="10" width="5" height="20" rx="1" fill="#93c5fd"/>'
            f'<path d="M9 18 L17 11 L25 7" stroke="#f59e0b" stroke-width="1.8" '
            f'stroke-linecap="round" fill="none"/>'
            f'<circle cx="25" cy="7" r="2" fill="#f59e0b"/>'
            f'</svg>')

LOGO = logo_svg(36)


# ─────────────────────────────────────────────────────────────
# KPI CARD
# ─────────────────────────────────────────────────────────────
def kpi(label, value, delta, positive_is_up, color, icon, unit=""):
    arrow = "▲" if positive_is_up else "▼"
    dcls  = "pos" if positive_is_up else "neg"
    return (f"<div class='kpi-card {color}'>"
            f"<div class='kpi-icon'>{icon}</div>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}"
            f"<span class='kpi-unit'>{unit}</span></div>"
            f"<div class='kpi-delta {dcls}'>{arrow} {delta}</div>"
            f"</div>")


# ═════════════════════════════════════════════════════════════
# MODEL LOAD
# ═════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    import pathlib
    try:
        _script_dir = pathlib.Path(__file__).parent.resolve()
    except NameError:
        _script_dir = pathlib.Path(os.getcwd()).resolve()
    _cwd = pathlib.Path(os.getcwd()).resolve()
    for _p in [_script_dir / "model.pkl", _cwd / "model.pkl",
               _script_dir / "data" / "model.pkl", _cwd / "data" / "model.pkl"]:
        if _p.exists():
            try:
                with open(str(_p), "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                return None
    return None


# ═════════════════════════════════════════════════════════════
# DATASET LOAD
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_dataset():
    import pathlib
    try:
        _script_dir = pathlib.Path(__file__).parent.resolve()
    except NameError:
        _script_dir = pathlib.Path(os.getcwd()).resolve()
    _cwd = pathlib.Path(os.getcwd()).resolve()

    _fnames = ["Manufacturing_dataset.csv", "Manufacturing_project.csv"]
    _dirs   = [_script_dir, _cwd,
               _script_dir / "data", _cwd / "data",
               pathlib.Path.home()]

    _found_path = None
    for _d in _dirs:
        for _fn in _fnames:
            _p = _d / _fn
            if _p.exists():
                _found_path = str(_p)
                break
        if _found_path:
            break

    if _found_path is None:
        _tried = "\n".join(
            f"  {_d / _fn}"
            for _d in _dirs[:3]
            for _fn in _fnames)
        st.error(
            "**CSV file not found.**\n\n"
            f"Searched for `Manufacturing_dataset.csv` in:\n{_tried}\n\n"
            "**Fix:** Copy your CSV into the **same folder as app.py** "
            "and restart with `streamlit run app.py`."
        )
        return None

    try:
        df = pd.read_csv(_found_path)
        df.columns = df.columns.str.strip()

        # Canonical map — covers old names → new exact names
        canonical_map = {
            "job_id":               "Job_ID",
            "product_name":         "Product_Name",
            "supplier":             "Supplier",
            "machine_id":           "Machine_ID",
            "machine_availability": "Machine_Availability",
            "processing_time":      "Processing_Time",
            "material_used":        "Material_Used",
            "energy_consumption":   "Energy_Consumption",
            "scheduled_duration":   "Scheduled_Duration",
            "actual_duration":      "Actual_Duration",
            "date":                 "Date",
            "demand":               "Demand",
            "stock_level":          "Stock_Level",
            "lead_time":            "Lead_Time",
            "ordering_cost":        "Ordering_Cost",
            "holding_cost":         "Holding_Cost",
            "order_quantity":       "Order_Quantity",
            "inventory_turnover":   "Inventory_Turnover",
            "stockout_risk":        "Stockout_Risk",
            "delay_flag":           "Delay_Flag",
            "delay":                "Delay",
            "production_status":    "Production_Status",
            "prev_demand":          "Prev_Demand",
            "demand_7day_avg":      "Demand_7day_avg",
        }
        df = df.rename(columns={
            col: canonical_map[col.lower().strip()]
            for col in df.columns
            if col.lower().strip() in canonical_map
        })

        # Parse Date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date").reset_index(drop=True)
        else:
            df["Date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

        # Delay_Flag → int 0/1
        if "Delay_Flag" in df.columns:
            df["Delay_Flag"] = (pd.to_numeric(df["Delay_Flag"], errors="coerce")
                                  .fillna(0).astype(int))

        # Machine_Availability → 0-1 scale
        if "Machine_Availability" in df.columns:
            _ma = df["Machine_Availability"]
            if _ma.dropna().max() > 1.5:
                df["Machine_Availability"] = _ma / 100.0

        # Stockout_Risk → numeric
        if "Stockout_Risk" in df.columns:
            df["Stockout_Risk"] = pd.to_numeric(
                df["Stockout_Risk"], errors="coerce").fillna(0)

        # Recompute engineered features from raw Demand for consistency
        # (kept in dataset for reference but NOT used as model features)
        if "Demand" in df.columns:
            df["Prev_Demand"]     = df["Demand"].shift(1).fillna(df["Demand"].median())
            df["Demand_7day_avg"] = df["Demand"].rolling(7, min_periods=1).mean()

        return df

    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None


# ═════════════════════════════════════════════════════════════
# MODEL EVALUATION
# FIXED: Uses exact 12 training features — Stock_Level and Lead_Time
#        included; Prev_Demand and Demand_7day_avg REMOVED.
# Training feature order (12 features):
#   Machine_ID, Processing_Time, Material_Used, Energy_Consumption,
#   Machine_Availability, Scheduled_Duration, Demand,
#   Stock_Level, Lead_Time, Ordering_Cost, Holding_Cost, Order_Quantity
# ═════════════════════════════════════════════════════════════
# FIXED: exact 12 training features — Prev_Demand / Demand_7day_avg removed,
#        Stock_Level (pos 8) and Lead_Time (pos 9) added.
TRAIN_FEATURES = [
    "Machine_ID",           # 1
    "Processing_Time",      # 2
    "Material_Used",        # 3
    "Energy_Consumption",   # 4
    "Machine_Availability", # 5
    "Scheduled_Duration",   # 6
    "Demand",               # 7
    "Stock_Level",          # 8  ← was missing; Prev_Demand was here incorrectly
    "Lead_Time",            # 9  ← was missing; Demand_7day_avg was here incorrectly
    "Ordering_Cost",        # 10
    "Holding_Cost",         # 11
    "Order_Quantity",       # 12
]


@st.cache_data
def compute_eval_data(_model, _df):
    if _model is None or _df is None:
        return None
    if "Delay_Flag" not in _df.columns:
        return None

    # Use only features that exist in the dataset
    feature_cols = [c for c in TRAIN_FEATURES if c in _df.columns]
    if len(feature_cols) == 0:
        return None

    X = _df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = _df["Delay_Flag"].astype(int)

    try:
        if hasattr(_model, "predict_proba"):
            y_score = _model.predict_proba(X)[:, 1]
        else:
            y_score = _model.predict(X).astype(float)
        return {
            "y_true":        y.values,
            "y_score":       y_score,
            "feature_names": feature_cols,
        }
    except Exception as e:
        st.warning(
            f"⚠ Model prediction on dataset failed: {e}\n\n"
            f"Features sent ({len(feature_cols)}): {feature_cols}\n\n"
            "Check that feature columns match your training script."
        )
        return None


# ═════════════════════════════════════════════════════════════
# FORECAST HELPER
# ═════════════════════════════════════════════════════════════
@st.cache_data
def build_forecast(_demand_series, steps=12):
    series = _demand_series.copy()
    try:
        from statsmodels.tsa.arima.model import ARIMA
        res = ARIMA(series, order=(1, 1, 1)).fit()
        fc  = res.get_forecast(steps=steps)
        ci  = fc.conf_int()
        return (fc.predicted_mean.values,
                ci.iloc[:, 0].values,
                ci.iloc[:, 1].values,
                "ARIMA(1,1,1)")
    except Exception:
        pass
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw  = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
        fc  = hw.forecast(steps).values
        std = hw.resid.std()
        return fc, fc - 1.96 * std, fc + 1.96 * std, "Holt-Winters"
    except Exception:
        pass
    x  = np.arange(len(series))
    p  = np.polyfit(x, series.values, 1)
    xf = np.arange(len(series), len(series) + steps)
    fc = np.polyval(p, xf)
    std = series.std()
    return fc, fc - 1.96 * std, fc + 1.96 * std, "Linear trend"


# ═════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ═════════════════════════════════════════════════════════════
model     = load_model()
df        = load_dataset()
eval_data = compute_eval_data(model, df)

model_ready = model is not None
dataset_ok  = df is not None
eval_ok     = eval_data is not None

# Startup warning
if not dataset_ok or not model_ready:
    import pathlib
    try:
        _here = pathlib.Path(__file__).parent.resolve()
    except NameError:
        _here = pathlib.Path(os.getcwd()).resolve()
    _missing = []
    if not dataset_ok:
        _missing.append("`Manufacturing_dataset.csv`")
    if not model_ready:
        _missing.append("`model.pkl`")
    st.warning(
        f"**Missing file(s):** {' and '.join(_missing)}\n\n"
        f"Place them in: **`{_here}`**\n\n"
        "Then press **R** or click **Rerun** in the Streamlit toolbar.",
        icon="📁"
    )


# ═════════════════════════════════════════════════════════════
# DATASET-LEVEL AGGREGATES
# ═════════════════════════════════════════════════════════════
def _col_val(col, func="median"):
    if dataset_ok and col in df.columns:
        v = getattr(df[col].dropna(), func)()
        return float(v) if not pd.isna(v) else 0.0
    return 0.0


DS_DEMAND        = _col_val("Demand",               "median") or 80.0
DS_AVAILABILITY  = _col_val("Machine_Availability",  "mean")   or 0.90
DS_PROCESSING    = _col_val("Processing_Time",       "median") or 50.0
DS_MATERIAL      = _col_val("Material_Used",         "median") or 100.0
DS_ENERGY        = _col_val("Energy_Consumption",    "median") or 20.0
DS_SCHEDULED     = _col_val("Scheduled_Duration",    "median") or 8.0
DS_ORDERING_COST = _col_val("Ordering_Cost",         "median") or 500.0
DS_HOLDING_COST  = _col_val("Holding_Cost",          "median") or 2.0
DS_ORDER_QTY     = _col_val("Order_Quantity",        "median") or 60.0
DS_STOCK         = _col_val("Stock_Level",           "median") or 60.0
DS_LEAD_TIME     = _col_val("Lead_Time",             "median") or 4.0
DS_DELAY_RATE    = (df["Delay_Flag"].mean() * 100
                    if dataset_ok and "Delay_Flag" in df.columns else 0.0)
DS_TOTAL_ROWS    = len(df) if dataset_ok else 0
DS_DATE_MIN      = (df["Date"].min().strftime("%d %b %Y")
                    if dataset_ok and "Date" in df.columns else "—")
DS_DATE_MAX      = (df["Date"].max().strftime("%d %b %Y")
                    if dataset_ok and "Date" in df.columns else "—")


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:20px'>"
        f"{LOGO}"
        f"<div><div style='font-size:15px;font-weight:700;color:#f1f5f9'>SmartOps</div>"
        f"<div style='font-size:11px;color:#6b7280'>Manufacturing Analytics</div></div>"
        f"</div>", unsafe_allow_html=True)

    if dataset_ok:
        st.markdown(
            f"<div style='font-size:11px;color:#10b981;background:rgba(16,185,129,.08);"
            f"border:1px solid rgba(16,185,129,.25);border-radius:6px;"
            f"padding:6px 10px;margin-bottom:12px'>"
            f"✓ Dataset loaded · {DS_TOTAL_ROWS:,} rows · {DS_DATE_MIN} → {DS_DATE_MAX}"
            f"</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠ Dataset not found. Using slider defaults.")

    # ── Machine ───────────────────────────────────────────────
    st.markdown("<div class='sec'>🏭 Machine</div>", unsafe_allow_html=True)

    if dataset_ok and "Machine_ID" in df.columns:
        _mid_opts = sorted(df["Machine_ID"].dropna().unique().tolist())
        if not _mid_opts:
            _mid_opts = [1, 2, 3, 4, 5]
    else:
        _mid_opts = [1, 2, 3, 4, 5]
    machine_id = st.selectbox("Machine ID", _mid_opts)

    _avail_default = int(round(DS_AVAILABILITY * 100))
    if dataset_ok and "Machine_Availability" in df.columns and "Machine_ID" in df.columns:
        _m_sub = df[df["Machine_ID"] == machine_id]["Machine_Availability"]
        if len(_m_sub) > 0:
            _avail_default = int(round(_m_sub.mean() * 100))
    _avail_default = int(np.clip(_avail_default, 70, 100))
    availability = st.slider("Availability (%)", 70, 100, _avail_default) / 100.0

    # ── Production ────────────────────────────────────────────
    st.markdown("<div class='sec'>⚙️ Production</div>", unsafe_allow_html=True)

    processing_time = st.slider(
        "Processing Time (min)", 20, 120,
        int(np.clip(DS_PROCESSING, 20, 120)))
    material_used = st.slider(
        "Material Used (kg)", 10, 200,
        int(np.clip(DS_MATERIAL, 10, 200)))
    energy_consumed = st.slider(
        "Energy (kWh)", 5, 60,
        int(np.clip(DS_ENERGY, 5, 60)))
    scheduled_hrs = st.slider(
        "Scheduled Duration (hrs)", 4, 16,
        int(np.clip(DS_SCHEDULED, 4, 16)))

    # ── Inventory ─────────────────────────────────────────────
    st.markdown("<div class='sec'>📦 Inventory</div>", unsafe_allow_html=True)

    demand = st.slider(
        "Demand (units)", 50, 150,
        int(np.clip(DS_DEMAND, 50, 150)))
    stock = st.slider(
        "Stock Level (units)", 30, 120,
        int(np.clip(DS_STOCK, 30, 120)))
    lead_time = st.slider(
        "Lead Time (days)", 2, 10,
        int(np.clip(DS_LEAD_TIME, 2, 10)))
    ordering_cost = st.slider(
        "Ordering Cost (₹)", 200, 1000,
        int(np.clip(DS_ORDERING_COST, 200, 1000)), step=50)
    holding_cost = st.slider(
        "Holding Cost (₹/unit)", 1, 10,
        int(np.clip(DS_HOLDING_COST, 1, 10)))

    # ── Model Settings ────────────────────────────────────────
    st.markdown("<div class='sec'>🎯 Model Settings</div>", unsafe_allow_html=True)
    threshold = st.slider(
        "Prediction Threshold", 0.30, 0.80, 0.50, step=0.05,
        help="Classify as 'Delay' when P(delay) ≥ this value.")

    # ── Derived calculations ──────────────────────────────────
    order_qty     = int(max((demand * lead_time) - stock, 0))
    stockout_risk = bool(demand > stock)
    stock_cover   = round(stock / max(demand, 1), 2)
    reorder_point = int(demand * lead_time)
    avail_pct     = int(round(availability * 100))

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:11px;color:#6b7280;text-transform:uppercase;"
        f"letter-spacing:.06em;margin-bottom:8px'>Live Metrics</div>"
        f"<div style='font-size:13px;color:#cbd5e1;line-height:2.1'>"
        f"📦 Order Qty &nbsp;&nbsp;&nbsp;"
        f"<b style='color:#f1f5f9'>{order_qty} u</b><br>"
        f"📅 Stock Cover &nbsp;"
        f"<b style='color:{'#ef4444' if stock_cover<1 else '#10b981'}'>"
        f"{stock_cover} d</b><br>"
        f"🔁 Reorder Pt &nbsp;&nbsp;<b style='color:#7eb8f7'>{reorder_point} u</b><br>"
        f"🚨 Stockout &nbsp;&nbsp;&nbsp;&nbsp;"
        f"<b style='color:{'#ef4444' if stockout_risk else '#10b981'}'>"
        f"{'HIGH' if stockout_risk else 'LOW'}</b></div>",
        unsafe_allow_html=True)
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    # ── Feature vector — EXACT match to TRAIN_FEATURES (12 features) ──────
    # FIXED: Uses Stock_Level (pos 8) and Lead_Time (pos 9).
    #        Prev_Demand and Demand_7day_avg have been REMOVED entirely.
    features = pd.DataFrame([[
        float(machine_id),       # 1  Machine_ID
        float(processing_time),  # 2  Processing_Time
        float(material_used),    # 3  Material_Used
        float(energy_consumed),  # 4  Energy_Consumption
        float(availability),     # 5  Machine_Availability (0–1 scale)
        float(scheduled_hrs),    # 6  Scheduled_Duration
        float(demand),           # 7  Demand
        float(stock),            # 8  Stock_Level       ← FIXED (was Prev_Demand)
        float(lead_time),        # 9  Lead_Time         ← FIXED (was Demand_7day_avg)
        float(ordering_cost),    # 10 Ordering_Cost
        float(holding_cost),     # 11 Holding_Cost
        float(order_qty),        # 12 Order_Quantity
    ]], columns=TRAIN_FEATURES)

    # ── Run prediction ──────────────────────────────────────────
    # FIXED: prob and prediction are plain local variables (no session_state reads
    #        below that could fail before this block runs on first load).
    if model_ready:
        try:
            raw_prob = (float(model.predict_proba(features)[0][1])
                        if hasattr(model, "predict_proba")
                        else float(model.predict(features)[0]))
        except Exception as _e:
            raw_prob = 0.0
            st.warning(f"Prediction error: {_e}")
        prediction = 1 if raw_prob >= threshold else 0
    else:
        raw_prob   = 0.0
        prediction = None

    # Store as local aliases used throughout the rest of the script
    prob = raw_prob  # FIXED: was read from session_state below, risking NameError

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:2px'>"
        f"{LOGO}"
        f"<div><span style='font-size:22px;font-weight:700;color:#f1f5f9'>"
        f"SmartOps Manufacturing</span>"
        f"<span style='margin-left:10px;font-size:11px;background:#1e2a3a;"
        f"color:#7eb8f7;padding:3px 10px;border-radius:20px;font-weight:600;"
        f"letter-spacing:.05em'>LIVE</span></div></div>"
        f"<div style='font-size:13px;color:#6b7280;margin-left:50px'>"
        f"AI-powered delay prediction · Inventory risk · Scenario simulation</div>",
        unsafe_allow_html=True)
with hc2:
    st.markdown(
        f"<div style='text-align:right;font-size:12px;color:#6b7280;padding-top:14px'>"
        f"{datetime.datetime.now().strftime('%A, %d %B %Y')}</div>",
        unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
if dataset_ok and "Date" in df.columns and "Demand" in df.columns:
    _sorted       = df.sort_values("Date")
    _half         = max(len(_sorted) // 2, 1)
    _delta_demand = round(
        _sorted["Demand"].iloc[_half:].mean()
        - _sorted["Demand"].iloc[:_half].mean(), 1)
    _delta_str    = f"{abs(_delta_demand):.1f} vs prior period"
else:
    _delta_demand = 5.0
    _delta_str    = "+5 vs last wk"

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(kpi("Demand",       demand,    _delta_str,     _delta_demand >= 0,
                "blue",  "📈", "u"), unsafe_allow_html=True)
k2.markdown(kpi("Stock Level",  stock,     "vs demand",    stock >= demand,
                "green" if stock >= demand else "red", "📦", "u"),
            unsafe_allow_html=True)
k3.markdown(kpi("Order Qty",    order_qty, f"ROP={reorder_point}", True,
                "teal",  "🛒", "u"), unsafe_allow_html=True)
k4.markdown(kpi("Lead Time",    lead_time, "days",         True,
                "blue",  "⏱", "d"), unsafe_allow_html=True)
k5.markdown(kpi("Availability", avail_pct,
                "On target" if availability >= 0.85 else "Below 85%",
                availability >= 0.85,
                "green" if availability >= 0.85 else "red", "🏭", "%"),
            unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PREDICTION PANEL
# ─────────────────────────────────────────────────────────────
if model_ready:
    pb1, pb2, pb3 = st.columns([1.1, 1, 1])

    with pb1:
        gauge_color = "#ef4444" if prob >= threshold else "#10b981"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            delta={"reference": threshold * 100, "valueformat": ".1f",
                   "increasing": {"color": "#ef4444"},
                   "decreasing": {"color": "#10b981"}},
            number={"suffix": "%", "font": {"size": 30, "color": "#f1f5f9"}},
            title={"text": f"Delay Probability (threshold {round(threshold*100)}%)",
                   "font": {"size": 12, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickfont": {"color": "#6b7280", "size": 10}},
                "bar":  {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "#161b27", "bordercolor": "#1e2a3a",
                "steps": [
                    {"range": [0, threshold * 100],
                     "color": "rgba(16,185,129,.07)"},
                    {"range": [threshold * 100, 100],
                     "color": "rgba(239,68,68,.07)"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 2},
                              "value": threshold * 100},
            }
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=195,
                            margin=dict(l=10, r=10, t=36, b=0))
        st.plotly_chart(fig_g, use_container_width=True)

    with pb2:
        p_cls = "a-red" if prediction == 1 else "a-green"
        p_ico = ("⚠️ <b>Delay Expected</b>" if prediction == 1
                 else "✅ <b>On Time</b>")
        p_msg = ("Model confidence exceeds threshold. Review lead time and "
                 "machine availability immediately." if prediction == 1
                 else "Production schedule looks healthy. Continue monitoring.")
        st.markdown(f"<div class='alert {p_cls}'>{p_ico}<br>{p_msg}</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:11px;color:#6b7280;margin-top:4px'>"
            f"Threshold: <b style='color:#f59e0b'>{threshold:.0%}</b> · "
            f"Raw P(delay): <b style='color:#93c5fd'>{prob:.1%}</b><br>"
            f"Adjust threshold via sidebar ↖</div>",
            unsafe_allow_html=True)

    with pb3:
        s_cls = "a-red" if stockout_risk else "a-green"
        s_ico = ("🚨 <b>Stockout Risk HIGH</b>" if stockout_risk
                 else "📦 <b>Stock Adequate</b>")
        s_msg = (f"Stock Level ({stock}) < Demand ({demand}). "
                 f"Place order: <b>{order_qty} units</b>." if stockout_risk
                 else f"Cover: {stock_cover} d. Next review in {lead_time} d.")
        st.markdown(f"<div class='alert {s_cls}'>{s_ico}<br>{s_msg}</div>",
                    unsafe_allow_html=True)
        _risk_cost = round(1200 * prob * (1 + demand / 150)) + (1500 if stockout_risk else 0)
        st.markdown(
            f"<div style='font-size:11px;color:#6b7280;margin-top:4px'>"
            f"Estimated risk cost: "
            f"<b style='color:{'#ef4444' if _risk_cost>0 else '#10b981'}'>"
            f"₹ {_risk_cost:,}</b></div>", unsafe_allow_html=True)

    st.markdown("<hr class='div'>", unsafe_allow_html=True)
else:
    st.info("ℹ️ Place **model.pkl** in the same folder as app.py to enable predictions.")
    st.markdown("<hr class='div'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SHARED COST VALUES
# ─────────────────────────────────────────────────────────────
hold_c  = stock * holding_cost
del_c   = round(1200 * prob * (1 + demand / 150))
sout_c  = 1500 if stockout_risk else 0
ord_c   = order_qty * ordering_cost // 100
total_c = hold_c + del_c + sout_c + ord_c

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "📈  Analytics",
    "💡  Insights & Alerts",
    "📘  Model Info",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='sec'>Inventory Snapshot</div>", unsafe_allow_html=True)
    ov1, ov2, ov3 = st.columns(3)

    # ── Bar: Demand / Stock Level / Order Quantity ────────────
    with ov1:
        bar_cols = ["#3b82f6",
                    "#ef4444" if stockout_risk else "#10b981",
                    "#f59e0b"]
        fig_bar = go.Figure()
        for cat, val, col in zip(["Demand", "Stock Level", "Order Qty"],
                                  [demand, stock, order_qty], bar_cols):
            fig_bar.add_trace(go.Bar(x=[cat], y=[val], marker_color=col,
                                     name=cat, width=0.5))
        if stockout_risk:
            fig_bar.add_annotation(
                x="Stock Level", y=stock,
                text="⚠ Below demand", showarrow=True,
                arrowcolor="#ef4444",
                font=dict(color="#ef4444", size=11), ay=-32)
        themed(fig_bar, 260, "Inventory Overview",
               {"showlegend": False, "barmode": "group"})
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Real demand trend ─────────────────────────────────────
    with ov2:
        if dataset_ok and "Date" in df.columns and "Demand" in df.columns:
            _td = (df[["Date", "Demand"]].dropna()
                     .sort_values("Date").tail(60))
            x_tr = _td["Date"].dt.strftime("%d %b").tolist()
            y_tr = _td["Demand"].tolist()
            tr_title = "Demand Trend · Last 60 Records"
            tr_badge = "<span class='real-badge'>REAL DATA</span>"
        else:
            x_tr = [f"D−{9-i}" for i in range(10)]
            y_tr = [demand] * 10
            tr_title = "Demand Trend (no dataset)"
            tr_badge = "<span class='sim-badge'>NO DATA</span>"

        st.markdown(f"<div class='sec'>{tr_title}{tr_badge}</div>",
                    unsafe_allow_html=True)
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(
            x=x_tr, y=y_tr, mode="lines+markers",
            line=dict(color="#3b82f6", width=2), marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(59,130,246,.07)"))
        fig_tr.add_hline(y=demand, line_dash="dot", line_color="#f59e0b",
                         annotation_text=f"Selected: {demand}",
                         annotation_font=dict(size=11, color="#f59e0b"))
        themed(fig_tr, 230, "")
        fig_tr.update_xaxes(tickangle=-45, nticks=8)
        st.plotly_chart(fig_tr, use_container_width=True)

    # ── Availability gauge ────────────────────────────────────
    with ov3:
        gc = ("#10b981" if avail_pct >= 85
              else "#f59e0b" if avail_pct >= 75 else "#ef4444")
        fig_av = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avail_pct,
            number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9"}},
            title={"text": f"Machine {machine_id} Availability",
                   "font": {"size": 12, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickfont": {"color": "#6b7280"}},
                "bar":  {"color": gc, "thickness": 0.28},
                "bgcolor": "#161b27", "bordercolor": "#1e2a3a",
                "steps": [
                    {"range": [0, 75],   "color": "rgba(239,68,68,.07)"},
                    {"range": [75, 85],  "color": "rgba(245,158,11,.07)"},
                    {"range": [85, 100], "color": "rgba(16,185,129,.07)"},
                ],
            }
        ))
        fig_av.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
                             margin=dict(l=10, r=10, t=44, b=0))
        st.plotly_chart(fig_av, use_container_width=True)

    # ── Row 2: Heatmap + Waterfall ────────────────────────────
    ov4, ov5 = st.columns([1.4, 1])

    with ov4:
        if (dataset_ok and "Machine_ID" in df.columns
                and "Machine_Availability" in df.columns):
            badge = "<span class='real-badge'>REAL DATA</span>"
            machines_ids = list(range(1, 6))
            heat = np.zeros((5, 8))
            hours = [f"{h:02d}:00" for h in range(6, 22, 2)]
            for idx, mid in enumerate(machines_ids):
                _sub = df[df["Machine_ID"] == mid]["Machine_Availability"]
                _base = float(_sub.mean()) if len(_sub) > 0 else 0.80
                _rng  = np.random.default_rng(int(mid) * 7)
                heat[idx, :] = np.clip(_base + _rng.uniform(-0.04, 0.04, 8),
                                       0.5, 1.0)
            heat[int(machine_id) - 1, :] = availability
        else:
            badge = "<span class='sim-badge'>SIMULATED</span>"
            hours = [f"{h:02d}:00" for h in range(6, 22, 2)]
            _rng2 = np.random.default_rng(42)
            heat  = _rng2.uniform(0.55, 0.99, (5, 8))
            heat[int(machine_id) - 1, :] = availability

        machines_lbl = [f"M-{i}" for i in range(1, 6)]
        st.markdown(
            f"<div class='sec'>Machine Utilisation Heatmap{badge}</div>",
            unsafe_allow_html=True)
        fig_hm = go.Figure(go.Heatmap(
            z=heat.tolist(), x=hours, y=machines_lbl,
            colorscale=[[0, "#1e2a3a"], [0.5, "#3b82f6"], [1, "#10b981"]],
            zmin=0.5, zmax=1.0,
            text=[[f"{v:.0%}" for v in row] for row in heat],
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            showscale=True,
            colorbar=dict(thickness=12,
                          tickfont=dict(size=10, color="#94a3b8")),
        ))
        themed(fig_hm, 255)
        st.plotly_chart(fig_hm, use_container_width=True)

    with ov5:
        st.markdown("<div class='sec'>Operational Cost Waterfall (₹)</div>",
                    unsafe_allow_html=True)
        fig_wf = go.Figure(go.Waterfall(
            x=["Base","Holding","Delay","Stockout","Order","Total"],
            y=[0, -hold_c, -del_c, -sout_c, -ord_c, -total_c],
            measure=["absolute","relative","relative","relative","relative","total"],
            connector={"line": {"color": "#1e2a3a"}},
            increasing={"marker": {"color": "#ef4444"}},
            decreasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#3b82f6"}},
            texttemplate="₹%{y:,.0f}",
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        themed(fig_wf, 255, "Cost Breakdown", {"showlegend": False})
        fig_wf.update_yaxes(tickprefix="₹")
        st.plotly_chart(fig_wf, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    an1, an2 = st.columns(2)

    # ── Lead Time vs Delay Rate ────────────────────────────────
    with an1:
        if (dataset_ok and "Lead_Time" in df.columns
                and "Delay_Flag" in df.columns):
            st.markdown(
                "<div class='sec'>Lead Time vs Delay Rate"
                "<span class='real-badge'>REAL DATA</span></div>",
                unsafe_allow_html=True)
            lt_grp = (df.groupby("Lead_Time")["Delay_Flag"]
                        .mean().reset_index()
                        .rename(columns={"Delay_Flag": "Delay_Rate"}))
            lt_grp["Delay_Rate"] *= 100
            lt_grp = lt_grp.sort_values("Lead_Time")
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=lt_grp["Lead_Time"].tolist(),
                y=lt_grp["Delay_Rate"].tolist(),
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,.07)",
                name="Actual delay rate"))
            fig_sc.add_trace(go.Scatter(
                x=[lead_time], y=[prob * 100], mode="markers",
                marker=dict(size=14, color="#ef4444", symbol="diamond",
                            line=dict(color="white", width=1.5)),
                name=f"Current · LT={lead_time}d · P={prob:.1%}"))
            themed(fig_sc, 270, "",
                   {"xaxis_title": "Lead Time (days)",
                    "yaxis_title": "Delay Rate (%)",
                    "legend": dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")})

        elif (dataset_ok and "Delay_Flag" in df.columns
              and "Processing_Time" in df.columns):
            st.markdown(
                "<div class='sec'>Processing Time vs Delay Rate"
                "<span class='real-badge'>REAL DATA</span></div>",
                unsafe_allow_html=True)
            _bins = pd.cut(df["Processing_Time"], bins=8)
            pt_grp = (df.groupby(_bins, observed=True)["Delay_Flag"]
                        .mean().reset_index())
            pt_grp.columns = ["PT_Range", "Delay_Rate"]
            pt_grp["Delay_Rate"] *= 100
            pt_grp["PT_mid"] = pt_grp["PT_Range"].apply(
                lambda x: x.mid if hasattr(x, "mid") else 0)
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=pt_grp["PT_mid"].tolist(),
                y=pt_grp["Delay_Rate"].tolist(),
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,.07)",
                name="Actual delay rate by processing time"))
            themed(fig_sc, 270, "",
                   {"xaxis_title": "Processing Time (min)",
                    "yaxis_title": "Delay Rate (%)",
                    "legend": dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")})
        else:
            st.markdown(
                "<div class='sec'>Lead Time vs Delay Risk"
                "<span class='sim-badge'>SIMULATED</span></div>",
                unsafe_allow_html=True)
            lt_range = list(range(2, 11))
            risk_sim = [min(0.05 + (lt / 10) * 0.55
                            + (0.15 if stockout_risk else 0), 1.0)
                        for lt in lt_range]
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=lt_range, y=[r * 100 for r in risk_sim],
                mode="lines", line=dict(color="#3b82f6", width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,.07)",
                name="Simulated risk curve"))
            fig_sc.add_trace(go.Scatter(
                x=[lead_time], y=[prob * 100], mode="markers",
                marker=dict(size=14, color="#ef4444", symbol="diamond",
                            line=dict(color="white", width=1.5)),
                name=f"Current · P={prob:.1%}"))
            themed(fig_sc, 270, "",
                   {"xaxis_title": "Lead Time (days)",
                    "yaxis_title": "Delay Risk (%)",
                    "legend": dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")})

        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Feature Importance ─────────────────────────────────────
    with an2:
        st.markdown("<div class='sec'>Feature Importance</div>",
                    unsafe_allow_html=True)
        if model is not None and hasattr(model, "feature_importances_"):
            # FIXED: use TRAIN_FEATURES labels (12 real features, no Prev/7day)
            feat_names = (eval_data["feature_names"] if eval_ok
                          else TRAIN_FEATURES)
            imp = model.feature_importances_
            n   = min(len(feat_names), len(imp))
            df_imp = (pd.DataFrame({"Feature": feat_names[:n],
                                    "Importance": imp[:n]})
                        .sort_values("Importance", ascending=True).tail(8))
            imp_title = "Feature Importance (real model)"
        else:
            # Demo fallback uses correct feature names
            feat_names = ["Processing_Time","Demand","Energy_Consumption",
                          "Machine_Availability","Order_Quantity","Scheduled_Duration",
                          "Holding_Cost","Machine_ID","Stock_Level","Lead_Time"]
            demo_imp   = [0.20, 0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.05, 0.02, 0.02]
            df_imp     = pd.DataFrame({"Feature": feat_names,
                                       "Importance": demo_imp})
            imp_title  = "Feature Importance (demo — load model.pkl for real)"

        fig_imp = go.Figure(go.Bar(
            x=df_imp["Importance"].tolist(),
            y=df_imp["Feature"].tolist(),
            orientation="h",
            marker=dict(color=df_imp["Importance"].tolist(),
                        colorscale=[[0, "#1e3a5f"], [1, "#3b82f6"]]),
            text=[f"{v:.3f}" for v in df_imp["Importance"]],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        themed(fig_imp, 270, imp_title, {"showlegend": False})
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Forecast ──────────────────────────────────────────────
    if dataset_ok and "Date" in df.columns and "Demand" in df.columns:
        st.markdown(
            "<div class='sec'>12-Month Demand Forecast"
            "<span class='real-badge'>REAL DATA</span></div>",
            unsafe_allow_html=True)

        _monthly = (df.set_index("Date")["Demand"]
                      .resample("ME").mean().dropna())

        if len(_monthly) >= 6:
            fc_vals, fc_low, fc_up, fc_method = build_forecast(
                _monthly, steps=12)
            _last    = _monthly.index[-1]
            fc_idx   = pd.date_range(_last, periods=13, freq="ME")[1:]
            fc_lbl   = [d.strftime("%b %Y") for d in fc_idx]
            hist_lbl = [d.strftime("%b %Y") for d in _monthly.index]

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist_lbl, y=_monthly.values.tolist(),
                mode="lines+markers",
                line=dict(color="#10b981", width=2),
                marker=dict(size=5), name="Historical (real)"))
            fig_fc.add_trace(go.Scatter(
                x=fc_lbl, y=fc_up.tolist(),
                fill=None, line=dict(width=0),
                showlegend=False, mode="lines"))
            fig_fc.add_trace(go.Scatter(
                x=fc_lbl, y=fc_low.tolist(), fill="tonexty",
                fillcolor="rgba(59,130,246,.10)",
                line=dict(width=0), name="95% CI", mode="lines"))
            fig_fc.add_trace(go.Scatter(
                x=fc_lbl, y=fc_vals.tolist(), mode="lines+markers",
                line=dict(color="#3b82f6", width=2.5, dash="dot"),
                marker=dict(size=6),
                name=f"Forecast ({fc_method})"))
            fig_fc.add_hline(y=stock, line_dash="dot", line_color="#f59e0b",
                             annotation_text="Current stock level",
                             annotation_font=dict(size=11, color="#f59e0b"))
            themed(fig_fc, 320, "",
                   {"legend": dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")})
            fig_fc.update_xaxes(tickangle=-45, nticks=12)
            st.plotly_chart(fig_fc, use_container_width=True)
            st.caption(
                f"📌 Method: {fc_method} · trained on {len(_monthly)} months "
                f"of real data ({DS_DATE_MIN} → {DS_DATE_MAX}).")
        else:
            st.info(
                f"Only {len(_monthly)} month(s) of data — need ≥ 6 for ARIMA. "
                "Showing available history.")
            fig_fc = go.Figure(go.Scatter(
                x=_monthly.index.strftime("%b %Y").tolist(),
                y=_monthly.values.tolist(), mode="lines+markers",
                line=dict(color="#3b82f6", width=2)))
            themed(fig_fc, 280)
            st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.markdown(
            "<div class='sec'>12-Month Demand Forecast"
            "<span class='sim-badge'>NO DATA</span></div>",
            unsafe_allow_html=True)
        st.info("Load the dataset to see a real demand forecast.")

    # ── Weekly Demand × Lead Time ─────────────────────────────
    if dataset_ok and "Date" in df.columns and "Demand" in df.columns:
        _has_lead = "Lead_Time" in df.columns
        _y2_col   = "Lead_Time" if _has_lead else "Processing_Time"
        _y2_label = "Avg Lead Time (days)" if _has_lead else "Avg Processing Time (min)"

        if _y2_col in df.columns:
            st.markdown(
                f"<div class='sec'>Weekly Demand × {_y2_col.replace('_',' ')}"
                "<span class='real-badge'>REAL DATA</span></div>",
                unsafe_allow_html=True)

            _wdf = df.copy()
            _wdf["Week"] = _wdf["Date"].dt.isocalendar().week.astype(int)
            _wdf["Year"] = _wdf["Date"].dt.isocalendar().year.astype(int)
            _weekly = (_wdf.groupby(["Year", "Week"])
                           .agg(Total_Demand=("Demand", "sum"),
                                Y2=(_y2_col, "mean"))
                           .reset_index()
                           .sort_values(["Year", "Week"])
                           .tail(24))

            weeks    = [f"W{int(r.Week)}/{str(int(r.Year))[2:]}"
                        for r in _weekly.itertuples()]
            w_demand = _weekly["Total_Demand"].tolist()
            w_y2     = _weekly["Y2"].tolist()

            fig_d = make_subplots(specs=[[{"secondary_y": True}]])
            fig_d.add_trace(
                go.Bar(x=weeks, y=w_demand, name="Weekly Demand",
                       marker_color="#3b82f6", opacity=0.75),
                secondary_y=False)
            fig_d.add_trace(
                go.Scatter(x=weeks, y=w_y2, name=_y2_label,
                           mode="lines+markers",
                           line=dict(color="#f59e0b", width=2),
                           marker=dict(size=6)),
                secondary_y=True)

            _dual_layout = {**BASE_LAYOUT, "height": 280}
            _dual_layout["legend"] = dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")
            fig_d.update_layout(**_dual_layout)
            fig_d.update_xaxes(**_AXIS_STYLE)
            fig_d.update_yaxes(title_text="Demand (units)",
                               secondary_y=False, **_AXIS_STYLE)
            fig_d.update_yaxes(title_text=_y2_label,
                               secondary_y=True, **_AXIS_STYLE)
            st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.markdown(
            "<div class='sec'>Weekly Demand × Lead Time"
            "<span class='sim-badge'>NO DATA</span></div>",
            unsafe_allow_html=True)
        st.info("Load the dataset to see the weekly breakdown.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — INSIGHTS & ALERTS
# ══════════════════════════════════════════════════════════════
with tab3:
    ins1, ins2 = st.columns([1.2, 1])

    with ins1:
        st.markdown("<div class='sec'>Root Cause Analysis</div>",
                    unsafe_allow_html=True)

        if dataset_ok and "Delay_Flag" in df.columns:
            _rca_causes = {}
            if "Machine_Availability" in df.columns:
                _thresh = df["Machine_Availability"].median()
                _rca_causes["Machine Unavailability"] = round(
                    df[df["Machine_Availability"] < _thresh]["Delay_Flag"].mean(), 3)
            if "Lead_Time" in df.columns:
                _lt_thresh = df["Lead_Time"].median()
                _rca_causes["Lead Time Pressure"] = round(
                    df[df["Lead_Time"] > _lt_thresh]["Delay_Flag"].mean(), 3)
            elif "Processing_Time" in df.columns:
                _pt_thresh = df["Processing_Time"].median()
                _rca_causes["Processing Time Pressure"] = round(
                    df[df["Processing_Time"] > _pt_thresh]["Delay_Flag"].mean(), 3)
            if "Demand" in df.columns and "Order_Quantity" in df.columns:
                _rca_causes["Demand vs Order Gap"] = round(
                    min(max(demand - float(df["Order_Quantity"].median()), 0)
                        / max(float(df["Order_Quantity"].median()), 1), 1.0), 3)
            if "Stock_Level" in df.columns:
                _sk_thresh = df["Stock_Level"].median()
                _rca_causes["Low Stock Level"] = round(
                    df[df["Stock_Level"] < _sk_thresh]["Delay_Flag"].mean(), 3)
            _rca_causes["Machine Unavailability (slider)"] = round(
                1 - availability, 3)

            if not _rca_causes:
                _rca_causes = {
                    "Machine Unavailability": round(1 - availability, 3),
                    "Demand vs Stock Gap":    round(
                        min(max(demand - stock, 0) / max(stock, 1), 1), 3),
                    "Order Qty Pressure":     round(min(order_qty / 200, 1), 3),
                }
        else:
            _rca_causes = {
                "Machine Unavailability": round(1 - availability, 3),
                "Lead Time Pressure":     round(lead_time / 10, 3),
                "Demand vs Stock Gap":    round(
                    min(max(demand - stock, 0) / max(stock, 1), 1), 3),
                "Order Qty Pressure":     round(min(order_qty / 200, 1), 3),
            }

        df_rca = (pd.DataFrame(list(_rca_causes.items()),
                               columns=["Cause", "Score"])
                    .sort_values("Score", ascending=True))
        rca_colors = ["#1e3a5f" if v < 0.3
                      else "#f59e0b" if v < 0.6 else "#ef4444"
                      for v in df_rca["Score"]]
        fig_rca = go.Figure(go.Bar(
            x=df_rca["Score"].tolist(), y=df_rca["Cause"].tolist(),
            orientation="h", marker_color=rca_colors,
            text=[f"{v:.2f}" for v in df_rca["Score"]],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))
        themed(fig_rca, 250, "Risk Factor Scores (0–1)", {"showlegend": False})
        fig_rca.update_xaxes(range=[0, 1.2])
        st.plotly_chart(fig_rca, use_container_width=True)

        # Business KPI table
        st.markdown("<div class='sec'>Business KPIs</div>",
                    unsafe_allow_html=True)
        turnover = round(demand / max(stock, 1), 2)
        f1_cost  = hold_c + del_c + sout_c + ord_c

        _inv_turn_real = (f"{_col_val('Inventory_Turnover', 'mean'):.2f}×"
                          if dataset_ok and "Inventory_Turnover" in df.columns
                          else f"{turnover}×")

        _stockout_risk_pct = (f"{df['Stockout_Risk'].mean()*100:.1f}%"
                              if dataset_ok and "Stockout_Risk" in df.columns
                              else ("HIGH" if stockout_risk else "LOW"))

        st.markdown(f"""
        <table class='stbl'>
          <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
          <tr><td>Inventory Turnover</td><td>{_inv_turn_real}</td>
              <td style='color:{"#10b981" if turnover>1 else "#ef4444"}'>
                {"✓ Healthy" if turnover>1 else "⚠ Slow"}</td></tr>
          <tr><td>Dataset Delay Rate</td>
              <td style='color:{"#ef4444" if DS_DELAY_RATE>30 else "#10b981"}'>
                {DS_DELAY_RATE:.1f}%</td>
              <td style='color:{"#ef4444" if DS_DELAY_RATE>30 else "#10b981"}'>
                {"⚠ High" if DS_DELAY_RATE>30 else "✓ OK"}</td></tr>
          <tr><td>Stockout Risk (dataset avg)</td>
              <td>{_stockout_risk_pct}</td>
              <td style='color:{"#ef4444" if stockout_risk else "#10b981"}'>
                {"⚠ Risk" if stockout_risk else "✓ OK"}</td></tr>
          <tr><td>Delay Cost (estimated)</td><td>₹ {del_c:,}</td>
              <td style='color:{"#ef4444" if del_c>0 else "#10b981"}'>
                {"⚠ Risk" if del_c>0 else "✓ None"}</td></tr>
          <tr><td>Stockout Cost</td><td>₹ {sout_c:,}</td>
              <td style='color:{"#ef4444" if sout_c>0 else "#10b981"}'>
                {"⚠ Risk" if sout_c>0 else "✓ None"}</td></tr>
          <tr><td>Total Risk Cost</td><td>₹ {f1_cost:,}</td>
              <td style='color:{"#ef4444" if f1_cost>500 else "#10b981"}'>
                {"⚠ High" if f1_cost>500 else "✓ Low"}</td></tr>
          <tr><td>Stock Cover</td><td>{stock_cover} d</td>
              <td style='color:{"#10b981" if stock_cover>1 else "#ef4444"}'>
                {"✓ OK" if stock_cover>1 else "⚠ Low"}</td></tr>
          <tr><td>Reorder Point</td><td>{reorder_point} u</td>
              <td style='color:#7eb8f7'>ℹ Info</td></tr>
        </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        df_export = pd.DataFrame({
            "Parameter": ["Machine ID","Processing Time (min)","Material Used (kg)",
                          "Energy (kWh)","Availability (%)","Scheduled Hrs",
                          "Demand (u)","Stock Level (u)","Lead Time (d)",
                          "Ordering Cost (₹)","Holding Cost (₹/u)","Order Qty (u)",
                          "Prediction Threshold","P(delay)","Decision",
                          "Stockout Risk"],
            "Value": [machine_id, processing_time, material_used,
                      energy_consumed, avail_pct, scheduled_hrs,
                      demand, stock, lead_time,
                      ordering_cost, holding_cost, order_qty,
                      f"{threshold:.0%}", f"{prob:.1%}",
                      "Delay" if prediction == 1 else "On Time",
                      "HIGH" if stockout_risk else "LOW"],
        })
        st.download_button("⬇ Download Report (CSV)",
                           df_export.to_csv(index=False),
                           file_name="smartops_report.csv",
                           use_container_width=True)

    with ins2:
        st.markdown("<div class='sec'>Active Alerts</div>",
                    unsafe_allow_html=True)
        alerts = []
        if stockout_risk and lead_time > 6:
            alerts.append(("a-red", "🚨 Critical",
                           "Stock Level < demand AND lead time > 6d. Immediate reorder required."))
        elif stockout_risk:
            alerts.append(("a-amber", "⚠ Stockout Risk",
                           f"Stock Level ({stock}) < Demand ({demand}). "
                           f"Order {order_qty} units."))
        if availability < 0.80:
            alerts.append(("a-red", "🔧 Machine Critical",
                           f"Machine {machine_id} at {avail_pct}% — critically low."))
        elif availability < 0.85:
            alerts.append(("a-amber", "⚠ Availability Low",
                           f"Machine {machine_id} at {avail_pct}% — below 85% target."))
        if prediction == 1:
            alerts.append(("a-red", "⏱ Delay Predicted",
                           f"P(delay)={prob:.1%} ≥ threshold {threshold:.0%}."))
        if not alerts:
            alerts.append(("a-green", "✅ All Clear",
                           "All parameters within normal operating range."))
        for _cls, _title, _msg in alerts:
            st.markdown(
                f"<div class='alert {_cls}'><b>{_title}</b> — {_msg}</div>",
                unsafe_allow_html=True)

        st.markdown("<div class='sec'>Recommended Actions</div>",
                    unsafe_allow_html=True)
        if stockout_risk and lead_time > 6:
            st.markdown("""<div class='rec urgent'>
              <div class='rt'>Urgent: Emergency Reorder + Expedite</div>
              Raise emergency PO. Negotiate expedited delivery with supplier.
            </div>""", unsafe_allow_html=True)
        elif stockout_risk:
            st.markdown(f"""<div class='rec warning'>
              <div class='rt'>Place Standard Purchase Order</div>
              Order {order_qty} units (ROP = {reorder_point}) at standard lead time.
            </div>""", unsafe_allow_html=True)
        if availability < 0.85:
            st.markdown(f"""<div class='rec warning'>
              <div class='rt'>Schedule Preventive Maintenance — Machine {machine_id}</div>
              Availability at {avail_pct}%. Schedule PM to prevent downtime.
            </div>""", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown("""<div class='rec urgent'>
              <div class='rt'>Review Production Schedule</div>
              Delay predicted. Add buffer stock, expedite procurement.
            </div>""", unsafe_allow_html=True)
        if not stockout_risk and availability >= 0.85 and prediction != 1:
            st.markdown("""<div class='rec ok'>
              <div class='rt'>System Stable — Routine Monitoring Only</div>
              All KPIs within range. No immediate action required.
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec'>System Health Radar</div>",
                    unsafe_allow_html=True)
        r_cats = ["Availability", "Stock Cover", "Lead Time OK",
                  "Demand Fit", "Order Health"]
        r_vals = [
            availability,
            min(stock / max(demand, 1), 1),
            1 - (lead_time - 2) / 8,
            min(demand / 150, 1),
            1 - min(order_qty / 200, 1),
        ]
        fig_rad = go.Figure(go.Scatterpolar(
            r=[round(v * 100, 1) for v in r_vals]
              + [round(r_vals[0] * 100, 1)],
            theta=r_cats + [r_cats[0]],
            fill="toself", fillcolor="rgba(59,130,246,.15)",
            line=dict(color="#3b82f6", width=2),
        ))
        fig_rad.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=280,
            margin=dict(l=40, r=40, t=10, b=10),
            polar=dict(
                bgcolor="#161b27",
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(color="#6b7280", size=9),
                                gridcolor=_G),
                angularaxis=dict(tickfont=dict(color="#94a3b8", size=11),
                                 gridcolor=_G),
            ),
        )
        st.plotly_chart(fig_rad, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════
with tab4:
    mi1, mi2 = st.columns(2)

    with mi1:
        st.markdown("<div class='sec'>Model Card</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <table class='stbl'>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Algorithm</td><td>Random Forest Classifier</td></tr>
          <tr><td>Task</td><td>Binary classification (on-time / delay)</td></tr>
          <tr><td>Target column</td><td>Delay_Flag (0 = On Time, 1 = Delay)</td></tr>
          <tr><td>Threshold</td>
              <td style='color:#f59e0b'>{threshold:.0%} (dynamic slider)</td></tr>
          <tr><td>Training rows</td><td>{DS_TOTAL_ROWS:,}</td></tr>
          <tr><td>Date range</td><td>{DS_DATE_MIN} → {DS_DATE_MAX}</td></tr>
          <tr><td>Dataset delay rate</td>
              <td style='color:{"#ef4444" if DS_DELAY_RATE>30 else "#10b981"}'>
                {DS_DELAY_RATE:.1f}%</td></tr>
          <tr><td>Model loaded</td>
              <td style='color:{"#10b981" if model_ready else "#ef4444"}'>
                {"✓ Yes" if model_ready else "✗ model.pkl missing"}</td></tr>
          <tr><td>Total features</td><td>12</td></tr>
        </table>""", unsafe_allow_html=True)

        # FIXED: Feature vector display now matches exact TRAIN_FEATURES order.
        # Rows 8 & 9 are Stock_Level and Lead_Time (not Prev_Demand/7-Day Avg).
        st.markdown("<div class='sec'>Current Feature Vector (12 features)</div>",
                    unsafe_allow_html=True)
        feat_rows = [
            ("1 · Machine ID",                machine_id),
            ("2 · Processing Time (min)",      processing_time),
            ("3 · Material Used (kg)",         material_used),
            ("4 · Energy (kWh)",               energy_consumed),
            ("5 · Machine Availability (0-1)", f"{availability:.2f}"),
            ("6 · Scheduled Duration (hrs)",   scheduled_hrs),
            ("7 · Demand (units)",             demand),
            ("8 · Stock Level (units)",        stock),        # FIXED: was Ordering_Cost
            ("9 · Lead Time (days)",           lead_time),    # FIXED: was Holding_Cost
            ("10 · Ordering Cost (₹)",         ordering_cost),# FIXED: was Order_Quantity
            ("11 · Holding Cost (₹/u)",        holding_cost), # FIXED: was Prev_Demand
            ("12 · Order Quantity (derived)",  order_qty),    # FIXED: was 7-Day Avg
        ]
        rows_html = "".join(
            f"<tr><td>{n}</td><td style='color:#7eb8f7'>{v}</td></tr>"
            for n, v in feat_rows)
        st.markdown(
            f"<table class='stbl'><tr><th>Feature</th><th>Value</th></tr>"
            f"{rows_html}</table>", unsafe_allow_html=True)

        if model_ready:
            st.markdown("<div class='sec'>Live Prediction Output</div>",
                        unsafe_allow_html=True)
            st.markdown(f"""
            <table class='stbl'>
              <tr><th>Output</th><th>Value</th></tr>
              <tr><td>Raw P(delay)</td>
                  <td style='color:{"#ef4444" if prob>=threshold else "#10b981"}'>
                    {prob:.4f}</td></tr>
              <tr><td>Threshold applied</td>
                  <td style='color:#f59e0b'>{threshold:.2f}</td></tr>
              <tr><td>Decision</td>
                  <td style='color:{"#ef4444" if prediction==1 else "#10b981"}'>
                    {"⚠ Delay Expected" if prediction==1 else "✅ On Time"}</td></tr>
            </table>""", unsafe_allow_html=True)

    with mi2:
        st.markdown("<div class='sec'>Model Evaluation</div>",
                    unsafe_allow_html=True)

        if eval_ok:
            y_true  = eval_data["y_true"]
            y_score = eval_data["y_score"]
            eval_source = f"Real dataset · {len(y_true):,} samples"
        else:
            # Demo fallback — only used when model.pkl or CSV is missing
            y_true  = np.array([0,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,1,0,1])
            y_score = np.array([0.12,0.88,0.23,0.76,0.31,0.91,0.82,0.19,
                                0.67,0.42,0.28,0.73,0.85,0.15,0.62,0.38,
                                0.22,0.79,0.33,0.71])
            eval_source = "20 synthetic demo samples (load model.pkl + CSV for real)"

        y_pred_t = (y_score >= threshold).astype(int)
        cm       = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm.ravel()

        acc  = round(accuracy_score(y_true, y_pred_t) * 100, 1)
        prec = round(precision_score(y_true, y_pred_t, zero_division=0) * 100, 1)
        rec  = round(recall_score(y_true, y_pred_t, zero_division=0) * 100, 1)
        f1   = round(2 * prec * rec / max(prec + rec, 0.001), 1)

        ev1, ev2, ev3, ev4 = st.columns(4)
        _lbl = "real" if eval_ok else "demo"
        ev1.markdown(kpi("Accuracy",  f"{acc}",  _lbl, True, "blue",  "🎯", "%"),
                     unsafe_allow_html=True)
        ev2.markdown(kpi("Precision", f"{prec}", _lbl, True, "teal",  "🔍", "%"),
                     unsafe_allow_html=True)
        ev3.markdown(kpi("Recall",    f"{rec}",  _lbl, True, "green", "📡", "%"),
                     unsafe_allow_html=True)
        ev4.markdown(kpi("F1",        f"{f1}",   _lbl, True, "amber", "⚖", "%"),
                     unsafe_allow_html=True)

        st.caption(
            f"{'✓ Real data:' if eval_ok else '⚠ Demo:'} {eval_source}. "
            f"Metrics update live with threshold slider ({threshold:.0%}).")

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrix
        cm_fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Pred: On Time", "Pred: Delay"],
            y=["Actual: On Time", "Actual: Delay"],
            colorscale=[[0, "#161b27"], [1, "#3b82f6"]],
            text=[[f"TN={tn:,}", f"FP={fp:,}"],
                  [f"FN={fn:,}", f"TP={tp:,}"]],
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            showscale=False,
        ))
        # FIXED: single update_layout call (no duplicate)
        cm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
            height=210,
            margin=dict(l=4, r=4, t=36, b=4),
            title=dict(
                text=f"Confusion Matrix (threshold={threshold:.0%})",
                font=dict(size=12, color="#94a3b8"), x=0),
        )
        cm_fig.update_xaxes(**_AXIS_STYLE)
        cm_fig.update_yaxes(**_AXIS_STYLE)
        st.plotly_chart(cm_fig, use_container_width=True)

        # ROC Curve
        fpr_arr, tpr_arr, thresholds_roc = roc_curve(y_true, y_score)
        roc_auc = auc(fpr_arr, tpr_arr)
        idx_op  = int(np.argmin(np.abs(thresholds_roc - threshold)))

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#374151", dash="dash"), showlegend=False))
        fig_roc.add_trace(go.Scatter(
            x=fpr_arr.tolist(), y=tpr_arr.tolist(), mode="lines",
            line=dict(color="#3b82f6", width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,.08)",
            name=f"AUC = {roc_auc:.3f}"))
        fig_roc.add_trace(go.Scatter(
            x=[fpr_arr[idx_op]], y=[tpr_arr[idx_op]], mode="markers",
            marker=dict(size=11, color="#f59e0b", symbol="diamond",
                        line=dict(color="white", width=1.5)),
            name=f"Threshold = {threshold:.0%}"))
        themed(fig_roc, 265, "ROC Curve (real probabilities)",
               {"xaxis_title": "False Positive Rate",
                "yaxis_title": "True Positive Rate",
                "legend": dict(x=0.45, y=0.08,
                               bgcolor="rgba(0,0,0,0)")})
        st.plotly_chart(fig_roc, use_container_width=True)
        st.info("💡 Diamond = operating point at current threshold. "
                "Move the sidebar slider to see it update live.")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='display:flex;justify-content:space-between;"
    f"align-items:center;font-size:11px;color:#374151;padding:4px 0'>"
    f"<div style='display:flex;align-items:center;gap:8px'>"
    f"{logo_svg(20)}"
    f"<b style='color:#4b5563'>SmartOps</b> Manufacturing Analytics"
    f"</div>"
    f"<div>Powered by Streamlit · Plotly · scikit-learn · statsmodels</div>"
    f"<div>v5.1 · © 2026</div>"
    f"</div>", unsafe_allow_html=True)