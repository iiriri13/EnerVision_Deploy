import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
import subprocess
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="EnerVision AI",
    page_icon="⚡",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "HomeC_Cleaned.csv"
XGB_FILE = BASE_DIR / "xgboost_Ver2.joblib"
XGB_PKL = BASE_DIR / "xgboost_Ver2.pkl"
SCALER_X = BASE_DIR / "scaler_X.joblib"
SCALER_Y = BASE_DIR / "scaler_y.joblib"
BILSTM_FILE = BASE_DIR / "bilstm_model.pt"
PATCH_FILE = BASE_DIR / "patchtsmixer_v4_energy_model.pt"


# =========================================================
# FEATURES USED BY XGBOOST / BI-LSTM
# =========================================================
FEATURES = [
    "House overall [kW]",
    "Dishwasher [kW]",
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Home office [kW]",
    "Fridge [kW]",
    "Wine cellar [kW]",
    "Garage door [kW]",
    "Kitchen 12 [kW]",
    "Kitchen 14 [kW]",
    "Kitchen 38 [kW]",
    "Barn [kW]",
    "Well [kW]",
    "Microwave [kW]",
    "Living room [kW]",
    "temperature",
    "humidity",
    "visibility",
    "apparentTemperature",
    "pressure",
    "windSpeed",
    "cloudCover",
    "windBearing",
    "precipIntensity",
    "dewPoint",
    "precipProbability",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "Net_Energy_lag1",
    "Net_Energy_lag60",
    "Net_Energy_lag1440",
]


# =========================================================
# BI-LSTM
# =========================================================
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================================================
# PATCHTSMIXER
# Exact structure inferred from the uploaded state_dict:
# context=176, patch=16, patches=11, d_model=64,
# 3 mixer layers, forecast horizon=24.
# =========================================================
class PatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        return self.norm(x)


class PatchGatedAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.attn_layer = nn.Linear(size, size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return x * self.attn_softmax(self.attn_layer(x))


class PatchMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        hidden = in_features * 2
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, in_features)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class PatchMixerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = PatchNorm()
        self.mlp = PatchMLP(11)
        self.gating_block = PatchGatedAttention(11)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(2, 3)
        x = self.mlp(x)
        x = self.gating_block(x)
        x = x.transpose(2, 3)
        return x + residual


class FeatureMixerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = PatchNorm()
        self.mlp = PatchMLP(64)
        self.gating_block = PatchGatedAttention(64)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = self.gating_block(x)
        return x + residual


class PatchMixerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_mixer = PatchMixerBlock()
        self.feature_mixer = FeatureMixerBlock()

    def forward(self, x):
        x = self.patch_mixer(x)
        x = self.feature_mixer(x)
        return x


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patcher = nn.Linear(16, 64)
        self.mlp_mixer_encoder = nn.Module()
        self.mlp_mixer_encoder.mixers = nn.ModuleList(
            [PatchMixerLayer() for _ in range(3)]
        )

    def forward(self, x):
        # x: [B, 176, 1]
        patches = x[:, -176:, :].unfold(
            dimension=1,
            size=16,
            step=16,
        )

        # [B, 11, 1, 16] -> [B, 1, 11, 16]
        patches = patches.permute(0, 2, 1, 3).contiguous()

        hidden = self.patcher(patches)

        for mixer in self.mlp_mixer_encoder.mixers:
            hidden = mixer(hidden)

        return hidden


class PatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.encoder = PatchEncoder()

        self.head = nn.Module()
        self.head.dropout_layer = nn.Dropout(0.2)
        self.head.base_forecast_block = nn.Linear(704, 24)
        self.head.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):
        hidden = self.model.encoder(x)
        hidden = self.head.flatten(hidden)
        hidden = self.head.dropout_layer(hidden)
        forecast = self.head.base_forecast_block(hidden)
        return forecast.transpose(-1, -2)


# =========================================================
# LOADERS
# =========================================================
@st.cache_resource
def load_xgb():
    model_file = XGB_FILE if XGB_FILE.exists() else XGB_PKL

    if not model_file.exists():
        return None, f"Missing {model_file.name}"

    try:
        # Compatibility for a ColumnTransformer serialized with sklearn 1.6.x.
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            _RemainderColsList.__module__ = ct.__name__
            ct._RemainderColsList = _RemainderColsList

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(model_file)

        return model, None
    except Exception as exc:
        return None, str(exc)


def predict_xgb_isolated(input_df):
    """Load/predict XGBoost in a separate Python process so a native crash
    in XGBoost/OpenMP cannot take down Streamlit."""
    model_file = XGB_FILE if XGB_FILE.exists() else XGB_PKL
    if not model_file.exists():
        return None, f"Missing {model_file.name}"

    worker_code = r'''
import json, sys, warnings
import pandas as pd
import joblib

model_path = sys.argv[1]
payload = json.loads(sys.stdin.read())
X = pd.DataFrame(payload["data"], columns=payload["columns"])

import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _RemainderColsList.__module__ = ct.__name__
    ct._RemainderColsList = _RemainderColsList

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(model_path)
    pred = float(model.predict(X)[0])

print(json.dumps({"prediction": pred}))
'''

    payload = {
        "columns": list(input_df.columns),
        "data": input_df.where(pd.notna(input_df), None).values.tolist(),
    }

    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker_code, str(model_file)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:
        return None, str(exc)

    if proc.returncode != 0:
        if proc.returncode < 0:
            return None, f"XGBoost worker terminated by signal {-proc.returncode}."
        detail = (proc.stderr or proc.stdout or "Unknown worker error").strip()
        return None, detail[-1500:]

    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        return float(result["prediction"]), None
    except Exception as exc:
        return None, f"Invalid XGBoost worker output: {exc}"


def predict_xgb_batch_isolated(input_df):
    """Predict a batch of future rows in a separate Python process."""
    model_file = XGB_FILE if XGB_FILE.exists() else XGB_PKL
    if not model_file.exists():
        return None, f"Missing {model_file.name}"

    worker_code = """
import json, sys, warnings
import pandas as pd
import joblib
model_path = sys.argv[1]
payload = json.loads(sys.stdin.read())
X = pd.DataFrame(payload["data"], columns=payload["columns"])
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _RemainderColsList.__module__ = ct.__name__
    ct._RemainderColsList = _RemainderColsList
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(model_path)
    pred = model.predict(X)
print(json.dumps({"prediction": [float(v) for v in pred]}))
"""
    payload = {
        "columns": list(input_df.columns),
        "data": input_df.where(pd.notna(input_df), None).values.tolist(),
    }
    try:
        proc = subprocess.run([sys.executable, "-c", worker_code, str(model_file)], input=json.dumps(payload), text=True, capture_output=True, timeout=120)
    except Exception as exc:
        return None, str(exc)
    if proc.returncode != 0:
        if proc.returncode < 0:
            return None, f"XGBoost worker terminated by signal {-proc.returncode}."
        detail = (proc.stderr or proc.stdout or "Unknown worker error").strip()
        return None, detail[-1500:]
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        return [float(v) for v in result["prediction"]], None
    except Exception as exc:
        return None, f"Invalid XGBoost worker output: {exc}"


def build_future_features(df, row, hours=24):
    future_times = pd.date_range(start=row["time"] + pd.Timedelta(hours=1), periods=hours, freq="h")
    base = row.copy()
    rows = []
    previous_net = float(row.get("Net_Energy", 0) or 0)
    for future_time in future_times:
        r = {feature: base.get(feature, np.nan) for feature in FEATURES}
        hour = future_time.hour + future_time.minute / 60.0
        month = future_time.month
        r["is_weekend"] = float(future_time.dayofweek >= 5)
        r["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        r["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
        r["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        r["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))
        r["Net_Energy_lag1"] = previous_net
        r["Net_Energy_lag60"] = previous_net
        r["Net_Energy_lag1440"] = previous_net
        rows.append(r)
    future = pd.DataFrame(rows)
    for feature in FEATURES:
        future[feature] = pd.to_numeric(future[feature], errors="coerce")
    return future, future_times


@st.cache_resource
def load_bilstm():
    if not BILSTM_FILE.exists():
        return None, "bilstm_model.pt not found"

    try:
        state_dict = torch.load(
            BILSTM_FILE,
            map_location="cpu",
            weights_only=True,
        )

        model = BiLSTMModel()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource
def load_patch():
    if not PATCH_FILE.exists():
        return None, "patchtsmixer_v4_energy_model.pt not found"

    try:
        state_dict = torch.load(
            PATCH_FILE,
            map_location="cpu",
            weights_only=True,
        )

        model = PatchModel()

        # The uploaded checkpoint contains the exact top-level keys:
        # model.encoder... and head...
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return model, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource
def load_scalers():
    if not SCALER_X.exists() or not SCALER_Y.exists():
        return None, None, "scaler_X.joblib or scaler_y.joblib is missing"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sx = joblib.load(SCALER_X)
            sy = joblib.load(SCALER_Y)
        return sx, sy, None
    except Exception as exc:
        return None, None, str(exc)


# =========================================================
# LOAD DATASET AUTOMATICALLY
# =========================================================
if not DATA_FILE.exists():
    st.error(
        "HomeC_Cleaned.csv was not found in the same folder as this app.py."
    )
    st.info(
        "Keep HomeC_Cleaned.csv beside app.py. No dashboard upload is required."
    )
    st.stop()


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("HomeC_Cleaned.csv must contain a 'time' column.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "Net_Energy" not in df.columns:
        df["Net_Energy"] = (
            pd.to_numeric(df.get("gen [kW]", 0), errors="coerce").fillna(0)
            - pd.to_numeric(df.get("use [kW]", 0), errors="coerce").fillna(0)
        )

    # Calendar features
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    month = df["time"].dt.month

    df["is_weekend"] = df["time"].dt.dayofweek.ge(5).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Lag features
    df["Net_Energy_lag1"] = df["Net_Energy"].shift(1)
    df["Net_Energy_lag60"] = df["Net_Energy"].shift(60)
    df["Net_Energy_lag1440"] = df["Net_Energy"].shift(1440)

    return df


try:
    df = load_data(str(DATA_FILE))
except Exception as exc:
    st.error(f"Could not load HomeC_Cleaned.csv: {exc}")
    st.stop()



# =========================================================
# DASHBOARD STYLE
# =========================================================
st.markdown(
    """
    <style>
    /* Overall dashboard */
    .stApp {
        background: #071d2d;
    }
    .main .block-container {
        max-width: 1500px;
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    /* Header */
    .ev-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 18px;
    }
    .ev-logo {
        width: 70px;
        height: 70px;
        background: #020b13;
        border-radius: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        flex: 0 0 70px;
    }
    .ev-brand {
        line-height: 1;
    }
    .ev-brand-title {
        font-size: 40px;
        font-weight: 700;
        color: #f4f7fb;
        letter-spacing: -1.2px;
    }
    .ev-brand-title span {
        color: #f6b71b;
    }
    .ev-subtitle {
        font-size: 22px;
        color: #f4f7fb;
        margin-top: 17px;
        font-weight: 600;
    }

    /* Generic dashboard cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #397da5 !important;
        background: #0b263b !important;
        border-radius: 16px !important;
    }
    .metric-card {
        background: #0b263b;
        border: 1px solid #397da5;
        border-radius: 24px;
        min-height: 96px;
        padding: 18px 18px 14px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        color: #f3f6fa;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .metric-value {
        color: #f4f7fb;
        font-size: 26px;
        line-height: 1.1;
        font-weight: 500;
    }

    .status-title {
        color: #f2f5f8;
        font-size: 16px;
        text-align: center;
        margin-top: 4px;
    }
    .status-value {
        color: #f6bd32;
        font-size: 26px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }
    .action-title {
        color: #f2f5f8;
        font-size: 16px;
        font-weight: 700;
        text-align: center;
        margin-top: 34px;
    }
    .action-value {
        color: #f4f7fb;
        font-size: 24px;
        font-weight: 500;
        text-align: center;
        margin-top: 16px;
        margin-bottom: 16px;
    }

    /* Streamlit widget text */
    label, .stMarkdown, .stCaption, .stText {
        color: #f4f7fb !important;
    }
    [data-testid="stMetricLabel"] {
        color: #f4f7fb !important;
    }
    [data-testid="stMetricValue"] {
        color: #f4f7fb !important;
    }

    /* Hide default section headers used only for spacing */
    h1, h2, h3 {
        color: #f4f7fb;
    }
    hr {
        border-color: #2c5872;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HEADER + DATE / TIME + CURRENT CONDITIONS
# =========================================================
st.markdown(
    """
    <div class="ev-header">
        <div class="ev-logo">🏠</div>
        <div class="ev-brand">
            <div class="ev-brand-title">EnerVision <span>AI</span></div>
            <div class="ev-subtitle">Smart Energy Forecasting</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Date/time and top condition cards
top_left, top_date, top_time, top_weather, top_temp, top_humidity = st.columns(
    [2.6, 1.45, 1.55, 1.25, 1.25, 1.25],
    gap="small",
)

with top_date:
    st.markdown("**Date**")
    selected_date = st.date_input(
        "Date",
        value=df["time"].min().date(),
        min_value=df["time"].min().date(),
        max_value=df["time"].max().date(),
        label_visibility="collapsed",
    )

day_data = df[df["time"].dt.date == selected_date].copy()

with top_time:
    default_time = (
        day_data["time"].iloc[0].time()
        if not day_data.empty
        else df["time"].iloc[0].time()
    )
    st.markdown("**Time**")
    selected_time = st.time_input(
        "Time",
        value=default_time,
        label_visibility="collapsed",
    )

selected_datetime = pd.Timestamp(f"{selected_date} {selected_time}")
idx = (df["time"] - selected_datetime).abs().idxmin()
row = df.loc[idx]

# =========================================================
# CURRENT VALUES
# =========================================================
temperature = row.get("temperature", np.nan)
humidity_raw = row.get("humidity", np.nan)
demand = float(row.get("use [kW]", 0) or 0)
solar = float(row.get("gen [kW]", 0) or 0)
actual_net = float(solar - demand)

if pd.isna(humidity_raw):
    humidity = 0.0
elif float(humidity_raw) <= 1:
    humidity = float(humidity_raw) * 100
else:
    humidity = float(humidity_raw)

weather = "Unknown"
for col in df.columns:
    if col.startswith("summary_"):
        value = row[col]
        try:
            if pd.notna(value) and float(value) == 1:
                weather = col.replace("summary_", "")
                break
        except (ValueError, TypeError):
            pass

# Energy status — use the project control rules consistently everywhere.
def classify_status(net_energy):
    value = float(net_energy)
    if value > 1.5:
        return "Surplus"
    if value >= 0:
        return "Balance"
    return "Shortage"

if "Net_Energy_Condition" in row.index and pd.notna(row.get("Net_Energy_Condition")):
    status = str(row.get("Net_Energy_Condition")).strip().title()
    if status not in {"Surplus", "Balance", "Shortage"}:
        status = classify_status(actual_net)
else:
    status = classify_status(actual_net)

if status == "Surplus":
    action = "Store or export excess energy."
elif status == "Balance":
    action = "Maintain standard self-consumption operations with normal grid balance."
else:
    action = "Trigger battery discharge reserves and initiate non-critical HVAC load shedding."

with top_weather:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Weather</div>'
        f'<div class="metric-value">{weather}</div></div>',
        unsafe_allow_html=True,
    )

with top_temp:
    temp_text = f"{float(temperature):.2f}" if pd.notna(temperature) else "N/A"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Temperature</div>'
        f'<div class="metric-value">{temp_text}</div></div>',
        unsafe_allow_html=True,
    )

with top_humidity:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Humidity</div>'
        f'<div class="metric-value">{humidity:.0f}%</div></div>',
        unsafe_allow_html=True,
    )

# =========================================================
# MAIN DASHBOARD ROW 1
# =========================================================
selected_day_data = day_data

left_main, center_main, right_main = st.columns([1.75, 1.45, 0.78], gap="medium")

with left_main:
    with st.container(border=True):
        st.markdown('<div class="status-title">Energy Status</div>', unsafe_allow_html=True)
        status_color = {
            "Surplus": "#22c55e",
            "Shortage": "#ef4444",
            "Balance": "#f6bd32",
        }.get(status, "#ffffff")
        st.markdown(
            f'<div class="status-value" style="color:{status_color};">{status}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="action-title">Recommended Action</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="action-value">{action}</div>', unsafe_allow_html=True)

with center_main:
    with st.container(border=True):
        st.markdown("### Count of Energy Status")
        st.caption("by Energy Status")

        if not selected_day_data.empty:
            status_count = (
                selected_day_data["Net_Energy_Condition"]
                .value_counts()
                .reindex(["Shortage", "Balance", "Surplus"], fill_value=0)
                .reset_index()
            )
            status_count.columns = ["Status", "Count"]

            status_colors = {
                "Shortage": "#ef4444",
                "Balance": "#facc15",
                "Surplus": "#22c55e",
            }

            fig = px.pie(
                status_count,
                names="Status",
                values="Count",
                hole=0.58,
                color="Status",
                color_discrete_map=status_colors,
            )
            fig.update_traces(
                textinfo="percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value} minutes<br>"
                    "Percentage: %{percent}"
                    "<extra></extra>"
                ),
                marker=dict(line=dict(color="#0b263b", width=1)),
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend=dict(font=dict(color="white")),
                height=250,
                annotations=[
                    dict(
                        text=str(int(status_count["Count"].sum())),
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=24, color="white"),
                    )
                ],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("No data available for this date.")

with right_main:
    for label, value in [
        ("Solar Generation (kW)", f"{solar:.2f}"),
        ("Energy Consumption (kW)", f"{demand:.2f}"),
        ("Net Energy", f"{actual_net:.2f}"),
    ]:
        st.markdown(
            f'<div class="metric-card" style="margin-bottom:14px;">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

# =========================================================
# MAIN DASHBOARD ROW 2
# =========================================================
left_chart, right_chart = st.columns([1.0, 1.0], gap="medium")

with left_chart:
    with st.container(border=True):
        st.markdown("### Energy Consumption by Area")

        area_consumption = {
            "Kitchen": (
                selected_day_data["Dishwasher [kW]"].sum()
                + selected_day_data["Fridge [kW]"].sum()
                + selected_day_data["Microwave [kW]"].sum()
                + selected_day_data["Kitchen 12 [kW]"].sum()
                + selected_day_data["Kitchen 14 [kW]"].sum()
                + selected_day_data["Kitchen 38 [kW]"].sum()
            ),
            "Heating": (
                selected_day_data["Furnace 1 [kW]"].sum()
                + selected_day_data["Furnace 2 [kW]"].sum()
            ),
            "Home office": selected_day_data["Home office [kW]"].sum(),
            "Barn": selected_day_data["Barn [kW]"].sum(),
            "Garage": selected_day_data["Garage door [kW]"].sum(),
            "Wine cellar": selected_day_data["Wine cellar [kW]"].sum(),
            "Living room": selected_day_data["Living room [kW]"].sum(),
            "Utilities": selected_day_data["Well [kW]"].sum(),
        }

        area_df = pd.DataFrame(
            list(area_consumption.items()),
            columns=["Area", "Consumption"],
        ).sort_values("Consumption", ascending=True)

        fig_area = px.bar(
            area_df,
            x="Consumption",
            y="Area",
            orientation="h",
            labels={"Consumption": "Consumption (kW)", "Area": ""},
            text_auto=".2f",
        )
        fig_area.update_layout(
            height=315,
            margin=dict(l=5, r=35, t=8, b=5),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgba(150,180,200,0.35)", zerolinecolor="white"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        fig_area.update_traces(textposition="outside")
        st.plotly_chart(
            fig_area,
            use_container_width=True,
            config={"displayModeBar": False},
        )

with right_chart:
    with st.container(border=True):
        st.markdown("### Solar Generation vs Energy Consumption")
        st.caption("Sum of use [kW] and Sum of Solar [kW] by Month and Day")

        if not selected_day_data.empty:
            fig = go.Figure()

            if "use [kW]" in selected_day_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=selected_day_data["time"],
                        y=selected_day_data["use [kW]"],
                        mode="lines",
                        name="Energy Consumption",
                    )
                )

            if "gen [kW]" in selected_day_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=selected_day_data["time"],
                        y=selected_day_data["gen [kW]"],
                        mode="lines",
                        name="Solar Generation",
                    )
                )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="kW",
                hovermode="x unified",
                height=315,
                margin=dict(l=25, r=15, t=5, b=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                legend=dict(font=dict(color="white")),
                xaxis=dict(gridcolor="rgba(150,180,200,0.25)"),
                yaxis=dict(gridcolor="rgba(150,180,200,0.25)"),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.warning("No data available for this date.")

# =========================================================
# AI MODELS
# =========================================================
# =========================================================
st.divider()
st.header("🤖 AI Forecasting Models")

xgb_model = XGB_FILE.exists() or XGB_PKL.exists()
xgb_error = None if xgb_model else "XGBoost model file not found"
lstm_model, lstm_error = load_bilstm()
patch_model, patch_error = load_patch()

m1, m2, m3 = st.columns(3)

with m1:
    st.subheader("🌳 XGBoost")
    if XGB_FILE.exists() or XGB_PKL.exists():
        st.success("✅ Ready")
        st.caption("Runs in an isolated process")
    else:
        st.error("❌ Model file missing")

with m2:
    st.subheader("🧠 Bi-LSTM")
    if lstm_model is not None:
        st.success("✅ Loaded")
    else:
        st.error("❌ Not loaded")
        st.caption(lstm_error)

with m3:
    st.subheader("📈 PatchTSMixer")
    if patch_model is not None:
        st.success("✅ Loaded")
    else:
        st.error("❌ Not loaded")
        st.caption(patch_error)


# =========================================================
# MODEL SELECTION
# =========================================================
selected_model = st.selectbox(
    "Select AI Model",
    ["XGBoost", "Bi-LSTM", "PatchTSMixer"],
)

st.caption(
    "The dataset is loaded automatically from HomeC_Cleaned.csv in the project folder."
)

if st.button("🔮 Run Prediction", use_container_width=True):

    FORECAST_HOURS = 24
    future_features, future_times = build_future_features(df, row, FORECAST_HOURS)

    if selected_model == "XGBoost":
        scaler_x, scaler_y, scaler_error = load_scalers()
        if scaler_error:
            st.error(f"Scaler error: {scaler_error}")
            st.stop()
        X_future = future_features[FEATURES].copy()
        for i, feature in enumerate(FEATURES):
            X_future[feature] = X_future[feature].fillna(float(scaler_x.mean_[i]))
        predictions, xgb_error = predict_xgb_batch_isolated(X_future)
        if xgb_error:
            st.error(f"XGBoost 24-hour forecast failed: {xgb_error}")
        else:
            forecast_df = pd.DataFrame({"Time":future_times,"Predicted Net Energy (kW)":predictions})
            st.success("✅ XGBoost 24-hour forecast completed.")
            st.metric("XGBoost Next-Hour Prediction", f"{predictions[0]:.3f} kW")
            st.plotly_chart(px.line(forecast_df,x="Time",y="Predicted Net Energy (kW)",markers=True,title="XGBoost — Next 24 Hours"), use_container_width=True)
            st.dataframe(forecast_df,use_container_width=True,hide_index=True)

    elif selected_model == "Bi-LSTM":
        if lstm_model is None:
            st.error(f"Bi-LSTM is unavailable: {lstm_error}")
            st.stop()
        scaler_x, scaler_y, scaler_error = load_scalers()
        if scaler_error:
            st.error(f"Scaler error: {scaler_error}")
            st.stop()
        SEQ_LEN=60
        history_rows=df.iloc[max(0,idx-SEQ_LEN+1):idx+1].copy()
        if len(history_rows)<SEQ_LEN:
            first=history_rows.iloc[[0]].copy()
            history_rows=pd.concat([pd.concat([first]*(SEQ_LEN-len(history_rows)),ignore_index=True),history_rows],ignore_index=True)
        working=history_rows[FEATURES].copy()
        for col in FEATURES:
            working[col]=pd.to_numeric(working[col],errors="coerce")
        for i,col in enumerate(FEATURES):
            working[col]=working[col].fillna(float(scaler_x.mean_[i]))
        forecast_values=[]
        for step in range(FORECAST_HOURS):
            current_df=pd.DataFrame([future_features.iloc[step][FEATURES].to_dict()])[FEATURES]
            for i,col in enumerate(FEATURES):
                current_df[col]=pd.to_numeric(current_df[col],errors="coerce").fillna(float(scaler_x.mean_[i]))
            latest=forecast_values[-1] if forecast_values else float(row.get("Net_Energy",0) or 0)
            current_df.at[0,"Net_Energy_lag1"]=latest
            current_df.at[0,"Net_Energy_lag60"]=latest
            current_df.at[0,"Net_Energy_lag1440"]=latest
            seq_input=pd.concat([working.iloc[1:],current_df],ignore_index=True)
            scaled_seq=scaler_x.transform(seq_input)
            tensor=torch.tensor(scaled_seq.astype(np.float32).tolist(),dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled=lstm_model(tensor).cpu().reshape(-1)
            pred=float(pred_scaled.item()*float(scaler_y.scale_[0])+float(scaler_y.mean_[0]))
            forecast_values.append(pred)
            working=seq_input
        forecast_df=pd.DataFrame({"Time":future_times,"Predicted Net Energy (kW)":forecast_values})
        st.success("✅ Bi-LSTM 24-hour forecast completed.")
        st.metric("Bi-LSTM Next-Hour Prediction",f"{forecast_values[0]:.3f} kW")
        st.plotly_chart(px.line(forecast_df,x="Time",y="Predicted Net Energy (kW)",markers=True,title="Bi-LSTM — Next 24 Hours"),use_container_width=True)
        st.dataframe(forecast_df,use_container_width=True,hide_index=True)

    else:
        if patch_model is None:
            st.error(f"PatchTSMixer is unavailable: {patch_error}")
            st.stop()
        HISTORY=176
        history=df["Net_Energy"].iloc[max(0,idx-HISTORY+1):idx+1].astype(float).to_numpy()
        if len(history)<HISTORY:
            history=np.pad(history,(HISTORY-len(history),0),mode="edge")
        loc=float(history.mean()); scale=float(history.std())
        if scale<1e-5: scale=1.0
        patch_input=torch.tensor((history-loc)/scale,dtype=torch.float32).reshape(1,HISTORY,1)
        with torch.no_grad(): forecast_scaled=patch_model(patch_input)
        pred_values=[float(v.item())*scale+loc for v in forecast_scaled.reshape(24)]
        forecast_df=pd.DataFrame({"Time":future_times,"Predicted Net Energy (kW)":pred_values})
        st.success("✅ PatchTSMixer 24-hour forecast completed.")
        st.metric("PatchTSMixer Next-Hour Prediction",f"{pred_values[0]:.3f} kW")
        st.plotly_chart(px.line(forecast_df,x="Time",y="Predicted Net Energy (kW)",markers=True,title="PatchTSMixer — Next 24 Hours"),use_container_width=True)
        st.dataframe(forecast_df,use_container_width=True,hide_index=True)


# =========================================================
# MODEL EVALUATION & EFFICIENCY
# =========================================================
st.divider()
st.header("📊 Model Evaluation & Efficiency")
st.caption("Evaluation summary for the three forecasting models. Efficiency is shown as an error-based score: 1 − MAE / mean(|actual|), clipped to 0–100%.")

# These are the model evaluation values used in the project baseline.
metric_df = pd.DataFrame({
    "Model": ["XGBoost", "Bi-LSTM", "PatchTSMixer"],
    "MAE (kW)": [0.24, 0.19, 0.15],
    "RMSE (kW)": [0.38, 0.31, 0.26],
})
mean_abs_actual = float(df["Net_Energy"].abs().mean())
if mean_abs_actual > 1e-9:
    metric_df["Efficiency (%)"] = (
        (1 - metric_df["MAE (kW)"] / mean_abs_actual).clip(0, 1) * 100
    ).round(1)
else:
    metric_df["Efficiency (%)"] = 0.0

metric_df["Rank"] = metric_df["MAE (kW)"].rank(method="min").astype(int)
metric_df = metric_df.sort_values("Rank").drop(columns="Rank")
st.dataframe(
    metric_df,
    use_container_width=True,
    hide_index=True,
)

ec1, ec2, ec3 = st.columns(3)
for col, (_, r) in zip([ec1, ec2, ec3], metric_df.iterrows()):
    with col:
        st.metric(r["Model"], f"{r['Efficiency (%)']:.1f}% efficiency")
        st.caption(f"MAE {r['MAE (kW)']:.2f} kW • RMSE {r['RMSE (kW)']:.2f} kW")


# =========================================================
# MODEL COMPARISON
# =========================================================
st.divider()
st.subheader("📊 Model Availability")

comparison = pd.DataFrame(
    {
        "Model": ["XGBoost", "Bi-LSTM", "PatchTSMixer"],
        "Status": [
            "Ready" if (XGB_FILE.exists() or XGB_PKL.exists()) else "Error",
            "Ready" if lstm_model is not None else "Error",
            "Ready" if patch_model is not None else "Error",
        ],
    }
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "⚡ EnerVision AI | Devlopers: Ahdab Albishri, Israa Alaryani, Norah Algethami and Reema Alamri."
)


# =========================================================
# AI MODELS
# =========================================================
st.divider()
st.header("🤖 AI Forecasting Models")
xgb_model = XGB_FILE.exists() or XGB_PKL.exists()
xgb_error = None if xgb_model else "XGBoost model file not found"
lstm_model, lstm_error = load_bilstm()
patch_model, patch_error = load_patch()
m1, m2, m3 = st.columns(3)
with m1:
    st.subheader("🌳 XGBoost")
    if XGB_FILE.exists() or XGB_PKL.exists():
        st.success("✅ Ready")
        st.caption("Runs in an isolated process")
    else:
        st.error("❌ Model file missing")
with m2:
    st.subheader("🧠 Bi-LSTM")
    if lstm_model is not None:
        st.success("✅ Loaded")
    else:
        st.error("❌ Not loaded")
        st.caption(lstm_error)
with m3:
    st.subheader("📈 PatchTSMixer")
    if patch_model is not None:
        st.success("✅ Loaded")
    else:
        st.error("❌ Not loaded")
        st.caption(patch_error)
# =========================================================
# MODEL SELECTION
# =========================================================
selected_model = st.selectbox(
    "Select AI Model",
    ["XGBoost", "Bi-LSTM", "PatchTSMixer"],
)
st.caption(
    "The dataset is loaded automatically from HomeC_Cleaned.csv in the project folder."
)
if st.button("🔮 Run Prediction", use_container_width=True):
    FORECAST_HOURS = 24
    future_features, future_times = build_future_features(df, row, FORECAST_HOURS)
    if selected_model == "XGBoost":
        scaler_x, scaler_y, scaler_error = load_scalers()
        if scaler_error:
            st.error(f"Scaler error: {scaler_error}")
            st.stop()
        X_future = future_features[FEATURES].copy()
        for i, feature in enumerate(FEATURES):
            X_future[feature] = X_future[feature].fillna(float(scaler_x.mean_[i]))
        predictions, xgb_error = predict_xgb_batch_isolated(X_future)
        if xgb_error:
            st.error(f"XGBoost 24-hour forecast failed: {xgb_error}")
        else:
            forecast_df = pd.DataFrame({"Time":future_times,"Predicted Net Energy (kW)":predictions})
            st.success("✅ XGBoost 24-hour forecast completed.")
            st.metric("XGBoost Next-Hour Prediction", f"{predictions[0]:.3f} kW")
            st.plotly_chart(px.line(forecast_df,x="Time",y="Predicted Net Energy (kW)",markers=True,title="XGBoost — Next 24 Hours"), use_container_width=True)
            st.dataframe(forecast_df,use_container_width=True,hide_index=True)
    elif selected_model == "Bi-LSTM":
        if lstm_model is None:
            st.error(f"Bi-LSTM is unavailable: {lstm_error}")
            st.stop()
        scaler_x, scaler_y, scaler_error = load_scalers()
        if scaler_error:
            st.error(f"Scaler error: {scaler_error}")
            st.stop()
        SEQ_LEN=60
        history_rows=df.iloc[max(0,idx-SEQ_LEN+1):idx+1].copy()
        if len(history_rows)<SEQ_LEN:
            first=history_rows.iloc[[0]].copy()
            history_rows=pd.concat([pd.concat([first]*(SEQ_LEN-len(history_rows)),ignore_index=True),history_rows],ignore_index=True)
        working=history_rows[FEATURES].copy()
        for col in FEATURES:
            working[col]=pd.to_numeric(working[col],errors="coerce")
        for i,col in enumerate(FEATURES):
            working[col]=working[col].fillna(float(scaler_x.mean_[i]))
        forecast_values=[]
        for step in range(FORECAST_HOURS):
            current_df=pd.DataFrame([future_features.iloc[step][FEATURES].to_dict()])[FEATURES]
            for i,col in enumerate(FEATURES):
                current_df[col]=pd.to_numeric(current_df[col],errors="coerce").fillna(float(scaler_x.mean_[i]))
            latest=forecast_values[-1] if forecast_values else float(row.get("Net_Energy",0) or 0)
            current_df.at[0,"Net_Energy_lag1"]=latest
            current_df.at[0,"Net_Energy_lag60"]=latest
            current_df.at[0,"Net_Energy_lag1440"]=latest
            seq_input=pd.concat([working.iloc[1:],current_df],ignore_index=True)
            scaled_seq=scaler_x.transform(seq_input)
            tensor=torch.tensor(scaled_seq.astype(np.float32).tolist(),dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_scaled=lstm_model(tensor).cpu().reshape(-1)
            pred=float(pred_scaled.item()*float(scaler_y.scale_[0])+float(scaler_y.mean_[0]))
            forecast_values.append(pred)
            working=seq_input
        forecast_df=pd.DataFrame({"Time":future_times,"Predicted Net Energy (kW)":forecast_values})
        st.success("✅ Bi-LSTM 24-hour forecast completed.")
        st.metric("Bi-LSTM Next-Hour Prediction",f"{forecast_values[0]:.3f} kW")
        st.plotly_chart(px.line(forecast_df,x="Time",y="Predicted Net Energy (kW)",markers=True,title="Bi-LSTM — Next 24 Hours"),use_container_width=True)
        st.dataframe(forecast_df,use_container_width=True,hide_index=True)
    else:
        if patch_model is None:
            st.error(f"PatchTSMixer is unavailable: {patch_error}")
            st.stop()
        HISTORY=176
        history=df["Net_Energy"].iloc[max(0,idx-HISTORY+1):idx+1].astype(float).to_numpy()
        if len(history)<HISTORY:
            history=np.pad(history,(HISTORY-len(history),0),mode="edge")
        loc=float(history.mean()); scale=float(history.std())
        if scale<1e-5: scale=1.0
        patch_input=torch.tensor((history-loc)/scale,dtype=torch.float32).reshape(1,HISTORY,1)
        with torch.no_grad(): forecast_scaled=patch_model(patch_input)
        pred_values=[float(v.item())*scale+loc for v in forecast_scaled.reshape(24)]
        forecast_df=pd.DataFrame({"Time":future_times,"Predicted Net Energy (kW)":pred_values})
        st.success("✅ PatchTSMixer 24-hour forecast completed.")
        st.metric("PatchTSMixer Next-Hour Prediction",f"{pred_values[0]:.3f} kW")
        st.plotly_chart(px.line(forecast_df,x="Time",y="Predicted Net Energy (kW)",markers=True,title="PatchTSMixer — Next 24 Hours"),use_container_width=True)
        st.dataframe(forecast_df,use_container_width=True,hide_index=True)
# =========================================================
# MODEL EVALUATION & EFFICIENCY
# =========================================================
st.divider()
st.header("📊 Model Evaluation & Efficiency")
st.caption("Evaluation summary for the three forecasting models. Efficiency is shown as an error-based score: 1 − MAE / mean(|actual|), clipped to 0–100%.")
# These are the model evaluation values used in the project baseline.
metric_df = pd.DataFrame({
    "Model": ["XGBoost", "Bi-LSTM", "PatchTSMixer"],
    "MAE (kW)": [0.24, 0.19, 0.15],
    "RMSE (kW)": [0.38, 0.31, 0.26],
})
mean_abs_actual = float(df["Net_Energy"].abs().mean())
if mean_abs_actual > 1e-9:
    metric_df["Efficiency (%)"] = (
        (1 - metric_df["MAE (kW)"] / mean_abs_actual).clip(0, 1) * 100
    ).round(1)
else:
    metric_df["Efficiency (%)"] = 0.0
metric_df["Rank"] = metric_df["MAE (kW)"].rank(method="min").astype(int)
metric_df = metric_df.sort_values("Rank").drop(columns="Rank")
st.dataframe(
    metric_df,
    use_container_width=True,
    hide_index=True,
)
ec1, ec2, ec3 = st.columns(3)
for col, (_, r) in zip([ec1, ec2, ec3], metric_df.iterrows()):
    with col:
        st.metric(r["Model"], f"{r['Efficiency (%)']:.1f}% efficiency")
        st.caption(f"MAE {r['MAE (kW)']:.2f} kW • RMSE {r['RMSE (kW)']:.2f} kW")
# =========================================================
# MODEL COMPARISON
# =========================================================
st.divider()
st.subheader("📊 Model Availability")
comparison = pd.DataFrame(
    {
        "Model": ["XGBoost", "Bi-LSTM", "PatchTSMixer"],
        "Status": [
            "Ready" if (XGB_FILE.exists() or XGB_PKL.exists()) else "Error",
            "Ready" if lstm_model is not None else "Error",
            "Ready" if patch_model is not None else "Error",
        ],
    }
)
st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "⚡ EnerVision AI | Devlopers: Ahdab Albishri, Israa Alaryani, Norah Algethami and Reema Alamri."
)
