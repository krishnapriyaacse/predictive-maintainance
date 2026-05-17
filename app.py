import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

/* Labels */
label, .stNumberInput label, .stSelectbox label {
    color: #FFD700 !important;
    font-size: 20px !important;
    font-weight: bold !important;
}

/* Headers */
h1 {
    text-align: center;
    color: #4CAF50;
}

h2 {
    text-align: center;
    color: #00E5FF;
}

h3 {
    text-align: center;
    color: #FFC107;
}

/* Input Boxes */
.stNumberInput input {
    background-color: #1E1E1E;
    color: white;
    border-radius: 10px;
}

/* Select Box */
.stSelectbox div {
    color: white;
}

/* Buttons */
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #45a049;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("machine_failure_model.pkl")

# =========================
# ANOMALY MODEL
# =========================

anomaly_model = IsolationForest(
    contamination=0.1,
    random_state=42
)

dummy_data = np.random.normal(
    loc=0.5,
    scale=0.2,
    size=(200, 7)
)

anomaly_model.fit(dummy_data)

# =========================
# SESSION STATE
# =========================

if "page" not in st.session_state:
    st.session_state.page = 1

# =========================
# PAGE 1
# =========================

if st.session_state.page == 1:

    st.markdown("""
    <h1>
    🏭 AI-Based Predictive Maintenance System
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3>
    Intelligent Industrial Machine Monitoring
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.success("✅ Failure Prediction")
    st.success("✅ Anomaly Detection")
    st.success("✅ Remaining Useful Life Estimation")
    st.success("✅ Real-Time AI Monitoring")

    st.markdown("---")

    if st.button("🚀 Start System"):

        st.session_state.page = 2
        st.rerun()

# =========================
# PAGE 2
# =========================

elif st.session_state.page == 2:

    st.markdown("""
    <h1>
    🌡 Temperature Details
    </h1>
    """, unsafe_allow_html=True)

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        value=310.0
    )

    st.progress(25)

    if st.button("➡️ Next"):

        st.session_state.air_temp = air_temp
        st.session_state.process_temp = process_temp

        st.session_state.page = 3
        st.rerun()

# =========================
# PAGE 3
# =========================

elif st.session_state.page == 3:

    st.markdown("""
    <h1>
    ⚙️ Machine Performance
    </h1>
    """, unsafe_allow_html=True)

    rot_speed = st.number_input(
        "Rotational Speed (rpm)",
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0
    )

    st.progress(50)

    if st.button("➡️ Next"):

        st.session_state.rot_speed = rot_speed
        st.session_state.torque = torque

        st.session_state.page = 4
        st.rerun()

# =========================
# PAGE 4
# =========================

elif st.session_state.page == 4:

    st.markdown("""
    <h1>
    🛠 Wear & Machine Details
    </h1>
    """, unsafe_allow_html=True)

    tool_wear = st.number_input(
        "Tool Wear (min)",
        value=100
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M"]
    )

    st.progress(75)

    if st.button("🔍 Predict Machine Condition"):

        st.session_state.tool_wear = tool_wear
        st.session_state.machine_type = machine_type

        st.session_state.page = 5
        st.rerun()

# =========================
# PAGE 5
# =========================

elif st.session_state.page == 5:

    st.markdown("""
    <h1>
    📊 AI Result Dashboard
    </h1>
    """, unsafe_allow_html=True)

    st.progress(100)

    air_temp = st.session_state.air_temp
    process_temp = st.session_state.process_temp
    rot_speed = st.session_state.rot_speed
    torque = st.session_state.torque
    tool_wear = st.session_state.tool_wear
    machine_type = st.session_state.machine_type

    type_L = 1 if machine_type == "L" else 0
    type_M = 1 if machine_type == "M" else 0

    features = np.array([[
        air_temp,
        process_temp,
        rot_speed,
        torque,
        tool_wear,
        type_L,
        type_M
    ]])

    # =========================
    # ANOMALY DETECTION
    # =========================

    st.markdown("## 🧠 Anomaly Detection")

    anomaly_input = features / np.max(features)

    anomaly_result = anomaly_model.predict(
        anomaly_input
    )

    if anomaly_result[0] == -1:

        st.error(
            "🚨 Abnormal Sensor Pattern Detected!"
        )

    else:

        st.success(
            "✅ Sensor Data is Normal"
        )

    st.markdown("---")

    # =========================
    # FAILURE RISK
    # =========================

    probability = model.predict_proba(
        features
    )[0][1] * 100

    st.markdown("## 📈 Failure Risk Assessment")

    st.metric(
        "Failure Risk (%)",
        f"{probability:.2f}%"
    )

    if probability < 30:

        st.success(
            "🟢 LOW RISK - Machine Operating Normally"
        )

    elif probability < 60:

        st.warning(
            "🟡 MEDIUM RISK - Preventive Maintenance Recommended"
        )

    else:

        st.error(
            "🔴 HIGH RISK - Immediate Maintenance Required"
        )

    st.markdown("---")

    # =========================
    # RUL
    # =========================

    st.markdown("## ⏳ Remaining Useful Life")

    base_life = 200

    wear_factor = tool_wear / 300
    torque_factor = torque / 100
    risk_factor = probability / 100

    life_reduction = (
        wear_factor * 0.4 +
        torque_factor * 0.3 +
        risk_factor * 0.3
    )

    remaining_life = base_life * (
        1 - life_reduction
    )

    if remaining_life < 0:
        remaining_life = 0

    st.metric(
        "Remaining Useful Life (Hours)",
        f"{remaining_life:.1f} hrs"
    )

    st.markdown("---")

    # =========================
    # TORQUE SIMULATION
    # =========================

    st.markdown("## 📊 Torque Impact Simulation")

    torque_values = np.linspace(
        10,
        120,
        30
    )

    risk_values = []

    for t in torque_values:

        sim_features = np.array([[
            air_temp,
            process_temp,
            rot_speed,
            t,
            tool_wear,
            type_L,
            type_M
        ]])

        sim_prob = model.predict_proba(
            sim_features
        )[0][1] * 100

        risk_values.append(sim_prob)

    fig, ax = plt.subplots()

    ax.plot(
        torque_values,
        risk_values,
        marker='o'
    )

    ax.set_ylim(0, 100)

    ax.set_xlabel("Torque (Nm)")
    ax.set_ylabel("Failure Risk (%)")

    ax.set_title(
        "Torque vs Failure Risk"
    )

    st.pyplot(fig)

    st.markdown("---")

    # =========================
    # FINAL MESSAGE
    # =========================

    st.success(
        "✅ Hybrid AI Analysis Completed Successfully"
    )

    if st.button("🔄 Restart System"):

        st.session_state.page = 1
        st.rerun()