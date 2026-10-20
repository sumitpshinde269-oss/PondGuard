import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aquaculture_model.pkl"

TARGET_NAMES = {
    0: "Excellent Quality",
    1: "Good Quality",
    2: "Poor Quality"
}

@st.cache_resource
def load_aquaculture_model():
    """Load the trained aquaculture Gradient Boosting pipeline with caching."""
    if not MODEL_PATH.exists():
        return None, f"Model artifact '{MODEL_PATH.name}' not found. Please train it first."
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error loading model artifact: {e}"

st.set_page_config(
    page_title="PondGuard | Aquaculture Alerts",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 PondGuard")
st.subheader("Early Water-Quality Alerts for Small Aquaculture Farms")
st.write(
    "Monitor pond conditions, identify potential water-quality risks, "
    "and record observations for follow-up."
)

st.info(
    "This prototype uses illustrative screening thresholds. "
    "Adjust them for your fish species and local aquaculture guidance. "
    "It does not diagnose fish disease or replace professional testing."
)

st.sidebar.header("Farm Details")
farm_name = st.sidebar.text_input("Farm name", "My Fish Farm")
pond_name = st.sidebar.text_input("Pond name or ID", "Pond 1")

st.sidebar.header("Demo Quick Presets")
preset_options = {
    "Custom / Manual Input": None,
    "🟢 Optimal Pond (Excellent)": {"temp": 24.6, "do": 4.2, "ph": 8.3, "nh3": 0.01, "no2": 0.001, "obs": "Fish active and feeding well"},
    "🟡 Routine Pond (Good)": {"temp": 33.7, "do": 7.8, "ph": 6.1, "nh3": 0.04, "no2": 1.87, "obs": "Normal morning activity"},
    "🔴 Critical Risk (Poor)": {"temp": 12.3, "do": 1.8, "ph": 5.1, "nh3": 0.21, "no2": 4.55, "obs": "Fish gasping near surface, sluggish swimming"},
}
selected_preset = st.sidebar.selectbox("Load Example Scenario", list(preset_options.keys()), index=0)
preset_data = preset_options[selected_preset]

st.sidebar.header("Water Measurements")
init_temp = preset_data["temp"] if preset_data else 25.0
init_do = preset_data["do"] if preset_data else 4.5
init_ph = preset_data["ph"] if preset_data else 7.8
init_amm = preset_data["nh3"] if preset_data else 0.01
init_nit = preset_data["no2"] if preset_data else 0.01
init_obs = preset_data["obs"] if preset_data else ""

temperature = st.sidebar.number_input(
    "Water temperature (°C)", min_value=0.0, max_value=50.0,
    value=float(init_temp), step=0.5
)
dissolved_oxygen = st.sidebar.number_input(
    "Dissolved oxygen (mg/L)", min_value=0.0, max_value=30.0,
    value=float(init_do), step=0.1
)
ph = st.sidebar.number_input(
    "pH", min_value=0.0, max_value=14.0,
    value=float(init_ph), step=0.1
)
ammonia = st.sidebar.number_input(
    "Total ammonia (mg/L)", min_value=0.0, max_value=100.0,
    value=float(init_amm), step=0.01, format="%.3f"
)
nitrite = st.sidebar.number_input(
    "Nitrite (mg/L)", min_value=0.0, max_value=100.0,
    value=float(init_nit), step=0.01, format="%.3f"
)
observations = st.sidebar.text_area(
    "Fish or pond observations",
    value=init_obs,
    placeholder="Example: fish gasping near the surface"
)

if st.sidebar.button("Analyze Pond", type="primary"):
    alerts = []

    if dissolved_oxygen < 3:
        alerts.append(("HIGH", "Dissolved oxygen is very low",
                       "Check aeration immediately and verify the reading."))
    elif dissolved_oxygen < 5:
        alerts.append(("MEDIUM", "Dissolved oxygen is low",
                       "Check aeration and monitor fish behavior."))

    if ph < 6.5 or ph > 8.5:
        alerts.append(("MEDIUM", "pH is outside the illustrative 6.5–8.5 range",
                       "Repeat the measurement and consult species-specific guidance."))

    if temperature < 20 or temperature > 32:
        alerts.append(("MEDIUM", "Temperature is outside the illustrative 20–32°C range",
                       "Check species requirements and monitor temperature."))

    if ammonia > 0.5:
        alerts.append(("HIGH", "Total ammonia reading is elevated",
                       "Verify the test and seek aquaculture guidance promptly."))
    elif ammonia > 0.1:
        alerts.append(("MEDIUM", "Total ammonia reading needs attention",
                       "Check water conditions and confirm the test result."))

    if nitrite > 0.5:
        alerts.append(("HIGH", "Nitrite reading is elevated",
                       "Verify the test and seek aquaculture guidance promptly."))
    elif nitrite > 0.1:
        alerts.append(("MEDIUM", "Nitrite reading needs attention",
                       "Check biofiltration, monitor aeration, and re-test."))

    if observations.strip():
        keywords = ["gasping", "lethargic", "not eating", "unusual swimming",
                    "dead fish", "lesion", "wound", "spots"]
        if any(word in observations.lower() for word in keywords):
            alerts.append((
                "HIGH",
                "Fish-behavior or health concern reported",
                "Inspect the fish and contact an aquaculture professional. "
                "This text check is not a disease diagnosis."
            ))

    # Run Machine Learning Inference using the 5 core measurements
    model, model_err = load_aquaculture_model()
    ml_prediction = None
    if model is not None:
        try:
            sample_df = pd.DataFrame([{
                "temperature": float(temperature),
                "dissolved_oxygen": float(dissolved_oxygen),
                "ph": float(ph),
                "ammonia": float(ammonia),
                "nitrite": float(nitrite)
            }])
            pred_class = int(model.predict(sample_df)[0])
            prob_dist = model.predict_proba(sample_df)[0]
            ml_prediction = {
                "class_id": pred_class,
                "class_name": TARGET_NAMES.get(pred_class, f"Class {pred_class}"),
                "confidence": float(prob_dist[pred_class]),
                "probabilities": {
                    TARGET_NAMES.get(i, f"Class {i}"): float(prob_dist[i])
                    for i in range(len(prob_dist))
                }
            }
        except Exception as e:
            model_err = f"Prediction failed: {e}"

    st.session_state["pondguard_ml"] = {
        "prediction": ml_prediction,
        "error": model_err
    }

    current_reading = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "farm": farm_name,
        "pond": pond_name,
        "temperature_c": temperature,
        "dissolved_oxygen_mg_l": dissolved_oxygen,
        "pH": ph,
        "total_ammonia_mg_l": ammonia,
        "nitrite_mg_l": nitrite,
        "observations": observations,
        "alert_count": len(alerts),
        "status": "Attention needed" if alerts else "No threshold alerts",
        "ml_predicted_quality": ml_prediction["class_name"] if ml_prediction else "Unavailable",
        "ml_confidence": f"{ml_prediction['confidence']:.2%}" if ml_prediction else "N/A",
        "ml_prob_excellent": f"{ml_prediction['probabilities']['Excellent Quality']:.2%}" if ml_prediction else "N/A",
        "ml_prob_good": f"{ml_prediction['probabilities']['Good Quality']:.2%}" if ml_prediction else "N/A",
        "ml_prob_poor": f"{ml_prediction['probabilities']['Poor Quality']:.2%}" if ml_prediction else "N/A",
    }
    st.session_state["pondguard_reading"] = current_reading
    st.session_state["pondguard_alerts"] = alerts

    # Maintain cumulative session history
    if "pondguard_history" not in st.session_state:
        st.session_state["pondguard_history"] = []
    st.session_state["pondguard_history"].append(current_reading)

if "pondguard_reading" in st.session_state:
    reading = st.session_state["pondguard_reading"]
    alerts = st.session_state["pondguard_alerts"]
    ml_state = st.session_state.get("pondguard_ml", {})

    st.divider()
    st.header("Pond Monitoring Dashboard")
    st.caption(
        f"Farm: {reading['farm']}  |  Pond: {reading['pond']}  |  "
        f"Checked: {reading['timestamp']}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{reading['temperature_c']:.1f} °C")
    c2.metric("Dissolved Oxygen", f"{reading['dissolved_oxygen_mg_l']:.1f} mg/L")
    c3.metric("pH", f"{reading['pH']:.1f}")

    c4, c5 = st.columns(2)
    c4.metric("Total Ammonia", f"{reading['total_ammonia_mg_l']:.3f} mg/L")
    c5.metric("Nitrite", f"{reading['nitrite_mg_l']:.3f} mg/L")

    # AI Water Quality Assessment (Machine Learning Model)
    st.subheader("🤖 AI Water-Quality Prediction")
    st.caption(
        "Statistical pattern classification trained on the Mendeley Aquaculture Dataset (Veeramsetty et al., 4,300 samples). "
        "Evaluates multi-variate parameter correlations. Kept strictly independent of the rule-based safety alerts below."
    )
    ml_pred = ml_state.get("prediction")
    ml_err = ml_state.get("error")

    if ml_pred:
        pred_label = ml_pred["class_name"]
        pred_conf = ml_pred["confidence"]
        probs = ml_pred["probabilities"]

        if pred_label == "Excellent Quality":
            st.success(f"🟢 **Predicted Category: {pred_label}** — Optimal conditions for aquatic life (Model Confidence: {pred_conf:.1%})")
        elif pred_label == "Good Quality":
            st.info(f"🟡 **Predicted Category: {pred_label}** — Acceptable conditions; continue standard monitoring (Model Confidence: {pred_conf:.1%})")
        else:
            st.error(f"🔴 **Predicted Category: {pred_label}** — Suboptimal / stressful conditions detected (Model Confidence: {pred_conf:.1%})")

        st.caption("Model Prediction Probabilities:")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Excellent Quality", f"{probs.get('Excellent Quality', 0.0):.1%}")
        p_col1.progress(min(max(probs.get("Excellent Quality", 0.0), 0.0), 1.0))

        p_col2.metric("Good Quality", f"{probs.get('Good Quality', 0.0):.1%}")
        p_col2.progress(min(max(probs.get("Good Quality", 0.0), 0.0), 1.0))

        p_col3.metric("Poor Quality", f"{probs.get('Poor Quality', 0.0):.1%}")
        p_col3.progress(min(max(probs.get("Poor Quality", 0.0), 0.0), 1.0))
    elif ml_err:
        st.warning(f"⚠️ Machine Learning model unavailable: {ml_err}")

    st.subheader("Threshold Alert Assessment")
    st.caption(
        "Deterministic physiological safety screening based on established freshwater fish tolerance limits "
        "(e.g., hypoxia DO < 3.0 mg/L, toxic ammonia > 0.5 mg/L, nitrite > 0.5 mg/L). Independent of ML statistical predictions."
    )

    if alerts:
        for level, title, advice in alerts:
            if level == "HIGH":
                st.error(f"🔴 {level}: {title}\n\n{advice}")
            else:
                st.warning(f"🟠 {level}: {title}\n\n{advice}")
    else:
        st.success(
            "No alerts were triggered by these illustrative thresholds. "
            "This does not guarantee that the pond is safe."
        )

    st.subheader("Farm Notes")
    st.write(reading["observations"] or "No observations entered.")

    # Export Buttons
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "Download Current Reading (CSV)",
            data=pd.DataFrame([reading]).to_csv(index=False).encode("utf-8"),
            file_name="pondguard_reading.csv",
            mime="text/csv"
        )
    with col_d2:
        history_list = st.session_state.get("pondguard_history", [reading])
        st.download_button(
            f"Download Session Log ({len(history_list)} Readings) (CSV)",
            data=pd.DataFrame(history_list).to_csv(index=False).encode("utf-8"),
            file_name="pondguard_session_log.csv",
            mime="text/csv"
        )

    if len(st.session_state.get("pondguard_history", [])) > 1:
        with st.expander("📊 View Session Readings History", expanded=False):
            history_df = pd.DataFrame(st.session_state["pondguard_history"])
            display_cols = ["timestamp", "pond", "temperature_c", "dissolved_oxygen_mg_l", "pH", "total_ammonia_mg_l", "nitrite_mg_l", "ml_predicted_quality", "ml_confidence", "alert_count"]
            st.dataframe(history_df[[c for c in display_cols if c in history_df.columns]], use_container_width=True)
else:
    st.write("Enter your pond measurements in the sidebar or pick an example preset, then click **Analyze Pond**.")

