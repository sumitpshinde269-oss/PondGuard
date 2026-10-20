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

st.warning(
    "⚠️ **Scientific Advisory & Simulation Disclaimer:** PondGuard is an educational decision-support prototype. "
    "The AI model is trained on a simulated benchmark aquaculture dataset (Veeramsetty et al., 4,300 samples) "
    "and **has not yet been validated on live farm telemetry**. "
    "Predictions are probabilistic advisory estimates, not guaranteed outcomes. "
    "Always verify pond conditions with calibrated physical test kits or probes before taking major farm management actions."
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

# Robust session state syncing for quick preset changes
if "current_preset" not in st.session_state:
    st.session_state["current_preset"] = selected_preset

if selected_preset != st.session_state["current_preset"]:
    st.session_state["current_preset"] = selected_preset
    if preset_data:
        st.session_state["f_temp"] = float(preset_data["temp"])
        st.session_state["f_do"] = float(preset_data["do"])
        st.session_state["f_ph"] = float(preset_data["ph"])
        st.session_state["f_nh3"] = float(preset_data["nh3"])
        st.session_state["f_no2"] = float(preset_data["no2"])
        st.session_state["f_obs"] = str(preset_data["obs"])

if "f_temp" not in st.session_state:
    st.session_state["f_temp"] = 25.0
    st.session_state["f_do"] = 4.5
    st.session_state["f_ph"] = 7.8
    st.session_state["f_nh3"] = 0.010
    st.session_state["f_no2"] = 0.010
    st.session_state["f_obs"] = ""

st.sidebar.header("Water Measurements")
temperature = st.sidebar.number_input(
    "Water temperature (°C)", min_value=0.0, max_value=50.0,
    key="f_temp", step=0.5,
    help="How warm or cold the water is. Ideal range for most freshwater fish: 20°C to 32°C."
)
dissolved_oxygen = st.sidebar.number_input(
    "Dissolved oxygen (mg/L)", min_value=0.0, max_value=30.0,
    key="f_do", step=0.1,
    help="Oxygen available for fish to breathe. Healthy: > 5.0 mg/L. Danger: < 3.0 mg/L."
)
ph = st.sidebar.number_input(
    "pH (Water Acidity)", min_value=0.0, max_value=14.0,
    key="f_ph", step=0.1,
    help="Measures how acidic or alkaline water is. Ideal: 6.5 to 8.5. Extreme pH harms fish skin & gills."
)
ammonia = st.sidebar.number_input(
    "Total ammonia (mg/L)", min_value=0.0, max_value=100.0,
    key="f_nh3", step=0.01, format="%.3f",
    help="Fish excrement and leftover feed. Safe: < 0.10 mg/L. Danger: > 0.50 mg/L."
)
nitrite = st.sidebar.number_input(
    "Nitrite (mg/L)", min_value=0.0, max_value=100.0,
    key="f_no2", step=0.01, format="%.3f",
    help="Toxic breakdown of ammonia causing blood suffocation. Safe: < 0.10 mg/L. Danger: > 0.50 mg/L."
)
observations = st.sidebar.text_area(
    "Fish or pond observations",
    key="f_obs",
    placeholder="Example: fish gasping near surface, not eating",
    help="Note any visible signs of distress, swimming habits, or mortalities."
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
        "temperature_c": float(temperature),
        "dissolved_oxygen_mg_l": float(dissolved_oxygen),
        "pH": float(ph),
        "total_ammonia_mg_l": float(ammonia),
        "nitrite_mg_l": float(nitrite),
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
    ml_pred = ml_state.get("prediction")
    ml_err = ml_state.get("error")

    st.divider()
    st.header("Pond Monitoring Dashboard")
    st.caption(
        f"Farm: **{reading['farm']}**  |  Pond: **{reading['pond']}**  |  "
        f"Checked: **{reading['timestamp']}**"
    )

    # 1. Overall Status Indicator
    has_critical_alert = any(level == "HIGH" for level, _, _ in alerts)
    is_poor_prediction = ml_pred and ml_pred.get("class_id") == 2
    has_medium_alert = any(level == "MEDIUM" for level, _, _ in alerts)
    is_good_prediction = ml_pred and ml_pred.get("class_id") == 1

    if has_critical_alert or is_poor_prediction:
        st.error("### 🔴 OVERALL POND STATUS: URGENT ACTION NEEDED\nHigh risk to fish health detected. Check aerators, cease feeding, and follow the action plan below.")
    elif has_medium_alert or is_good_prediction:
        st.warning("### 🟡 OVERALL POND STATUS: MODERATE CAUTION\nMild water quality stress or parameter drift detected. Increase monitoring frequency.")
    else:
        st.success("### 🟢 OVERALL POND STATUS: ALL CLEAR (HEALTHY)\nAll measured parameters and AI classifications indicate optimal pond conditions.")

    # 2. Plain-Language Parameter Statuses
    temp_val = reading["temperature_c"]
    do_val = reading["dissolved_oxygen_mg_l"]
    ph_val = reading["pH"]
    nh3_val = reading["total_ammonia_mg_l"]
    no2_val = reading["nitrite_mg_l"]

    temp_badge = "🟢 Ideal (20–32°C)" if 20.0 <= temp_val <= 32.0 else "🟡 Watch (Outside 20–32°C)"
    if do_val >= 5.0:
        do_badge = "🟢 Healthy (>5.0 mg/L)"
    elif do_val >= 3.0:
        do_badge = "🟡 Low (3–5 mg/L)"
    else:
        do_badge = "🔴 Danger: Hypoxia (<3 mg/L)"

    ph_badge = "🟢 Ideal (6.5–8.5)" if 6.5 <= ph_val <= 8.5 else "🟡 Watch (Outside 6.5–8.5)"

    if nh3_val <= 0.10:
        nh3_badge = "🟢 Safe (<0.10 mg/L)"
    elif nh3_val <= 0.50:
        nh3_badge = "🟡 Elevated (0.10–0.50 mg/L)"
    else:
        nh3_badge = "🔴 Danger: Toxic (>0.50 mg/L)"

    if no2_val <= 0.10:
        no2_badge = "🟢 Safe (<0.10 mg/L)"
    elif no2_val <= 0.50:
        no2_badge = "🟡 Elevated (0.10–0.50 mg/L)"
    else:
        no2_badge = "🔴 Danger: Toxic (>0.50 mg/L)"

    st.subheader("💧 Water Quality at a Glance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{temp_val:.1f} °C")
    c1.caption(f"Status: **{temp_badge}**")

    c2.metric("Dissolved Oxygen", f"{do_val:.1f} mg/L")
    c2.caption(f"Status: **{do_badge}**")

    c3.metric("pH (Acidity)", f"{ph_val:.1f}")
    c3.caption(f"Status: **{ph_badge}**")

    c4, c5 = st.columns(2)
    c4.metric("Total Ammonia (Waste)", f"{nh3_val:.3f} mg/L")
    c4.caption(f"Status: **{nh3_badge}**")

    c5.metric("Nitrite (Toxic Byproduct)", f"{no2_val:.3f} mg/L")
    c5.caption(f"Status: **{no2_badge}**")

    # 3. Farmer Practical Action Plan
    st.subheader("🧑‍🌾 Suggested Next Steps for Farm Operators")
    st.caption("Practical, safe steps to maintain fish welfare based on today's readings:")

    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown("**⚡ Aeration & Oxygen Management**")
        if do_val < 3.0:
            st.error("🚨 **Turn on all aerators immediately!** If power is off, pump water into the air or splash surface water vigorously to dissolve oxygen and prevent fish suffocation.")
        elif do_val < 5.0:
            st.warning("⚠️ **Increase nighttime aeration.** Algae consume oxygen in darkness, causing early dawn oxygen drops. Run aerators continuously from midnight to sunrise.")
        else:
            st.info("✅ **Oxygen is healthy.** Maintain regular daytime and nighttime aeration cycles.")

        st.markdown("**🍽️ Feeding Adjustments**")
        if nh3_val > 0.1 or no2_val > 0.1 or do_val < 5.0:
            st.warning("🛑 **Reduce feed by 50% or skip feeding today.** Uneaten feed and digestive waste spike toxic ammonia and consume dissolved oxygen.")
        else:
            st.info("✅ **Normal feeding.** Feed according to normal schedule, confirming fish consume all feed within 20 minutes.")

    with act_col2:
        st.markdown("**🚰 Water Exchange & Filtration**")
        if nh3_val > 0.5 or no2_val > 0.5:
            st.error("🚰 **Partial Water Exchange (20–30%):** Drain and replace 20% to 30% of pond volume with clean, well-aerated source water. Avoid draining more than 30% at once to prevent temperature shock.")
        else:
            st.info("✅ **Water exchange not urgently needed.** Maintain routine pond depth and inspect biofilters.")

        st.markdown("**🩺 Fish Behavior & Health Monitoring**")
        if any(level == "HIGH" and "Fish-behavior" in title for level, title, _ in alerts):
            st.error("🔍 **Inspect Fish Stock:** Fish distress reported. Look for surface piping, clamped fins, or body spots. Isolate distressed fish and consult your local aquaculture extension officer.")
        else:
            st.info("✅ **No physical distress reported.** Continue routine daily health checks during feeding.")

    # 4. AI Water Quality Assessment (Machine Learning Model)
    st.subheader("🤖 AI Water-Quality Prediction")
    st.caption(
        "Multi-variate pattern classification trained on the Mendeley Aquaculture Dataset (Veeramsetty et al., 4,300 simulated samples). "
        "Notice: This model has not yet been validated on real farm data; predictions are probabilistic decision-support estimates. "
        "Kept strictly independent of the deterministic safety alerts below."
    )

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

    # 5. Threshold Alerts
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

    # 6. Export Buttons
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


