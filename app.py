import streamlit as st
import pandas as pd
from datetime import datetime

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

st.sidebar.header("Water Measurements")
temperature = st.sidebar.number_input(
    "Water temperature (°C)", min_value=0.0, max_value=50.0,
    value=28.0, step=0.5
)
dissolved_oxygen = st.sidebar.number_input(
    "Dissolved oxygen (mg/L)", min_value=0.0, max_value=30.0,
    value=6.0, step=0.1
)
ph = st.sidebar.number_input(
    "pH", min_value=0.0, max_value=14.0,
    value=7.5, step=0.1
)
ammonia = st.sidebar.number_input(
    "Total ammonia (mg/L)", min_value=0.0, max_value=100.0,
    value=0.0, step=0.01, format="%.2f"
)
nitrite = st.sidebar.number_input(
    "Nitrite (mg/L)", min_value=0.0, max_value=100.0,
    value=0.0, step=0.01, format="%.2f"
)
observations = st.sidebar.text_area(
    "Fish or pond observations",
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

    st.session_state["pondguard_reading"] = {
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
        "status": "Attention needed" if alerts else "No threshold alerts"
    }
    st.session_state["pondguard_alerts"] = alerts

if "pondguard_reading" in st.session_state:
    reading = st.session_state["pondguard_reading"]
    alerts = st.session_state["pondguard_alerts"]

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
    c4.metric("Total Ammonia", f"{reading['total_ammonia_mg_l']:.2f} mg/L")
    c5.metric("Nitrite", f"{reading['nitrite_mg_l']:.2f} mg/L")

    st.subheader("Alert Assessment")

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

    st.download_button(
        "Download Current Reading (CSV)",
        data=pd.DataFrame([reading]).to_csv(index=False).encode("utf-8"),
        file_name="pondguard_reading.csv",
        mime="text/csv"
    )
else:
    st.write("Enter your pond measurements in the sidebar, then click **Analyze Pond**.")
