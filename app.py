import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from datetime import date

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(page_title="⚡ Household Power & Bill Predictor", layout="wide")

# ------------------------------------------
# LOAD MODEL + DATA
# ------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("peak_hour_model.pkl")

@st.cache_data
def load_hourly_data():
    df = pd.read_csv("data/household_power_consumption.txt", sep=';', low_memory=False)
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df['Hour'] = df['Datetime'].dt.hour
    return df

# -----------------------------------------
# SESSION STATE
# ------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
page = st.sidebar.radio(
    "📂 Navigation",
    ["🏠 Home", "🔮 Peak Hour Predictor", "💰 KSEB Bill Calculator"],
    index=["🏠 Home", "🔮 Peak Hour Predictor", "💰 KSEB Bill Calculator"].index(st.session_state.page)
)
st.session_state.page = page

# ------------------------------------------
# HOME
# ------------------------------------------
if page == "🏠 Home":
    st.title("⚡ Smart Household Power Management System")
    st.subheader("🌍 Save Electricity – Save Earth")

    st.markdown("""
    ### 💡 Why Save Electricity?
    - Reduce electricity bills 💰  
    - Avoid peak hour overloading ⚡  
    - Promote sustainable energy 🌱  
    - Ensure future energy security ♻️  
    """)

    if st.button("👉 Check Peak Hour"):
        st.session_state.page = "🔮 Peak Hour Predictor"
        st.session_state.predicted = False
        st.rerun()

# ------------------------------------------
# PEAK HOUR PREDICTOR
# ------------------------------------------
elif page == "🔮 Peak Hour Predictor":
    st.title("🔮 Household Peak Hour Prediction")

    model = load_model()

    selected_date = st.date_input(
        "📅 Select a Date",
        min_value=date(2000, 1, 1),
        max_value=date(2050, 12, 31)
    )

    if selected_date > date.today():
        st.warning("⚠️ Future prediction – actual usage may vary.")

    if st.button("🔍 Predict Peak Hour"):
        st.session_state.predicted = True

    if st.session_state.predicted:
        with st.spinner("⏳ Predicting daily demand..."):

            # Base ML prediction (same for all days)
            hours = pd.DataFrame({'Hour': range(24)})
            base_prediction = model.predict(hours)

            # -------------------------------
            # ✅ DAILY VARIATION LOGIC
            # -------------------------------
            weekday = selected_date.weekday()  # 0=Mon, 6=Sun

            # Weekday vs Weekend adjustment
            if weekday >= 5:
                day_factor = 0.90   # Weekend (less usage)
            else:
                day_factor = 1.05   # Weekday (more usage)

            # Controlled random variation
            np.random.seed(selected_date.day)
            noise = np.random.normal(0, 0.05, 24)

            final_prediction = base_prediction * day_factor * (1 + noise)

            df_pred = pd.DataFrame({
                "Hour": range(24),
                "Predicted_Power_kW": final_prediction
            })

            peak_hours = df_pred.sort_values("Predicted_Power_kW", ascending=False).head(3)
            low_hours = df_pred.sort_values("Predicted_Power_kW").head(3)

        st.success("✅ Prediction completed using ML + daily variation")

        # Plot
        fig = px.line(
            df_pred,
            x="Hour",
            y="Predicted_Power_kW",
            markers=True,
            title=f"Predicted Power Usage on {selected_date.strftime('%d %B %Y')}",
            labels={"Hour": "Hour of Day", "Predicted_Power_kW": "Power (kW)"}
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        # Peak hours
        st.subheader("⚡ Peak Usage Hours")
        for _, row in peak_hours.iterrows():
            hr = f"{(row['Hour'] % 12) or 12} {'AM' if row['Hour'] < 12 else 'PM'}"
            st.write(f"🟥 {hr} — {row['Predicted_Power_kW']:.2f} kW")

        # Low hours
        st.subheader("💡 Best Time to Use Heavy Appliances")
        for _, row in low_hours.iterrows():
            hr = f"{(row['Hour'] % 12) or 12} {'AM' if row['Hour'] < 12 else 'PM'}"
            st.write(f"🟩 {hr} — {row['Predicted_Power_kW']:.2f} kW")

        if st.button("💰 Go to Bill Calculator"):
            st.session_state.page = "💰 KSEB Bill Calculator"
            st.session_state.predicted = False
            st.rerun()

# ------------------------------------------
# BILL CALCULATOR
# ------------------------------------------
elif page == "💰 KSEB Bill Calculator":
    st.title("💰 KSEB Smart Bill & Savings Estimator")

    st.markdown("### 🔢 Consumption Inputs")
    daily_units = st.number_input(
        "Enter average daily consumption (kWh):",
        min_value=0.0,
        step=0.1
    )

    days = st.slider("Billing period (days)", 1, 31, 30)

    peak_pct = st.slider(
        "Peak-hour usage (%)",
        0, 100, 40,
        help="Peak hours usually between 6 PM – 10 PM"
    )

    # -------------------------------
    # TOD TARIFF (Simplified)
    # -------------------------------
    peak_rate = 7.0
    normal_rate = 5.0
    offpeak_rate = 3.0

    if st.button("🧾 Calculate Smart Bill"):
        total_units = daily_units * days

        peak_units = total_units * (peak_pct / 100)
        offpeak_units = total_units - peak_units

        # ToD billing
        peak_cost = peak_units * peak_rate
        offpeak_cost = offpeak_units * offpeak_rate
        tod_bill = peak_cost + offpeak_cost

        # Normal slab billing (comparison)
        if total_units <= 50:
            slab_bill = total_units * 3.15
        elif total_units <= 100:
            slab_bill = 50 * 3.15 + (total_units - 50) * 3.70
        elif total_units <= 250:
            slab_bill = 50 * 3.15 + 50 * 3.70 + (total_units - 100) * 5.80
        else:
            slab_bill = (
                50 * 3.15 +
                50 * 3.70 +
                150 * 5.80 +
                (total_units - 250) * 6.60
            )

        # -------------------------------
        # SAVINGS SIMULATION
        # -------------------------------
        shifted_peak_pct = max(peak_pct - 20, 0)
        shifted_peak_units = total_units * (shifted_peak_pct / 100)
        shifted_offpeak_units = total_units - shifted_peak_units
        optimized_bill = (
            shifted_peak_units * peak_rate +
            shifted_offpeak_units * offpeak_rate
        )

        savings = tod_bill - optimized_bill

        # -------------------------------
        # CARBON EMISSION
        # -------------------------------
        carbon = total_units * 0.82  # kg CO₂

        # -------------------------------
        # RESULTS
        # -------------------------------
        st.success(f"💸 Time-of-Day Bill: ₹{tod_bill:.2f}")
        st.info(f"📄 Normal Slab Bill (approx): ₹{slab_bill:.2f}")

        st.markdown("### 📊 Usage Breakdown")
        df_usage = pd.DataFrame({
            "Type": ["Peak Hours", "Off-Peak Hours"],
            "Units (kWh)": [peak_units, offpeak_units]
        })
        st.bar_chart(df_usage.set_index("Type"))

        st.markdown("### 💡 Smart Savings Insight")
        st.write(f"🔁 If you shift **20% usage to off-peak hours:**")
        st.write(f"💰 New Bill: ₹{optimized_bill:.2f}")
        st.success(f"✅ You can save approx ₹{savings:.2f} per month")

        st.markdown("### 🌱 Environmental Impact")
        st.write(f"Estimated CO₂ Emission: **{carbon:.2f} kg**")

        st.markdown("### 🏠 Appliance Scheduling Tips")
        if peak_pct > 50:
            st.warning(
                "⚠️ High peak usage detected.\n\n"
                "• Run washing machine after 10 PM\n"
                "• Avoid iron box during peak hours\n"
                "• Use AC efficiently at night"
            )
        else:
            st.success(
                "✅ Good energy habits!\n\n"
                "• You are using appliances efficiently\n"
                "• Peak load on grid is reduced"
            )

        st.markdown("### 📆 Long-Term Projection")
        st.write(f"📅 Estimated yearly bill: ₹{tod_bill * 12:.2f}")
        st.write(f"🌍 Yearly CO₂ emission: {carbon * 12:.2f} kg")

    # -------------------------------
    # NAVIGATION
    # -------------------------------
    if st.button("🔙 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()