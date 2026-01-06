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
    st.title("💰 KSEB Bill Estimator")

    units = st.number_input("Enter units consumed (kWh):", min_value=0.0, step=0.1)

    if st.button("🧾 Calculate Bill"):
        if units <= 50:
            bill = units * 3.15
        elif units <= 100:
            bill = 50 * 3.15 + (units - 50) * 3.70
        elif units <= 250:
            bill = 50 * 3.15 + 50 * 3.70 + (units - 100) * 5.80
        else:
            bill = 50 * 3.15 + 50 * 3.70 + 150 * 5.80 + (units - 250) * 6.60

        st.success(f"Estimated Bill: ₹{bill:.2f}")

    # ✅ BACK TO HOME BUTTON (FIXED)
    if st.button("🔙 Back to Home"):
        st.session_state.page = "🏠 Home"
        st.rerun()
