import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="EV Forecast Dashboard", layout="wide")

# === Load pre-trained forecasting model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Custom Styling ===
st.markdown("""
    <style>
        .stApp {
            background-image: url('background.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        h1, h2, h3, .stTextInput > label, .stSelectbox > label {
            color: #ffffff !important;
        }

        .block-container {
            padding: 2rem;
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
        }

        .title {
            font-size: 44px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            margin-top: 20px;
        }

        .subtitle {
            font-size: 22px;
            font-weight: 500;
            text-align: center;
            color: #f0f0f0;
            margin-bottom: 25px;
        }

        .instruction {
            font-size: 18px;
            color: #f0f0f0;
            padding-bottom: 10px;
        }

        .custom-success {
            font-size: 16px;
            color: #00ffb3;
        }
    </style>
""", unsafe_allow_html=True)

# === Title Section ===
st.markdown("<div class='title'>🔋 EV Forecast Dashboard - Washington State</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Visualizing Current Trends and Predicting the Future of Electric Vehicles</div>", unsafe_allow_html=True)

# Header Image
st.image("ev.jpg", use_container_width=True)

# === Instruction ===
st.markdown("<div class='instruction'>Choose a Washington county to explore historical trends and forecast EV adoption over the next 3 years.</div>", unsafe_allow_html=True)

# === Load and Cache Data ===
@st.cache_data

def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === County Dropdown ===
counties = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", counties)

# Handle edge case
if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

# === Filter by selected county ===
data = df[df['County'] == county].sort_values("Date")
code = data['county_encoded'].iloc[0]

# === Historical setup ===
hist = list(data['Electric Vehicle (EV) Total'].values[-6:])
cumulative = list(np.cumsum(hist))
months = data['months_since_start'].max()
last_date = data['Date'].max()

# === Forecast Generation ===
preds = []
H = 36  # forecast horizon (months)

for i in range(1, H + 1):
    next_date = last_date + pd.DateOffset(months=i)
    months += 1
    l1, l2, l3 = hist[-1], hist[-2], hist[-3]
    roll = np.mean([l1, l2, l3])
    pct1 = (l1 - l2) / l2 if l2 != 0 else 0
    pct3 = (l1 - l3) / l3 if l3 != 0 else 0
    slope = np.polyfit(range(6), cumulative[-6:], 1)[0] if len(cumulative) == 6 else 0

    features = pd.DataFrame([{
        'months_since_start': months,
        'county_encoded': code,
        'ev_total_lag1': l1,
        'ev_total_lag2': l2,
        'ev_total_lag3': l3,
        'ev_total_roll_mean_3': roll,
        'ev_total_pct_change_1': pct1,
        'ev_total_pct_change_3': pct3,
        'ev_growth_slope': slope
    }])

    prediction = model.predict(features)[0]
    preds.append({"Date": next_date, "Predicted EV Total": round(prediction)})

    hist.append(prediction)
    if len(hist) > 6:
        hist.pop(0)
    cumulative.append(cumulative[-1] + prediction)
    if len(cumulative) > 6:
        cumulative.pop(0)

# === Merge Historical + Forecast ===
hist_df = data[['Date', 'Electric Vehicle (EV) Total']].copy()
hist_df['Source'] = 'Historical'
hist_df['Cumulative EV'] = hist_df['Electric Vehicle (EV) Total'].cumsum()

future_df = pd.DataFrame(preds)
future_df['Source'] = 'Forecast'
future_df['Cumulative EV'] = future_df['Predicted EV Total'].cumsum() + hist_df['Cumulative EV'].iloc[-1]

merged = pd.concat([
    hist_df[['Date', 'Cumulative EV', 'Source']],
    future_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === Plot Forecast ===
st.subheader(f"📈 Cumulative Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, segment in merged.groupby('Source'):
    ax.plot(segment['Date'], segment['Cumulative EV'], label=label, marker='o')

ax.set_title(f"EV Growth Trend in {county} County", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)

# === Growth Summary ===
h_base = hist_df['Cumulative EV'].iloc[-1]
f_total = future_df['Cumulative EV'].iloc[-1]

if h_base > 0:
    growth = ((f_total - h_base) / h_base) * 100
    label = "increase 📈" if growth > 0 else "decrease 📉"
    st.success(f"EV registrations in **{county}** are projected to see a **{label} of {growth:.2f}%** in the next 3 years.")
else:
    st.warning("Not enough historical data for this county.")

# === Comparison Section ===
st.markdown("---")
st.header("Compare Up to 3 Counties")
selections = st.multiselect("Choose counties to compare", counties, max_selections=3)

if selections:
    series = []
    for name in selections:
        d = df[df['County'] == name].sort_values("Date")
        code = d['county_encoded'].iloc[0]
        h = list(d['Electric Vehicle (EV) Total'].values[-6:])
        c = list(np.cumsum(h))
        m = d['months_since_start'].max()
        last = d['Date'].max()

        preds = []
        for i in range(1, H + 1):
            m += 1
            future = last + pd.DateOffset(months=i)
            l1, l2, l3 = h[-1], h[-2], h[-3]
            roll = np.mean([l1, l2, l3])
            pct1 = (l1 - l2) / l2 if l2 != 0 else 0
            pct3 = (l1 - l3) / l3 if l3 != 0 else 0
            slope = np.polyfit(range(6), c[-6:], 1)[0] if len(c) == 6 else 0

            row = pd.DataFrame([{
                'months_since_start': m,
                'county_encoded': code,
                'ev_total_lag1': l1,
                'ev_total_lag2': l2,
                'ev_total_lag3': l3,
                'ev_total_roll_mean_3': roll,
                'ev_total_pct_change_1': pct1,
                'ev_total_pct_change_3': pct3,
                'ev_growth_slope': slope
            }])

            pred = model.predict(row)[0]
            preds.append({"Date": future, "Predicted EV Total": round(pred)})
            h.append(pred)
            if len(h) > 6: h.pop(0)
            c.append(c[-1] + pred)
            if len(c) > 6: c.pop(0)

        base = d[['Date', 'Electric Vehicle (EV) Total']].copy()
        base['Cumulative EV'] = base['Electric Vehicle (EV) Total'].cumsum()

        fc = pd.DataFrame(preds)
        fc['Cumulative EV'] = fc['Predicted EV Total'].cumsum() + base['Cumulative EV'].iloc[-1]

        final = pd.concat([
            base[['Date', 'Cumulative EV']],
            fc[['Date', 'Cumulative EV']]
        ], ignore_index=True)
        final['County'] = name
        series.append(final)

    plot_df = pd.concat(series)

    # Plot
    st.subheader("📊 Comparative Forecast View")
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, grp in plot_df.groupby("County"):
        ax.plot(grp['Date'], grp['Cumulative EV'], marker='o', label=name)
    ax.set_title("County-wise EV Growth Forecast", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EVs", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend()
    st.pyplot(fig)

    # Summary Stats
    summary = []
    for name in selections:
        sub = plot_df[plot_df['County'] == name].reset_index(drop=True)
        past = sub['Cumulative EV'].iloc[len(sub) - H - 1]
        future = sub['Cumulative EV'].iloc[-1]
        if past > 0:
            percent = ((future - past) / past) * 100
            summary.append(f"{name}: {percent:.2f}%")
        else:
            summary.append(f"{name}: N/A")
    st.success("Forecasted growth over 3 years → " + " | ".join(summary))

# === Footer ===
st.markdown("---")
st.markdown("<div class='custom-success'>Prepared by <strong>Riya Bajpai</strong> for the <strong>AICTE Internship Cycle 2 by S4F</strong></div>", unsafe_allow_html=True)
