import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ibadan House Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("artifacts/best_model_pipeline.joblib")
    return model

model = load_model()

st.title("🏡 Live Predictive House Prices — Ibadan")
st.caption("Localized predictive model for Ibadan's residential market")


st.sidebar.header("🏠 Property Features")

neighs = ["Agbowo", "Bodija", "GRA", "Moniya", "Ojoo", "Sango", "IbadanNorth"]
zones = ["RL", "RM", "C (Commercial)", "FV", "RH"]
land_contours = ["Lvl", "Bnk", "HLS", "Low"]

user_input = {
    "OverallQual": st.sidebar.slider("Overall Quality (1–10)", 1, 10, 6),
    "GrLivArea": st.sidebar.number_input("Above-ground Living Area (sqft)", 400, 6000, 2000),
    "Alley": st.sidebar.selectbox("Alley Access", ["Pave", "Grvl", "NA"]),
    "MSZoning": st.sidebar.selectbox("Zoning Classification", zones),
    "OverallCond": st.sidebar.slider("Overall Condition (1–10)", 1, 10, 5),
    "1stFlrSF": st.sidebar.number_input("1st Floor Area (sqft)", 300, 4000, 1000),
    "YearBuilt": st.sidebar.slider("Year Built", 1900, 2025, 2005),
    "2ndFlrSF": st.sidebar.number_input("2nd Floor Area (sqft)", 0, 3000, 500),
    "GarageCars": st.sidebar.slider("Garage Capacity", 0, 4, 2),
    "LandContour": st.sidebar.selectbox("Land Contour", land_contours),
    "PoolArea": st.sidebar.number_input("Pool Area (sqft)", 0, 1000, 0),
    "LotConfig": st.sidebar.selectbox("Lot Configuration", ["Inside", "Corner", "CulDSac", "FR2", "FR3"]),
    "Neighborhood": st.sidebar.selectbox("Original Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "BrkSide", "NridgHt"]),
    "YearRemodAdd": st.sidebar.slider("Year Remodeled/Added", 1950, 2025, 2010),
    "BsmtFullBath": st.sidebar.slider("Basement Full Baths", 0, 3, 1),
    "FullBath": st.sidebar.slider("Full Baths (Above Ground)", 1, 4, 2),
    "distance_to_ui_km": st.sidebar.number_input("Distance to UI (km)", 0.1, 25.0, 5.0),
    "flood_risk": st.sidebar.selectbox("Flood Risk (1=Yes, 0=No)", [0, 1]),
    "power_reliability": st.sidebar.slider("Power Reliability (0–1)", 0.0, 1.0, 0.7),
    "age": st.sidebar.slider("Age of Building (yrs)", 0, 100, 20)
}


user_input["price_per_sqm"] = (user_input["GrLivArea"] * 5000 + user_input["1stFlrSF"] * 10) / (user_input["GrLivArea"] + 1)
user_input["bed_bath_ratio"] = 3 / (user_input["FullBath"] + 0.5) 

input_df = pd.DataFrame([user_input])


st.write("### 🧾 Entered Property Details")
st.dataframe(input_df)

if st.button("💰 Predict House Price"):
    log_pred = model.predict(input_df)[0]
    price = np.expm1(log_pred)
    st.success(f"🏷️ Estimated Property Price: ₦{price:,.0f}")
    st.caption("Prediction localized to Ibadan housing context")


    try:
        booster = model.named_steps["model"].get_booster()
        explainer = shap.TreeExplainer(booster)
        X_trans = model.named_steps["pre"].transform(input_df)
        shap_values = explainer.shap_values(X_trans)
        feature_names = model.named_steps["pre"].get_feature_names_out()

        st.subheader("📊 Feature Contribution (SHAP)")
        shap.waterfall_plot = shap.plots.waterfall
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_trans[0],
            feature_names=feature_names
        ), show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
