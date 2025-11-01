import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import hashlib

st.set_page_config(page_title="Ibadan House Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("artifacts/best_model_pipeline.joblib")

model = load_model()

st.title("🏡 Live Predictive House Prices — Ibadan")
st.caption("Localized predictive model for Ibadan's residential market")

st.sidebar.header("🏠 Property Features")

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
    "BedroomAbvGr": st.sidebar.slider("Bedrooms Above Ground", 1, 10, 3),
    "distance_to_ui_km": st.sidebar.number_input("Distance to UI (km)", 0.1, 25.0, 5.0),
    "flood_risk": st.sidebar.selectbox("Flood Risk (1=Yes, 0=No)", [0, 1]),
    "power_reliability": st.sidebar.slider("Power Reliability (0–1)", 0.0, 1.0, 0.7),
    "age": st.sidebar.slider("Age of Building (yrs)", 0, 100, 20)
}

input_df = pd.DataFrame([user_input])

ibadan_neighs = ["Agbowo", "Bodija", "GRA", "Moniya", "Ojoo", "Sango", "IbadanNorth"]
def map_to_ibadan(nbh):
    if pd.isna(nbh) or str(nbh).lower() == "nan":
        return "Unknown"
    h = int(hashlib.md5(str(nbh).encode("utf-8")).hexdigest()[:8], 16)
    return ibadan_neighs[h % len(ibadan_neighs)]

input_df["ibadan_neighborhood"] = input_df["Neighborhood"].astype(str).apply(map_to_ibadan)
input_df["price_per_sqm"] = input_df["GrLivArea"] / (input_df["GrLivArea"] + 1)
input_df["bed_bath_ratio"] = input_df["BedroomAbvGr"] / (input_df["FullBath"] + 0.5)

st.write("### 🧾 Entered Property Details")
st.dataframe(input_df)
if st.button("💰 Predict House Price"):
    log_pred = model.predict(input_df)[0]
    pred_naira = np.expm1(log_pred)
    st.success(f"🏷️ Estimated Property Price: ₦{pred_naira:,.0f}")

    try:
        xgb_final = model.named_steps["model"]
        preprocessor = model.named_steps["pre"]

        X_trans = preprocessor.transform(input_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.array(X_trans, dtype=float)

        feature_names = np.array(preprocessor.get_feature_names_out())

        try:
            booster = xgb_final.get_booster()
            base_score_attr = booster.attr("base_score")
            if isinstance(base_score_attr, str):
                clean_val = base_score_attr.strip("[]")
                booster.set_attr(base_score=str(float(clean_val)))
        except:
            pass

        try:
            explainer = shap.TreeExplainer(xgb_final)
        except:
            explainer = shap.Explainer(xgb_final.predict, X_trans)

        shap_values = explainer(X_trans)
        base_log = explainer.expected_value if hasattr(explainer, "expected_value") else shap_values[0].base_values
        shap_vals_log = shap_values.values[0] if hasattr(shap_values, "values") else shap_values[0].values

        # Approximate contribution of each feature in Naira
        contributions_naira = (np.expm1(base_log + shap_vals_log) - np.expm1(base_log))

        # Select top 10 features by absolute contribution
        top_idx = np.argsort(np.abs(contributions_naira))[-10:][::-1]
        top_features = feature_names[top_idx]
        top_values = contributions_naira[top_idx]

        # Plot horizontal bar chart with numeric labels
        st.subheader("📊 Top 10 Feature Contributions in Naira")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_features, top_values, color='skyblue')
        ax.set_xlabel("Contribution to Price (₦)")
        ax.set_ylabel("Features")
        ax.set_title("Top 10 SHAP Feature Contributions — Naira")
        ax.invert_yaxis()  # largest at top

        # Add numeric labels
        for bar, value in zip(bars, top_values):
            width = bar.get_width()
            ax.text(width + 0.01*pred_naira, bar.get_y() + bar.get_height()/2,
                    f"₦{value:,.0f}", va='center')

        # Add vertical line for expected price
        ax.axvline(np.expm1(base_log), color='red', linestyle='--', label=f"Expected Price: ₦{np.expm1(base_log):,.0f}")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
