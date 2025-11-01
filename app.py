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
    # Predict
    log_pred = model.predict(input_df)[0]
    price = np.expm1(log_pred)
    st.success(f"🏷️ Estimated Property Price: ₦{price:,.0f}")
    st.caption("Prediction localized to Ibadan housing context")

    try:
        # Extract pipeline steps
        xgb_final = model.named_steps["model"]
        preprocessor = model.named_steps["pre"]

        # Transform inputs
        X_trans = preprocessor.transform(input_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.array(X_trans, dtype=float)

        feature_names = preprocessor.get_feature_names_out()

        # Patch base_score if needed
        try:
            booster = xgb_final.get_booster()
            base_score_attr = booster.attr("base_score")
            if isinstance(base_score_attr, str):
                clean_val = base_score_attr.strip("[]")
                booster.set_attr(base_score=str(float(clean_val)))
        except:
            pass

        # Initialize SHAP explainer
        try:
            explainer = shap.TreeExplainer(xgb_final)
        except Exception as e:
            st.warning(f"TreeExplainer failed ({e}), falling back to model-agnostic explainer.")
            explainer = shap.Explainer(xgb_final.predict, X_trans)

        # Compute SHAP values
        shap_values = explainer(X_trans)

        # Waterfall plot
        st.subheader("📊 Feature Contribution (SHAP)")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values.values[0] if hasattr(shap_values, "values") else shap_values[0].values,
                base_values=explainer.expected_value if hasattr(explainer, "expected_value") else shap_values[0].base_values,
                data=X_trans[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig)

        # Top-10 features by mean absolute SHAP
        mean_abs = np.abs(shap_values.values).mean(axis=0) if hasattr(shap_values, "values") else np.abs(shap_values[0].values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        st.subheader("🏆 Top 10 Features by SHAP Impact")
        st.bar_chart(shap_df.head(10).set_index("feature"))

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
