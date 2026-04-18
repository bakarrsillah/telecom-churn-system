st.subheader("🧠 SHAP Explainability Dashboard")

try:
    pipeline = predict.get_pipeline()

    # IMPORTANT: use raw sample
    X_sample = X.sample(min(100, len(X)), random_state=42)

    # -------------------------
    # TRANSFORM DATA FIRST
    # -------------------------
    X_transformed = pipeline[:-1].transform(X_sample)

    model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_transformed)

    # -------------------------
    # CUSTOMER SELECT
    # -------------------------
    idx = st.selectbox("Select Customer Index", X_sample.index)

    st.write("### 👤 Raw Customer Data")
    st.write(X_sample.loc[idx])

    row_pos = X_sample.index.get_loc(idx)

    # -------------------------
    # SHAP VALUES
    # -------------------------
    values = shap_values[row_pos]

    feature_names = pipeline[:-1].get_feature_names_out()

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": values
    })

    explanation_df = explanation_df.sort_values(
        by="Impact",
        key=abs,
        ascending=False
    )

    st.write("### 📊 Feature Impact")
    st.dataframe(explanation_df)

    # -------------------------
    # INSIGHT
    # -------------------------
    top = explanation_df.iloc[0]

    direction = "increases churn risk" if top["Impact"] > 0 else "reduces churn risk"

    st.info(f"Key driver: **{top['Feature']}** → {direction}")

    # -------------------------
    # TOP DRIVERS
    # -------------------------
    st.bar_chart(explanation_df.head(5).set_index("Feature"))

except Exception as e:
    st.warning(f"SHAP not available: {e}")
