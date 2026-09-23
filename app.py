import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="HEA/MPEA Yield Strength ML",
    page_icon="ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>
    .main {
        background: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #172033, #101722);
        border: 1px solid #29364d;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #aab6c8;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .pipeline-card {
        padding: 1.1rem 0.8rem;
        min-height: 115px;
        border-radius: 14px;
        background: #151c28;
        border: 1px solid #29364d;
        text-align: center;
    }

    .pipeline-number {
        font-size: 0.78rem;
        color: #7fa7ff;
        font-weight: 700;
    }

    .pipeline-name {
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }

    .model-card {
        padding: 1.25rem;
        min-height: 190px;
        border-radius: 16px;
        background: #151c28;
        border: 1px solid #29364d;
    }

    .model-name {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }

    .model-weight {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .model-role {
        color: #aab6c8;
        margin-top: 0.5rem;
        line-height: 1.45;
    }

    .result-card {
        padding: 1.35rem;
        border-radius: 16px;
        background: #151c28;
        border: 1px solid #29364d;
        text-align: center;
    }

    .result-label {
        color: #9ba8bb;
        font-size: 0.85rem;
    }

    .result-value {
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .small-note {
        color: #9ba8bb;
        font-size: 0.86rem;
    }

    div[data-testid="stMetric"] {
        background: #151c28;
        border: 1px solid #29364d;
        padding: 1rem;
        border-radius: 14px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #29364d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data
def load_results():
    prediction_path = PROJECT_DIR / "YS_GROUPED_OOF_PREDICTIONS_WITH_UNCERTAINTY.csv"
    interval_path = PROJECT_DIR / "YS_CONFORMAL_INTERVAL_RESULTS.csv"
    band_path = PROJECT_DIR / "YS_UNCERTAINTY_BY_TARGET_RANGE.csv"

    predictions = pd.read_csv(prediction_path)
    intervals = pd.read_csv(interval_path)
    bands = pd.read_csv(band_path)

    return predictions, intervals, bands


try:
    predictions, intervals, bands = load_results()
except Exception as e:
    st.error("Required result files could not be loaded.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

# ============================================================
# PRIMARY PERFORMANCE
# ============================================================

R2 = 0.841914
MAE = 133.121
RMSE = 220.723

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">AI-Assisted HEA/MPEA Yield Strength Prediction</div>
    <div class="hero-subtitle">
        Machine Learning · Ensemble Learning · Conformal Uncertainty · SHAP Explainability
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Dashboard")
    st.caption("ML-based yield strength prediction")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Model Performance",
            "Prediction & Uncertainty",
            "Production Prediction",
            "Uncertainty Analysis",
            "Explainable AI"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model")
    st.caption("ExtraTrees + HistGradientBoosting + XGBoost")
    st.caption("Ensemble: 40% / 30% / 30%")

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    st.markdown(
        """
        This project develops a machine-learning pipeline for predicting
        tensile yield strength of High-Entropy Alloys (HEAs) and
        Medium-Entropy/Complex Concentrated Alloys (MPEAs).
        """
    )

    # --------------------------------------------------------
    # KEY PROJECT METRICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Project at a Glance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Dataset Records",
            "1,941"
        )

    with c2:
        st.metric(
            "Engineered Features",
            "359"
        )

    with c3:
        st.metric(
            "Random CV R2",
            "0.842"
        )

    with c4:
        st.metric(
            "MAE",
            "133.12 MPa"
        )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Machine Learning Pipeline</div>',
        unsafe_allow_html=True
    )

    pipeline = [
        ("01", "Data", "1941 experimental records"),
        ("02", "Preprocessing", "Missing-value handling"),
        ("03", "Feature Engineering", "359 final features"),
        ("04", "Ensemble Learning", "3 tree-based models"),
        ("05", "YS Prediction", "Tensile yield strength"),
        ("06", "Uncertainty", "Conformal intervals"),
        ("07", "Explainability", "SHAP analysis")
    ]

    cols = st.columns(7)

    for col, (number, name, description) in zip(cols, pipeline):

        with col:

            st.markdown(
                f"""
                <div class="pipeline-card">
                    <div class="pipeline-number">{number}</div>
                    <div class="pipeline-name">{name}</div>
                    <div class="small-note">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # MODEL ARCHITECTURE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Ensemble Model Architecture</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            """
            <div class="model-card">
                <div class="model-name">ExtraTrees</div>
                <div class="model-weight">40%</div>
                <div class="model-role">
                    Randomized tree ensemble for nonlinear
                    relationships and feature interactions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            """
            <div class="model-card">
                <div class="model-name">HistGradientBoosting</div>
                <div class="model-weight">30%</div>
                <div class="model-role">
                    Gradient-boosted trees for nonlinear
                    predictive patterns.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            """
            <div class="model-card">
                <div class="model-name">XGBoost</div>
                <div class="model-weight">30%</div>
                <div class="model-role">
                    Boosted-tree model providing complementary
                    ensemble diversity.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "R2",
            "0.841914"
        )

    with p2:
        st.metric(
            "MAE",
            "133.121 MPa"
        )

    with p3:
        st.metric(
            "RMSE",
            "220.723 MPa"
        )

    st.caption(
        "Performance shown here is from random 5-fold cross-validation."
    )

    # --------------------------------------------------------
    # PROJECT OUTPUTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Key Outputs</div>',
        unsafe_allow_html=True
    )

    o1, o2, o3 = st.columns(3)

    with o1:
        st.info(
            "**Yield Strength Prediction**\n\n"
            "Ensemble prediction of tensile yield strength in MPa."
        )

    with o2:
        st.info(
            "**Conformal Uncertainty**\n\n"
            "Prediction intervals at 80%, 90% and 95% coverage levels."
        )

    with o3:
        st.info(
            "**SHAP Explainability**\n\n"
            "Feature-level interpretation of model predictions."
        )
elif page == "Model Performance":

    st.header("Model Performance")
    st.caption("Random 5-fold cross-validation")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("R2", f"{R2:.6f}")

    with c2:
        st.metric("MAE", f"{MAE:.3f} MPa")

    with c3:
        st.metric("RMSE", f"{RMSE:.3f} MPa")

    st.subheader("Actual vs Predicted Yield Strength")

    chart = predictions[
        ["Actual_YS_MPa", "Predicted_YS_MPa"]
    ].rename(
        columns={
            "Actual_YS_MPa": "Actual",
            "Predicted_YS_MPa": "Predicted"
        }
    )

    st.scatter_chart(
        chart,
        x="Actual",
        y="Predicted",
        use_container_width=True
    )

    st.subheader("Absolute Prediction Error")

    error_chart = predictions[
        ["Actual_YS_MPa", "Absolute_Error_MPa"]
    ].rename(
        columns={
            "Actual_YS_MPa": "Actual YS",
            "Absolute_Error_MPa": "Absolute Error"
        }
    )

    st.scatter_chart(
        error_chart,
        x="Actual YS",
        y="Absolute Error",
        use_container_width=True
    )

    st.info(
        f"The ensemble achieves R2 = {R2:.6f}, "
        f"MAE = {MAE:.3f} MPa and RMSE = {RMSE:.3f} MPa "
        "under random 5-fold cross-validation."
    )

# ============================================================
# PREDICTION + UNCERTAINTY
# ============================================================

elif page == "Prediction & Uncertainty":

    st.header("Prediction & Uncertainty")

    st.caption(
        "Inspect an out-of-fold prediction and its calibrated "
        "conformal prediction interval."
    )

    st.markdown('<div class="section-title">Validation Record</div>', unsafe_allow_html=True)

    index = st.slider(
        "Select validation record",
        min_value=0,
        max_value=len(predictions) - 1,
        value=0
    )

    row = predictions.iloc[index]

    actual = float(row["Actual_YS_MPa"])
    predicted = float(row["Predicted_YS_MPa"])
    error = float(row["Absolute_Error_MPa"])
    lower = float(row["Lower_95_MPa"])
    upper = float(row["Upper_95_MPa"])
    width = float(row["Interval_Width_MPa"])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Actual Yield Strength",
            f"{actual:.1f} MPa"
        )

    with c2:
        st.metric(
            "Predicted Yield Strength",
            f"{predicted:.1f} MPa"
        )

    with c3:
        st.metric(
            "Absolute Error",
            f"{error:.1f} MPa"
        )

    st.markdown('<div class="section-title">95% Conformal Prediction</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)

    with i1:
        st.metric(
            "Lower Bound",
            f"{lower:.1f} MPa"
        )

    with i2:
        st.metric(
            "Predicted YS",
            f"{predicted:.1f} MPa"
        )

    with i3:
        st.metric(
            "Upper Bound",
            f"{upper:.1f} MPa"
        )

    st.progress(
        min(max((actual - lower) / max(upper - lower, 1), 0.0), 1.0)
    )

    st.caption(
        f"95% prediction interval width: {width:.1f} MPa"
    )

    if bool(row["Covered_95pct"]):
        st.success(
            "The observed yield strength is inside the 95% "
            "conformal prediction interval."
        )
    else:
        st.warning(
            "The observed yield strength is outside the 95% "
            "conformal prediction interval."
        )

    st.markdown('<div class="section-title">Prediction Comparison</div>', unsafe_allow_html=True)

    comparison = pd.DataFrame({
        "Quantity": [
            "Actual YS",
            "Predicted YS",
            "Absolute Error",
            "Lower 95%",
            "Upper 95%"
        ],
        "Value (MPa)": [
            actual,
            predicted,
            error,
            lower,
            upper
        ]
    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        hide_index=True
    )
elif page == "Uncertainty Analysis":

    st.header("Conformal Uncertainty Analysis")

    st.caption(
        "Prediction intervals calibrated from out-of-fold residuals."
    )

    # --------------------------------------------------------
    # GLOBAL INTERVAL SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Prediction Interval Summary</div>',
        unsafe_allow_html=True
    )

    q80 = 266.232
    q90 = 408.218
    q95 = 563.710

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "80% Interval",
            f"± {q80:.1f} MPa"
        )

    with c2:
        st.metric(
            "90% Interval",
            f"± {q90:.1f} MPa"
        )

    with c3:
        st.metric(
            "95% Interval",
            f"± {q95:.1f} MPa"
        )

    st.markdown(
        '<div class="section-title">Empirical Coverage</div>',
        unsafe_allow_html=True
    )

    coverage = intervals.copy()

    if "Nominal_Coverage" in coverage.columns:
        coverage["Nominal Coverage (%)"] = (
            coverage["Nominal_Coverage"] * 100
        )

    if "Empirical_Coverage" in coverage.columns:
        coverage["Empirical Coverage (%)"] = (
            coverage["Empirical_Coverage"] * 100
        )

    coverage = coverage.rename(
        columns={
            "Quantile": "Conformal Quantile (MPa)",
            "Mean_Interval_Width": "Mean Width (MPa)",
            "Median_Interval_Width": "Median Width (MPa)"
        }
    )

    display_columns = [
        c for c in [
            "Nominal Coverage (%)",
            "Conformal Quantile (MPa)",
            "Empirical Coverage (%)",
            "Mean Width (MPa)",
            "Median Width (MPa)"
        ]
        if c in coverage.columns
    ]

    st.dataframe(
        coverage[display_columns].round(2),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # COVERAGE BY STRENGTH RANGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">95% Coverage by Yield Strength Range</div>',
        unsafe_allow_html=True
    )

    band_display = bands.copy()

    st.dataframe(
        band_display.round(2),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Interpretation</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        The global 95% conformal interval has approximately 95%
        empirical coverage across the evaluated out-of-fold records.

        Coverage is lower in the high-strength region. Therefore,
        the global interval provides an overall uncertainty estimate
        but should not be interpreted as equally informative across
        every yield-strength range.
        """
    )

    st.caption(
        "Conformal uncertainty describes predictive coverage; it does "
        "not establish causal relationships or guarantee future coverage "
        "for every composition or strength range."
    )
elif page == "Explainable AI":

    st.header("Explainable AI - SHAP")

    st.caption(
        "Global feature importance based on mean absolute SHAP values."
    )

    shap_path = PROJECT_DIR / "YS_SHAP_CORRECTED_ORIGINAL.csv"

    if shap_path.exists():

        shap_df = pd.read_csv(shap_path)

        feature_col = shap_df.columns[0]
        value_col = shap_df.columns[1]

        top25 = shap_df.head(25).copy()
        top10 = shap_df.head(10).copy()

        # ----------------------------------------------------
        # KEY INSIGHT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Key Model Drivers</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Top Feature",
                str(top10.iloc[0][feature_col])
            )

        with c2:
            st.metric(
                "Top Feature SHAP",
                f"{top10.iloc[0][value_col]:.2f}"
            )

        with c3:
            st.metric(
                "Features Shown",
                "25"
            )

        # ----------------------------------------------------
        # TOP FEATURES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Top 10 Features</div>',
            unsafe_allow_html=True
        )

        chart_data = (
            top10
            .set_index(feature_col)[value_col]
            .sort_values(ascending=True)
        )

        st.bar_chart(
            chart_data,
            horizontal=True,
            use_container_width=True
        )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">SHAP Feature Importance</div>',
            unsafe_allow_html=True
        )

        table = top25.copy()

        # Preserve every column from the corrected SHAP results.
        # Rename only the first two columns; keep any additional
        # information already present in the CSV.
        renamed_columns = list(table.columns)

        renamed_columns[0] = "Feature"
        renamed_columns[1] = "Mean |SHAP|"

        table.columns = renamed_columns

        table.insert(
            0,
            "Rank",
            range(1, len(table) + 1)
        )

        st.dataframe(
            table.round(4),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # IMPORTANT INTERPRETATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Interpretation</div>',
            unsafe_allow_html=True
        )

        st.info(
            """
            SHAP values quantify how individual features contribute
            to the predictions made by the trained ensemble model.

            Higher mean absolute SHAP values indicate features that
            have a larger influence on model predictions across the
            analyzed samples.

            SHAP importance describes model behavior and does not
            establish a causal physical relationship.
            """
        )

    else:

        st.warning(
            "YS_SHAP_CORRECTED_ORIGINAL.csv was not found."
        )
elif page == "Production Prediction":

    st.header("Production Yield Strength Prediction")

    st.caption(
        "Generate a prediction using the trained 359-feature ensemble."
    )

    MODEL_DIR = PROJECT_DIR / "models" / "YS_Final_359"

    @st.cache_resource
    def load_production_models():

        with open(MODEL_DIR / "YS_Final_359_Preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        with open(MODEL_DIR / "YS_Final_359_ET.pkl", "rb") as f:
            et = pickle.load(f)

        with open(MODEL_DIR / "YS_Final_359_HGB.pkl", "rb") as f:
            hgb = pickle.load(f)

        with open(MODEL_DIR / "YS_Final_359_XGB.pkl", "rb") as f:
            xgb = pickle.load(f)

        with open(MODEL_DIR / "YS_Final_359_Metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return preprocessor, et, hgb, xgb, metadata

    try:
        preprocessor, et, hgb, xgb, metadata = load_production_models()
        st.success("359-feature production ensemble loaded successfully.")
    except Exception as e:
        st.error("Production model could not be loaded.")
        st.code(f"{type(e).__name__}: {e}")
        st.stop()

    @st.cache_data
    def load_prediction_dataset():

        path = PROJECT_DIR / "YS_ML_Advanced_Composition_Features.csv"
        data = pd.read_csv(path)

        if "Record_ID" in data.columns:
            data = data[
                data["Record_ID"].astype(str) != "R00159"
            ].copy()

        return data.reset_index(drop=True)

    prediction_data = load_prediction_dataset()

    st.subheader("Select Dataset Record")

    selected_index = st.selectbox(
        "Dataset record",
        list(prediction_data.index),
        format_func=lambda i: (
            f"Row {i} | Record ID: "
            f"{prediction_data.iloc[i]['Record_ID']}"
            if "Record_ID" in prediction_data.columns
            else f"Row {i}"
        )
    )

    selected = prediction_data.iloc[selected_index]

    display_columns = [
        c for c in [
            "Record_ID",
            "Composition_Canonical",
            "Phase",
            "Processing_Method",
            "Test_Type",
            "Test_Temperature_C",
            "Grain_Size_um",
            "YS_Tensile_MPa"
        ]
        if c in prediction_data.columns
    ]

    st.dataframe(
        selected[display_columns].to_frame("Value"),
        use_container_width=True
    )

    st.subheader("Scenario Analysis")

    st.caption(
        "Modify test temperature or grain size while keeping the "
        "remaining characteristics of the selected record unchanged."
    )

    c1, c2 = st.columns(2)

    scenario_temperature = None
    scenario_grain_size = None

    if "Test_Temperature_C" in prediction_data.columns:

        series = pd.to_numeric(
            prediction_data["Test_Temperature_C"],
            errors="coerce"
        ).dropna()

        if len(series):

            minimum = float(series.min())
            maximum = float(series.max())

            current = (
                float(selected["Test_Temperature_C"])
                if pd.notna(selected["Test_Temperature_C"])
                else float(series.median())
            )

            with c1:
                scenario_temperature = st.number_input(
                    "Test Temperature (°C)",
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(current, minimum), maximum),
                    step=10.0
                )

    if "Grain_Size_um" in prediction_data.columns:

        series = pd.to_numeric(
            prediction_data["Grain_Size_um"],
            errors="coerce"
        ).dropna()

        if len(series):

            minimum = float(series.min())
            maximum = float(series.max())

            current = (
                float(selected["Grain_Size_um"])
                if pd.notna(selected["Grain_Size_um"])
                else float(series.median())
            )

            with c2:
                scenario_grain_size = st.number_input(
                    "Grain Size (µm)",
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(current, minimum), maximum),
                    step=0.1
                )

    input_df = prediction_data.iloc[[selected_index]].copy()

    if scenario_temperature is not None:
        input_df.loc[
            input_df.index[0],
            "Test_Temperature_C"
        ] = scenario_temperature

    if scenario_grain_size is not None:
        input_df.loc[
            input_df.index[0],
            "Grain_Size_um"
        ] = scenario_grain_size

    DROP = {
        "YS_Tensile_MPa",
        "Record_ID",
        "Composition_Canonical",
        "Predicted_Solidus_C"
    }

    categorical_features = [
        "Phase",
        "Processing_Method",
        "Alloy_Class",
        "Equilibrium_Condition",
        "Single_Multiphase",
        "Test_Type",
        "Precipitate_Info"
    ]

    categorical_features = [
        c for c in categorical_features
        if c in input_df.columns
    ]

    base_features = [
        c for c in input_df.columns
        if c not in DROP
    ]

    phase = (
        input_df["Phase"]
        .fillna("MISSING")
        .astype(str)
        .str.upper()
    )

    input_df["PHASE_BCC_IND"] = (
        phase.str.contains("BCC", regex=False, na=False)
        | phase.str.contains("BODY", regex=False, na=False)
    ).astype(int)

    input_df["PHASE_FCC_IND"] = (
        phase.str.contains("FCC", regex=False, na=False)
        | phase.str.contains("FACE", regex=False, na=False)
    ).astype(int)

    input_df["PHASE_HCP_IND"] = (
        phase.str.contains("HCP", regex=False, na=False)
        | phase.str.contains("HEXAGONAL", regex=False, na=False)
    ).astype(int)

    input_df["PHASE_B2_IND"] = (
        phase.str.contains("B2", regex=False, na=False)
        | phase.str.contains("BETA", regex=False, na=False)
    ).astype(int)

    input_df["PHASE_LAVES_IND"] = (
        phase.str.contains("LAVES", regex=False, na=False)
    ).astype(int)

    input_df["PHASE_SIGMA_IND"] = (
        phase.str.contains("SIGMA", regex=False, na=False)
    ).astype(int)

    input_df["PHASE_L12_IND"] = (
        phase.str.contains("L12", regex=False, na=False)
        | phase.str.contains("GAMMA PRIME", regex=False, na=False)
    ).astype(int)

    phase_indicator_cols = [
        "PHASE_BCC_IND",
        "PHASE_FCC_IND",
        "PHASE_HCP_IND",
        "PHASE_B2_IND",
        "PHASE_LAVES_IND",
        "PHASE_SIGMA_IND",
        "PHASE_L12_IND"
    ]

    input_df["PHASE_COMPLEX_IND"] = (
        input_df[phase_indicator_cols].sum(axis=1) >= 2
    ).astype(int)

    phase_test_features = phase_indicator_cols + [
        "PHASE_COMPLEX_IND"
    ]

    test = (
        input_df["Test_Type"]
        .fillna("MISSING")
        .astype(str)
        .str.upper()
    )

    input_df["PHASE_TEST_TENSILE"] = (
        test.str.contains("T", regex=False, na=False)
    ).astype(int)

    input_df["PHASE_TEST_COMPRESSION"] = (
        test.str.contains("C", regex=False, na=False)
    ).astype(int)

    phase_test_features += [
        "PHASE_TEST_TENSILE",
        "PHASE_TEST_COMPRESSION"
    ]

    numeric_candidates = [
        c for c in base_features
        if c not in categorical_features
    ]

    keywords = [
        "VEC",
        "atomic_radius",
        "atomic_size",
        "electronegativity",
        "mixing_enthalpy",
        "mixing_entropy",
        "atomic_mass",
        "melting",
        "valence",
        "density"
    ]

    selected_numeric = []

    for col in numeric_candidates:

        low = col.lower()

        if any(k.lower() in low for k in keywords):
            selected_numeric.append(col)

    selected_numeric = list(dict.fromkeys(selected_numeric))[:20]

    for phase_col in phase_test_features:

        for num_col in selected_numeric:

            name = phase_col + "__X__" + num_col

            median_value = prediction_data[num_col].median()

            input_df[name] = (
                input_df[phase_col]
                * input_df[num_col].fillna(median_value)
            )

    interaction_features = [
        phase_col + "__X__" + num_col
        for phase_col in phase_test_features
        for num_col in selected_numeric
    ]

    features = list(
        dict.fromkeys(
            base_features +
            phase_test_features +
            interaction_features
        )
    )

    X_input = input_df[features]

    transformed = preprocessor.transform(X_input)

    pred_et = et.predict(transformed)[0]
    pred_hgb = hgb.predict(transformed)[0]
    pred_xgb = xgb.predict(transformed)[0]

    prediction = (
        0.40 * pred_et +
        0.30 * pred_hgb +
        0.30 * pred_xgb
    )

    baseline_value = (
        float(selected["YS_Tensile_MPa"])
        if "YS_Tensile_MPa" in selected.index
        and pd.notna(selected["YS_Tensile_MPa"])
        else np.nan
    )

    q80 = 266.232
    q95 = 563.710

    st.divider()

    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Ensemble Prediction",
            f"{prediction:.1f} MPa"
        )

    with c2:
        st.metric(
            "80% Interval",
            f"{prediction - q80:.1f} to {prediction + q80:.1f} MPa"
        )

    with c3:
        st.metric(
            "95% Interval",
            f"{prediction - q95:.1f} to {prediction + q95:.1f} MPa"
        )

    if np.isfinite(baseline_value):

        delta = prediction - baseline_value

        st.subheader("Scenario vs Dataset Value")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Dataset YS",
                f"{baseline_value:.1f} MPa"
            )

        with c2:
            st.metric(
                "Scenario Prediction",
                f"{prediction:.1f} MPa",
                delta=f"{delta:+.1f} MPa"
            )

    st.subheader("Individual Model Predictions")

    model_predictions = pd.DataFrame({
        "Model": [
            "ExtraTrees",
            "HistGradientBoosting",
            "XGBoost",
            "Ensemble"
        ],
        "Prediction (MPa)": [
            pred_et,
            pred_hgb,
            pred_xgb,
            prediction
        ],
        "Weight": [
            "40%",
            "30%",
            "30%",
            "100%"
        ]
    })

    st.dataframe(
        model_predictions.round(2),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Assisted HEA/MPEA Materials Discovery | "
    "Machine Learning Dashboard"
)






