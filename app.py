"""
AI-Assisted Concrete Mix Design Platform
========================================

Streamlit prototype for:
- AI-based compressive strength prediction
- Training-range reliability checks
- Mix comparison
- Cost and GWP assessment
- Laboratory validation with actual results

Run:
    pip install -r requirements.txt
    streamlit run app.py

Required model files:
    model.pkl
    scaler.pkl

Important:
- Mix inputs and validation CSV files must use RAW engineering units.
- The app applies scaler.pkl internally before prediction.
- Cost and GWP factors are placeholders and must be replaced with verified
  project, supplier, or EPD data before practical use.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI-Assisted Concrete Mix Design Platform",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================
FEATURE_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

TARGET_COLUMN = "concrete_compressive_strength"

FEATURE_LABELS_KR = {
    "cement": "시멘트",
    "blast_furnace_slag": "고로 슬래그",
    "fly_ash": "플라이 애시",
    "water": "물",
    "superplasticizer": "고성능 감수제",
    "coarse_aggregate": "굵은 골재",
    "fine_aggregate": "잔 골재",
    "age": "양생 기간",
}

FEATURE_UNITS = {
    "cement": "kg/m³",
    "blast_furnace_slag": "kg/m³",
    "fly_ash": "kg/m³",
    "water": "kg/m³",
    "superplasticizer": "kg/m³",
    "coarse_aggregate": "kg/m³",
    "fine_aggregate": "kg/m³",
    "age": "일",
}

TRAIN_RANGES = {
    "cement": (100.0, 540.0),
    "blast_furnace_slag": (0.0, 359.0),
    "fly_ash": (0.0, 200.0),
    "water": (120.0, 247.0),
    "superplasticizer": (0.0, 32.0),
    "coarse_aggregate": (800.0, 1145.0),
    "fine_aggregate": (594.0, 992.0),
    "age": (1.0, 365.0),
}

DEFAULT_MIX = {
    "cement": 300.0,
    "blast_furnace_slag": 60.0,
    "fly_ash": 30.0,
    "water": 180.0,
    "superplasticizer": 6.0,
    "coarse_aggregate": 970.0,
    "fine_aggregate": 780.0,
    "age": 28.0,
}

SLIDER_STEP = {
    "cement": 5.0,
    "blast_furnace_slag": 5.0,
    "fly_ash": 5.0,
    "water": 1.0,
    "superplasticizer": 0.5,
    "coarse_aggregate": 5.0,
    "fine_aggregate": 5.0,
    "age": 1.0,
}

MATERIAL_FEATURES = [feature for feature in FEATURE_COLUMNS if feature != "age"]

# Placeholder values only.
DEFAULT_COST_PER_KG = {
    "cement": 0.12,
    "blast_furnace_slag": 0.03,
    "fly_ash": 0.02,
    "water": 0.001,
    "superplasticizer": 2.00,
    "coarse_aggregate": 0.015,
    "fine_aggregate": 0.015,
}

DEFAULT_GWP_PER_KG = {
    "cement": 0.90,
    "blast_furnace_slag": 0.03,
    "fly_ash": 0.02,
    "water": 0.0003,
    "superplasticizer": 1.20,
    "coarse_aggregate": 0.007,
    "fine_aggregate": 0.007,
}


# ============================================================
# MODEL AND CALCULATION FUNCTIONS
# ============================================================
@st.cache_resource
def load_model_and_scaler() -> tuple[Any | None, Any | None, str | None]:
    """Load model.pkl and scaler.pkl from the app directory."""
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path):
        return None, None, "model.pkl 파일을 찾을 수 없습니다."
    if not os.path.exists(scaler_path):
        return None, None, "scaler.pkl 파일을 찾을 수 없습니다."

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, None
    except Exception as exc:
        return None, None, f"모델 또는 스케일러 로드 실패: {exc}"


def demo_formula_predict(mix: dict[str, float]) -> float:
    """Fallback demonstration formula used only when the trained model is unavailable."""
    binder = (
        mix["cement"]
        + mix["blast_furnace_slag"]
        + mix["fly_ash"]
    )
    wb_ratio = mix["water"] / max(binder, 1e-6)
    age_factor = np.log1p(mix["age"]) / np.log1p(28)
    base_strength = 100.0 / (1.5 ** (wb_ratio * 3))
    return float(np.clip(base_strength * min(age_factor, 1.3), 2.0, 85.0))


def prepare_raw_input_frame(mix: dict[str, float]) -> pd.DataFrame:
    """Create one-row raw-input dataframe in the exact training feature order."""
    return pd.DataFrame(
        [[float(mix[column]) for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


def predict_strength(
    mix: dict[str, float],
    model: Any | None,
    scaler: Any | None,
) -> tuple[float, bool]:
    """Return predicted strength and whether the trained model was used."""
    if model is None or scaler is None:
        return demo_formula_predict(mix), False

    raw_input = prepare_raw_input_frame(mix)
    scaled_input = scaler.transform(raw_input)
    prediction = model.predict(scaled_input)
    return float(prediction[0]), True


def predict_dataframe(
    raw_features: pd.DataFrame,
    model: Any,
    scaler: Any,
) -> np.ndarray:
    """
    Predict from RAW engineering-unit inputs.

    This intentionally applies scaler.transform exactly once.
    """
    ordered = raw_features[FEATURE_COLUMNS].astype(float)
    scaled = scaler.transform(ordered)
    return np.asarray(model.predict(scaled), dtype=float)


def get_out_of_range_rows(mix: dict[str, float]) -> list[dict[str, str | float]]:
    """Return human-readable training-range warnings."""
    warnings: list[dict[str, str | float]] = []

    for feature, (minimum, maximum) in TRAIN_RANGES.items():
        value = float(mix[feature])
        if value < minimum or value > maximum:
            warnings.append(
                {
                    "변수": FEATURE_LABELS_KR[feature],
                    "입력값": value,
                    "단위": FEATURE_UNITS[feature],
                    "학습 범위": f"{minimum:g}–{maximum:g}",
                }
            )

    return warnings


def compute_cost_and_gwp(
    mix: dict[str, float],
    cost_factors: dict[str, float],
    gwp_factors: dict[str, float],
) -> tuple[float, float]:
    """Calculate indicative material cost and GWP per cubic metre."""
    total_cost = sum(mix[f] * cost_factors[f] for f in MATERIAL_FEATURES)
    total_gwp = sum(mix[f] * gwp_factors[f] for f in MATERIAL_FEATURES)
    return float(total_cost), float(total_gwp)


def calculate_validation_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate R², RMSE, and MAE without additional dependencies."""
    actual_values = actual.astype(float).to_numpy()
    residuals = actual_values - predicted

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((actual_values - actual_values.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"R²": r2, "RMSE": rmse, "MAE": mae}


def reset_saved_mixes() -> None:
    st.session_state.saved_mixes = []


# ============================================================
# SESSION STATE
# ============================================================
if "saved_mixes" not in st.session_state:
    st.session_state.saved_mixes = []

if "cost_factors" not in st.session_state:
    st.session_state.cost_factors = DEFAULT_COST_PER_KG.copy()

if "gwp_factors" not in st.session_state:
    st.session_state.gwp_factors = DEFAULT_GWP_PER_KG.copy()


model, scaler, model_error = load_model_and_scaler()
using_trained_model = model is not None and scaler is not None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Platform Settings")

    if using_trained_model:
        st.success("XGBoost model and scaler connected")
    else:
        st.error(model_error or "학습 모델을 사용할 수 없습니다.")
        st.warning(
            "현재 데모 근사식으로 실행됩니다. 실제 발표용 배포에서는 "
            "model.pkl과 scaler.pkl을 반드시 포함하세요."
        )

    st.divider()
    st.subheader("Material Cost Factors")
    st.caption("단위: $/kg · 실제 공급업체 단가로 교체")

    for feature in MATERIAL_FEATURES:
        st.session_state.cost_factors[feature] = st.number_input(
            FEATURE_LABELS_KR[feature],
            min_value=0.0,
            value=float(st.session_state.cost_factors[feature]),
            step=0.001,
            format="%.3f",
            key=f"cost_{feature}",
        )

    st.divider()
    st.subheader("Material GWP Factors")
    st.caption("단위: kgCO₂e/kg · 검증된 EPD 데이터로 교체")

    for feature in MATERIAL_FEATURES:
        st.session_state.gwp_factors[feature] = st.number_input(
            FEATURE_LABELS_KR[feature],
            min_value=0.0,
            value=float(st.session_state.gwp_factors[feature]),
            step=0.001,
            format="%.3f",
            key=f"gwp_{feature}",
        )

    st.divider()
    st.caption(
        "Prototype note: cost and GWP outputs are indicative until verified "
        "regional price and EPD datasets are connected."
    )


# ============================================================
# HEADER AND PLATFORM OVERVIEW
# ============================================================
st.title("🧱 AI-Assisted Concrete Mix Design Platform")
st.markdown(
    """
    **콘크리트 배합 설계를 위한 AI 기반 엔지니어링 의사결정 지원 프로토타입**

    압축강도 예측뿐 아니라 **배합 비교, 비용, GWP, 실험 결과 검증**을 하나의
    workflow로 연결합니다.
    """
)

overview_1, overview_2, overview_3, overview_4, overview_5 = st.columns(5)
overview_1.metric("AI Prediction", "Strength")
overview_2.metric("Reliability", "Range Check")
overview_3.metric("Decision Support", "Mix Compare")
overview_4.metric("Sustainability", "Cost + GWP")
overview_5.metric("Validation", "Actual vs AI")

st.info(
    "AI 결과는 후보 배합 검토를 지원하기 위한 참고 정보입니다. "
    "최종 판단에는 재료 품질, 배합 시험, 시공성 및 현장 검증이 필요합니다."
)


# ============================================================
# MAIN TABS
# ============================================================
tab_design, tab_compare, tab_validation, tab_sustainability = st.tabs(
    [
        "🧪 Mix Design & AI Evaluation",
        "⚖️ Alternative Comparison",
        "📈 Laboratory Validation",
        "🌱 Cost & Sustainability",
    ]
)


# ============================================================
# TAB 1 — MIX DESIGN & AI EVALUATION
# ============================================================
with tab_design:
    st.subheader("Mix Design Input")

    input_column, result_column = st.columns([1.35, 1])

    with input_column:
        current_mix: dict[str, float] = {}
        input_left, input_right = st.columns(2)

        for index, feature in enumerate(FEATURE_COLUMNS):
            minimum, maximum = TRAIN_RANGES[feature]
            widget_column = input_left if index % 2 == 0 else input_right

            current_mix[feature] = widget_column.slider(
                f"{FEATURE_LABELS_KR[feature]} ({FEATURE_UNITS[feature]})",
                min_value=float(minimum * 0.5),
                max_value=float(maximum * 1.3),
                value=float(DEFAULT_MIX[feature]),
                step=float(SLIDER_STEP[feature]),
                key=f"mix_input_{feature}",
            )

        mix_name = st.text_input(
            "배합 이름",
            value="Candidate Mix A",
            help="저장 후 Alternative Comparison 탭에서 여러 배합을 비교할 수 있습니다.",
        )

    predicted_strength, used_real_model = predict_strength(
        current_mix,
        model,
        scaler,
    )
    estimated_cost, estimated_gwp = compute_cost_and_gwp(
        current_mix,
        st.session_state.cost_factors,
        st.session_state.gwp_factors,
    )

    binder = (
        current_mix["cement"]
        + current_mix["blast_furnace_slag"]
        + current_mix["fly_ash"]
    )
    water_binder_ratio = current_mix["water"] / max(binder, 1e-6)

    with result_column:
        st.markdown("#### AI Evaluation")

        st.metric(
            "Predicted Compressive Strength",
            f"{predicted_strength:.1f} MPa",
        )

        result_1, result_2 = st.columns(2)
        result_1.metric("Estimated Cost", f"${estimated_cost:.2f}/m³")
        result_2.metric("Estimated GWP", f"{estimated_gwp:.1f} kgCO₂e/m³")

        result_3, result_4 = st.columns(2)
        result_3.metric("Total Binder", f"{binder:.1f} kg/m³")
        result_4.metric("Water/Binder Ratio", f"{water_binder_ratio:.3f}")

        if not used_real_model:
            st.warning("현재 결과는 데모 근사식입니다.")

        range_warnings = get_out_of_range_rows(current_mix)

        if range_warnings:
            st.error(
                f"{len(range_warnings)}개 입력값이 학습 데이터 범위를 벗어났습니다. "
                "해당 예측은 외삽(extrapolation)이므로 신뢰도가 낮을 수 있습니다."
            )
            st.dataframe(
                pd.DataFrame(range_warnings),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("모든 입력값이 학습 데이터 범위 안에 있습니다.")

        if st.button(
            "💾 Save Candidate Mix",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.saved_mixes.append(
                {
                    "name": mix_name.strip() or f"Mix {len(st.session_state.saved_mixes) + 1}",
                    "mix": current_mix.copy(),
                    "predicted_strength": predicted_strength,
                    "cost": estimated_cost,
                    "gwp": estimated_gwp,
                    "wb_ratio": water_binder_ratio,
                    "in_training_range": not bool(range_warnings),
                }
            )
            st.success(f"'{mix_name}' 배합을 비교 목록에 저장했습니다.")


# ============================================================
# TAB 2 — ALTERNATIVE COMPARISON
# ============================================================
with tab_compare:
    st.subheader("Candidate Mix Comparison")
    st.caption(
        "후보 배합의 강도, 비용, 탄소 효율을 동시에 검토합니다."
    )

    if not st.session_state.saved_mixes:
        st.info(
            "저장된 후보 배합이 없습니다. Mix Design & AI Evaluation 탭에서 "
            "2개 이상의 배합을 저장하세요."
        )
    else:
        comparison_rows = []

        for saved in st.session_state.saved_mixes:
            comparison_rows.append(
                {
                    "Mix": saved["name"],
                    "Predicted Strength (MPa)": round(saved["predicted_strength"], 2),
                    "Cost ($/m³)": round(saved["cost"], 2),
                    "GWP (kgCO₂e/m³)": round(saved["gwp"], 2),
                    "W/B": round(saved["wb_ratio"], 3),
                    "Strength / Cost": round(
                        saved["predicted_strength"] / max(saved["cost"], 1e-6),
                        3,
                    ),
                    "Strength / GWP": round(
                        saved["predicted_strength"] / max(saved["gwp"], 1e-6),
                        4,
                    ),
                    "Within Training Range": "Yes" if saved["in_training_range"] else "No",
                }
            )

        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        chart_left, chart_right = st.columns(2)

        with chart_left:
            strength_chart = px.bar(
                comparison_df,
                x="Mix",
                y="Predicted Strength (MPa)",
                text_auto=".1f",
                title="Predicted Strength by Candidate Mix",
            )
            strength_chart.update_layout(height=390)
            st.plotly_chart(strength_chart, use_container_width=True)

        with chart_right:
            pareto_chart = px.scatter(
                comparison_df,
                x="GWP (kgCO₂e/m³)",
                y="Predicted Strength (MPa)",
                size="Cost ($/m³)",
                color="Mix",
                hover_data=[
                    "Cost ($/m³)",
                    "W/B",
                    "Strength / Cost",
                    "Strength / GWP",
                ],
                title="Strength–Carbon–Cost Trade-off",
            )
            pareto_chart.update_layout(height=390)
            st.plotly_chart(pareto_chart, use_container_width=True)

        best_strength = comparison_df.loc[
            comparison_df["Predicted Strength (MPa)"].idxmax(),
            "Mix",
        ]
        best_cost_efficiency = comparison_df.loc[
            comparison_df["Strength / Cost"].idxmax(),
            "Mix",
        ]
        best_carbon_efficiency = comparison_df.loc[
            comparison_df["Strength / GWP"].idxmax(),
            "Mix",
        ]

        summary_1, summary_2, summary_3 = st.columns(3)
        summary_1.success(f"Highest Strength\n\n**{best_strength}**")
        summary_2.success(f"Best Strength / Cost\n\n**{best_cost_efficiency}**")
        summary_3.success(f"Best Strength / GWP\n\n**{best_carbon_efficiency}**")

        st.warning(
            "최고 강도 배합이 반드시 최적 배합을 의미하지 않습니다. "
            "요구 성능, 재료 가용성, 시공성, 비용 및 환경영향을 함께 검토해야 합니다."
        )

        if st.button("🗑️ Clear Saved Mixes"):
            reset_saved_mixes()
            st.rerun()


# ============================================================
# TAB 3 — LABORATORY VALIDATION
# ============================================================
with tab_validation:
    st.subheader("Laboratory Validation")
    st.markdown(
        """
        새로운 배합의 **실측 강도와 AI 예측값을 비교**하여 모델 성능을 지속적으로
        확인할 수 있습니다.

        업로드 CSV는 반드시 **스케일링 전 원본 단위**를 사용해야 합니다.
        """
    )

    template_df = pd.DataFrame(
        [
            {
                **DEFAULT_MIX,
                TARGET_COLUMN: 40.0,
            }
        ]
    )

    st.download_button(
        "⬇️ Download Validation CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="validation_template_raw_units.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Validation CSV Upload",
        type=["csv"],
        help=(
            "필수 입력 컬럼: 8개 배합 변수. "
            "성능 지표 계산을 위해 concrete_compressive_strength 컬럼도 포함하세요."
        ),
    )

    if uploaded_file is not None:
        try:
            validation_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"CSV 파일을 읽을 수 없습니다: {exc}")
            validation_df = None

        if validation_df is not None:
            missing_features = [
                feature
                for feature in FEATURE_COLUMNS
                if feature not in validation_df.columns
            ]

            if missing_features:
                st.error(
                    "다음 필수 컬럼이 없습니다: "
                    + ", ".join(missing_features)
                )
            elif not using_trained_model:
                st.error(
                    "검증 기능에는 실제 model.pkl과 scaler.pkl이 필요합니다."
                )
            else:
                try:
                    validation_predictions = predict_dataframe(
                        validation_df[FEATURE_COLUMNS],
                        model,
                        scaler,
                    )
                except Exception as exc:
                    st.error(f"예측 처리 중 오류가 발생했습니다: {exc}")
                    validation_predictions = None

                if validation_predictions is not None:
                    validation_result = validation_df.copy()
                    validation_result["predicted_strength"] = validation_predictions

                    if TARGET_COLUMN in validation_result.columns:
                        metrics = calculate_validation_metrics(
                            validation_result[TARGET_COLUMN],
                            validation_predictions,
                        )
                        validation_result["residual"] = (
                            validation_result[TARGET_COLUMN]
                            - validation_result["predicted_strength"]
                        )
                        validation_result["absolute_error"] = validation_result[
                            "residual"
                        ].abs()

                        metric_1, metric_2, metric_3 = st.columns(3)
                        metric_1.metric("R²", f"{metrics['R²']:.4f}")
                        metric_2.metric("RMSE", f"{metrics['RMSE']:.3f} MPa")
                        metric_3.metric("MAE", f"{metrics['MAE']:.3f} MPa")

                        validation_left, validation_right = st.columns(2)

                        with validation_left:
                            actual_min = float(
                                validation_result[TARGET_COLUMN].min()
                            )
                            actual_max = float(
                                validation_result[TARGET_COLUMN].max()
                            )

                            actual_predicted_chart = px.scatter(
                                validation_result,
                                x=TARGET_COLUMN,
                                y="predicted_strength",
                                labels={
                                    TARGET_COLUMN: "Actual Strength (MPa)",
                                    "predicted_strength": "Predicted Strength (MPa)",
                                },
                                title="Actual vs Predicted",
                                hover_data=["absolute_error"],
                            )
                            actual_predicted_chart.add_trace(
                                go.Scatter(
                                    x=[actual_min, actual_max],
                                    y=[actual_min, actual_max],
                                    mode="lines",
                                    line={"color": "red", "dash": "dash"},
                                    name="Ideal 1:1 Line",
                                )
                            )
                            st.plotly_chart(
                                actual_predicted_chart,
                                use_container_width=True,
                            )

                        with validation_right:
                            residual_chart = px.histogram(
                                validation_result,
                                x="residual",
                                nbins=30,
                                labels={"residual": "Residual (Actual − Predicted)"},
                                title="Residual Distribution",
                            )
                            residual_chart.add_vline(
                                x=0,
                                line_color="red",
                                line_dash="dash",
                            )
                            st.plotly_chart(
                                residual_chart,
                                use_container_width=True,
                            )

                        st.markdown("#### Largest Prediction Errors")
                        largest_errors = validation_result.nlargest(
                            min(10, len(validation_result)),
                            "absolute_error",
                        )
                        display_columns = (
                            FEATURE_COLUMNS
                            + [
                                TARGET_COLUMN,
                                "predicted_strength",
                                "residual",
                                "absolute_error",
                            ]
                        )
                        st.dataframe(
                            largest_errors[display_columns],
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "⬇️ Download Validation Results",
                            data=validation_result.to_csv(index=False).encode("utf-8-sig"),
                            file_name="validation_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info(
                            f"'{TARGET_COLUMN}' 컬럼이 없어 예측값만 생성했습니다."
                        )
                        st.dataframe(
                            validation_result,
                            use_container_width=True,
                            hide_index=True,
                        )

    st.divider()
    st.markdown("#### Single-Test Comparison")

    single_1, single_2, single_3 = st.columns(3)
    measured_strength = single_1.number_input(
        "Measured Strength (MPa)",
        min_value=0.0,
        value=40.0,
        step=0.5,
    )
    ai_strength = single_2.number_input(
        "AI Prediction (MPa)",
        min_value=0.0,
        value=float(predicted_strength),
        step=0.5,
    )

    difference = measured_strength - ai_strength
    percentage_difference = (
        difference / measured_strength * 100
        if measured_strength > 0
        else 0.0
    )

    single_3.metric(
        "Actual − AI",
        f"{difference:+.2f} MPa",
        delta=f"{percentage_difference:+.1f}%",
    )


# ============================================================
# TAB 4 — COST & SUSTAINABILITY
# ============================================================
with tab_sustainability:
    st.subheader("Cost & Sustainability Assessment")
    st.caption(
        "현재 Mix Design 탭에 입력된 배합을 기준으로 재료별 비용과 GWP 기여도를 계산합니다."
    )

    detail_rows = []

    for feature in MATERIAL_FEATURES:
        quantity = current_mix[feature]
        material_cost = quantity * st.session_state.cost_factors[feature]
        material_gwp = quantity * st.session_state.gwp_factors[feature]

        detail_rows.append(
            {
                "Material": FEATURE_LABELS_KR[feature],
                "Quantity (kg/m³)": round(quantity, 2),
                "Cost ($/m³)": round(material_cost, 3),
                "GWP (kgCO₂e/m³)": round(material_gwp, 3),
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    total_cost = float(detail_df["Cost ($/m³)"].sum())
    total_gwp = float(detail_df["GWP (kgCO₂e/m³)"].sum())

    sustainability_left, sustainability_right = st.columns([1.2, 1])

    with sustainability_left:
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        total_1, total_2 = st.columns(2)
        total_1.metric("Total Cost", f"${total_cost:.2f}/m³")
        total_2.metric("Total GWP", f"{total_gwp:.1f} kgCO₂e/m³")

    with sustainability_right:
        gwp_chart = px.pie(
            detail_df,
            names="Material",
            values="GWP (kgCO₂e/m³)",
            hole=0.45,
            title="Material Contribution to GWP",
        )
        st.plotly_chart(gwp_chart, use_container_width=True)

    efficiency_1, efficiency_2 = st.columns(2)
    efficiency_1.metric(
        "Cost per MPa",
        f"${total_cost / max(predicted_strength, 1e-6):.3f}",
    )
    efficiency_2.metric(
        "GWP per MPa",
        f"{total_gwp / max(predicted_strength, 1e-6):.2f} kgCO₂e",
    )

    st.warning(
        "본 비용 및 GWP 결과는 예시 계수를 사용한 상대 비교용입니다. "
        "실제 프로젝트 적용 전 공급업체 단가, 운송 조건 및 검증된 EPD 값을 반영해야 합니다."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Prototype AI-assisted engineering decision support platform · "
    "XGBoost-based compressive strength prediction · "
    "AI supports engineering judgment and does not replace laboratory or field validation."
)
