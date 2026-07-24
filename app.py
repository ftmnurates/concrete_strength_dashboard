"""
콘크리트 압축강도 예측 대시보드
================================
Day 4-5 프로젝트에서 학습한 모델(기본값: XGBoost 튜닝 모델)을 사용하는
Streamlit 대시보드입니다.

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py

모델 파일이 없어도 실행은 되지만(데모 모드, 근사 공식 사용), 실제 예측 정확도를
위해서는 model.pkl과 scaler.pkl을 이 파일과 같은 폴더에 두세요.
(model.pkl / scaler.pkl 만드는 방법은 save_model_snippet.py 참고)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    import joblib
except ImportError:
    joblib = None


# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(
    page_title="콘크리트 압축강도 예측 대시보드",
    page_icon="🧱",
    layout="wide",
)

FEATURE_COLUMNS = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
]
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
    "cement": "kg/m³", "blast_furnace_slag": "kg/m³", "fly_ash": "kg/m³",
    "water": "kg/m³", "superplasticizer": "kg/m³", "coarse_aggregate": "kg/m³",
    "fine_aggregate": "kg/m³", "age": "일",
}
# Day 4 EDA에서 확인한 학습 데이터의 실제 범위 (min, max) — 슬라이더 기본값 및
# "훈련 범위 밖" 경고에 사용. 본인 데이터의 실제 min/max로 교체하세요.
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
    "cement": 300.0, "blast_furnace_slag": 60.0, "fly_ash": 30.0,
    "water": 180.0, "superplasticizer": 6.0, "coarse_aggregate": 970.0,
    "fine_aggregate": 780.0, "age": 28.0,
}

# 재료별 단가($/kg)와 GWP(kgCO2e/kg) 기본값 — 업계 평균을 참고한 예시값입니다.
# 실제 프로젝트에서는 지역·공급업체별 실제 단가/EPD(환경성적표지) 데이터로 교체하세요.
DEFAULT_COST_PER_KG = {
    "cement": 0.12, "blast_furnace_slag": 0.03, "fly_ash": 0.02,
    "water": 0.001, "superplasticizer": 2.00, "coarse_aggregate": 0.015,
    "fine_aggregate": 0.015,
}
DEFAULT_GWP_PER_KG = {  # kg CO2e / kg material
    "cement": 0.90, "blast_furnace_slag": 0.03, "fly_ash": 0.02,
    "water": 0.0003, "superplasticizer": 1.20, "coarse_aggregate": 0.007,
    "fine_aggregate": 0.007,
}
MIX_FEATURES = [c for c in FEATURE_COLUMNS if c != "age"]  # cost/GWP는 age 제외 7개 재료 기준


# ============================================================
# 모델 로딩
# ============================================================
@st.cache_resource
def load_model_and_scaler():
    """model.pkl / scaler.pkl 을 시도해서 불러온다. 없으면 (None, None) 반환."""
    model, scaler = None, None
    if joblib is not None:
        if os.path.exists("model.pkl"):
            try:
                model = joblib.load("model.pkl")
            except Exception as e:
                st.warning(f"model.pkl 로드 실패: {e}")
        if os.path.exists("scaler.pkl"):
            try:
                scaler = joblib.load("scaler.pkl")
            except Exception as e:
                st.warning(f"scaler.pkl 로드 실패: {e}")
    return model, scaler


def demo_formula_predict(mix: dict) -> float:
    """model.pkl이 없을 때 사용하는 아주 단순한 근사 공식 (데모 전용).
    실제 학습된 모델과 무관하며, 대시보드 기능을 시연하기 위한 대체값입니다."""
    c = mix["cement"]
    w = mix["water"]
    a = mix["age"]
    binder = c + mix["blast_furnace_slag"] + mix["fly_ash"]
    wb_ratio = w / max(binder, 1e-6)
    # Abrams' law 스타일의 대략적인 근사식 (데모 목적)
    base = 100.0 / (1.5 ** (wb_ratio * 3))
    age_factor = np.log1p(a) / np.log1p(28)
    strength = base * min(age_factor, 1.3)
    return float(np.clip(strength, 2, 85))


def predict_strength(mix: dict, model, scaler) -> tuple[float, bool]:
    """예측 강도(MPa)와 '실제 모델 사용 여부'를 반환."""
    if model is not None and scaler is not None:
        X = pd.DataFrame([[mix[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        return float(pred), True
    return demo_formula_predict(mix), False


def out_of_range_warnings(mix: dict) -> list[str]:
    warnings = []
    for feat, (lo, hi) in TRAIN_RANGES.items():
        v = mix[feat]
        if v < lo or v > hi:
            warnings.append(
                f"**{FEATURE_LABELS_KR[feat]}** = {v:g}{FEATURE_UNITS[feat]} — "
                f"학습 데이터 범위({lo:g}~{hi:g}) 밖입니다. 이 구간의 예측은 신뢰도가 낮을 수 있습니다."
            )
    return warnings


def compute_cost_gwp(mix: dict, cost_table: dict, gwp_table: dict) -> tuple[float, float]:
    total_cost = sum(mix[f] * cost_table[f] for f in MIX_FEATURES)
    total_gwp = sum(mix[f] * gwp_table[f] for f in MIX_FEATURES)
    return total_cost, total_gwp


# ============================================================
# 세션 상태 초기화
# ============================================================
if "saved_mixes" not in st.session_state:
    st.session_state.saved_mixes = []  # list of dicts: {name, mix, pred, cost, gwp}

if "cost_table" not in st.session_state:
    st.session_state.cost_table = DEFAULT_COST_PER_KG.copy()

if "gwp_table" not in st.session_state:
    st.session_state.gwp_table = DEFAULT_GWP_PER_KG.copy()


model, scaler = load_model_and_scaler()
USING_REAL_MODEL = model is not None and scaler is not None


# ============================================================
# 사이드바 — 재료 단가 / GWP 계수 설정 (모든 탭에서 공유)
# ============================================================
with st.sidebar:
    st.header("⚙️ 설정")

    if USING_REAL_MODEL:
        st.success("학습된 모델(model.pkl) 사용 중")
    else:
        st.warning("model.pkl 없음 — 데모 근사 공식 사용 중\n(실제 프로젝트 모델을 연결하세요)")

    st.markdown("---")
    st.subheader("재료 단가 ($/kg)")
    for f in MIX_FEATURES:
        st.session_state.cost_table[f] = st.number_input(
            FEATURE_LABELS_KR[f], min_value=0.0,
            value=float(st.session_state.cost_table[f]), step=0.001, format="%.3f",
            key=f"cost_{f}",
        )

    st.markdown("---")
    st.subheader("GWP 계수 (kgCO₂e/kg)")
    for f in MIX_FEATURES:
        st.session_state.gwp_table[f] = st.number_input(
            FEATURE_LABELS_KR[f], min_value=0.0,
            value=float(st.session_state.gwp_table[f]), step=0.01, format="%.3f",
            key=f"gwp_{f}",
        )
    st.caption("⚠️ 위 단가/GWP 값은 예시입니다. 실제 지역·공급업체 데이터로 교체하세요.")


# ============================================================
# 헤더
# ============================================================
st.title("🧱 콘크리트 압축강도 예측 대시보드")
st.caption("Day 4–5 프로젝트 · UCI Concrete Compressive Strength 데이터 기반")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 예측", "⚖️ 배합 비교", "📈 실측 vs 예측", "💰 비용 & GWP",
])


# ============================================================
# TAB 1 — 예측
# ============================================================
with tab1:
    st.subheader("배합 입력")
    col_input, col_result = st.columns([1.3, 1])

    with col_input:
        mix = {}
        c1, c2 = st.columns(2)
        SLIDER_STEP = {
            "cement": 5.0, "blast_furnace_slag": 5.0, "fly_ash": 5.0,
            "water": 1.0, "superplasticizer": 0.5, "coarse_aggregate": 5.0,
            "fine_aggregate": 5.0, "age": 1.0,
        }
        for i, feat in enumerate(FEATURE_COLUMNS):
            lo, hi = TRAIN_RANGES[feat]
            target_col = c1 if i % 2 == 0 else c2
            mix[feat] = target_col.slider(
                f"{FEATURE_LABELS_KR[feat]} ({FEATURE_UNITS[feat]})",
                min_value=float(lo * 0.5), max_value=float(hi * 1.3),
                value=float(DEFAULT_MIX[feat]), step=SLIDER_STEP[feat],
                key=f"pred_{feat}",
            )

        mix_name = st.text_input("배합 이름 (비교 탭에 저장할 때 사용)", value="배합 A")
        save_clicked = st.button("💾 이 배합을 비교 목록에 저장", use_container_width=True)

    pred, is_real = predict_strength(mix, model, scaler)
    cost, gwp = compute_cost_gwp(mix, st.session_state.cost_table, st.session_state.gwp_table)

    with col_result:
        st.metric("예측 압축강도", f"{pred:.1f} MPa")
        m1, m2 = st.columns(2)
        m1.metric("배합 비용 (1m³ 기준)", f"${cost:.2f}")
        m2.metric("GWP (1m³ 기준)", f"{gwp:.1f} kgCO₂e")

        if not is_real:
            st.info("현재 데모 근사 공식으로 계산되었습니다. model.pkl / scaler.pkl을 연결하면 실제 XGBoost 모델 예측으로 바뀝니다.")

        warns = out_of_range_warnings(mix)
        if warns:
            with st.expander(f"⚠️ 학습 범위 밖 입력 {len(warns)}건", expanded=True):
                for w in warns:
                    st.markdown(f"- {w}")

    if save_clicked:
        st.session_state.saved_mixes.append({
            "name": mix_name, "mix": mix.copy(), "pred": pred, "cost": cost, "gwp": gwp,
        })
        st.success(f"'{mix_name}' 배합이 저장되었습니다 — '⚖️ 배합 비교' 탭에서 확인하세요.")


# ============================================================
# TAB 2 — 배합 비교
# ============================================================
with tab2:
    st.subheader("저장된 배합 비교")

    if not st.session_state.saved_mixes:
        st.info("아직 저장된 배합이 없습니다. '🔮 예측' 탭에서 배합을 만들고 저장해보세요.")
    else:
        rows = []
        for m in st.session_state.saved_mixes:
            row = {"이름": m["name"], "예측 강도(MPa)": round(m["pred"], 1),
                   "비용($/m³)": round(m["cost"], 2), "GWP(kgCO₂e/m³)": round(m["gwp"], 1)}
            row["강도/비용 효율"] = round(m["pred"] / max(m["cost"], 1e-6), 2)
            row["강도/GWP 효율"] = round(m["pred"] / max(m["gwp"], 1e-6), 3)
            rows.append(row)
        df_compare = pd.DataFrame(rows)

        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_compare["이름"], y=df_compare["예측 강도(MPa)"],
                                  marker_color="#1E2761", name="예측 강도"))
            fig.update_layout(title="배합별 예측 압축강도", yaxis_title="MPa", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_compare["이름"], y=df_compare["비용($/m³)"],
                                   marker_color="#FF6B35", name="비용", yaxis="y"))
            fig2.add_trace(go.Scatter(x=df_compare["이름"], y=df_compare["GWP(kgCO₂e/m³)"],
                                       mode="lines+markers", name="GWP", yaxis="y2",
                                       marker_color="#2C5F2D"))
            fig2.update_layout(
                title="배합별 비용 vs GWP",
                yaxis=dict(title="비용 ($/m³)"),
                yaxis2=dict(title="GWP (kgCO₂e/m³)", overlaying="y", side="right"),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

        best_value = df_compare.loc[df_compare["강도/비용 효율"].idxmax(), "이름"]
        best_gwp = df_compare.loc[df_compare["강도/GWP 효율"].idxmax(), "이름"]
        st.success(f"💰 비용 대비 강도 효율 최고: **{best_value}**  ·  🌱 GWP 대비 강도 효율 최고: **{best_gwp}**")

        if st.button("🗑️ 저장된 배합 전체 삭제"):
            st.session_state.saved_mixes = []
            st.rerun()


# ============================================================
# TAB 3 — 실측 vs 예측
# ============================================================
with tab3:
    st.subheader("실측값 vs 예측값 비교")
    st.caption("Day 4의 test_processed.csv 형식(8개 feature + concrete_compressive_strength 컬럼)의 CSV를 업로드하세요.")

    uploaded = st.file_uploader("CSV 업로드", type=["csv"])

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV를 읽을 수 없습니다: {e}")
            df_up = None

        if df_up is not None:
            target_col = "concrete_compressive_strength"
            missing = [c for c in FEATURE_COLUMNS if c not in df_up.columns]
            if missing:
                st.error(f"다음 컬럼이 CSV에 없습니다: {missing}")
            else:
                preds = []
                for _, row in df_up.iterrows():
                    mix_row = {f: row[f] for f in FEATURE_COLUMNS}
                    p, _ = predict_strength(mix_row, model, scaler)
                    preds.append(p)
                df_up["predicted"] = preds

                if target_col in df_up.columns:
                    y_true = df_up[target_col]
                    y_pred = df_up["predicted"]
                    residuals = y_true - y_pred
                    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                    mae = float(np.mean(np.abs(residuals)))
                    ss_res = float(np.sum(residuals ** 2))
                    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("R²", f"{r2:.4f}")
                    m2.metric("RMSE", f"{rmse:.3f} MPa")
                    m3.metric("MAE", f"{mae:.3f} MPa")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig = px.scatter(df_up, x=target_col, y="predicted",
                                          labels={target_col: "실제값 (MPa)", "predicted": "예측값 (MPa)"},
                                          title="Actual vs Predicted")
                        lo, hi = float(y_true.min()), float(y_true.max())
                        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                                  line=dict(color="red", dash="dash"), name="perfect fit"))
                        st.plotly_chart(fig, use_container_width=True)
                    with col_b:
                        fig2 = px.histogram(residuals, nbins=30, labels={"value": "Residual (실제-예측)"},
                                             title="Residual 분포")
                        fig2.add_vline(x=0, line_color="red", line_dash="dash")
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("업로드한 CSV에 실제값 컬럼이 없어 예측값만 표시합니다.")
                    st.dataframe(df_up, use_container_width=True)

    st.markdown("---")
    st.subheader("단건 수동 비교")
    c1, c2, c3 = st.columns(3)
    manual_actual = c1.number_input("실제 측정 강도 (MPa)", min_value=0.0, value=40.0, step=0.5)
    manual_pred = c2.number_input("예측 강도 (MPa)", min_value=0.0, value=float(pred), step=0.5)
    diff = manual_actual - manual_pred
    c3.metric("차이 (실제 - 예측)", f"{diff:+.2f} MPa", delta=f"{(diff/max(manual_actual,1e-6))*100:+.1f}%")


# ============================================================
# TAB 4 — 비용 & GWP 계산기
# ============================================================
with tab4:
    st.subheader("배합별 비용 & GWP 상세 분해")
    st.caption("좌측 사이드바에서 단가/GWP 계수를 조정할 수 있습니다.")

    mix_for_calc = mix  # TAB 1에서 마지막으로 입력한 배합 재사용

    detail_rows = []
    for f in MIX_FEATURES:
        amt = mix_for_calc[f]
        c = amt * st.session_state.cost_table[f]
        g = amt * st.session_state.gwp_table[f]
        detail_rows.append({
            "재료": FEATURE_LABELS_KR[f],
            "사용량 (kg/m³)": round(amt, 1),
            "비용 ($/m³)": round(c, 3),
            "GWP (kgCO₂e/m³)": round(g, 3),
        })
    df_detail = pd.DataFrame(detail_rows)
    total_cost = df_detail["비용 ($/m³)"].sum()
    total_gwp = df_detail["GWP (kgCO₂e/m³)"].sum()

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.dataframe(df_detail, use_container_width=True, hide_index=True)
        st.markdown(f"**합계 — 비용: ${total_cost:.2f}/m³, GWP: {total_gwp:.1f} kgCO₂e/m³**")

    with col_b:
        fig = px.pie(df_detail, names="재료", values="GWP (kgCO₂e/m³)",
                      title="재료별 GWP 기여도", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    pred_calc, _ = predict_strength(mix_for_calc, model, scaler)
    if pred_calc > 0:
        st.info(
            f"이 배합의 예측 강도는 **{pred_calc:.1f} MPa**입니다. "
            f"MPa당 비용은 **${total_cost/pred_calc:.3f}**, MPa당 GWP는 **{total_gwp/pred_calc:.2f} kgCO₂e**입니다. "
            "(값이 낮을수록 '단위 강도당' 효율이 좋다는 의미입니다.)"
        )


st.markdown("---")
st.caption(
    "이 대시보드는 Day 4–5 프로젝트(UCI Concrete Compressive Strength, XGBoost R²=0.9318)를 기반으로 합니다. "
    "비용/GWP 계수는 예시값이며 실제 프로젝트에는 검증된 데이터로 교체해야 합니다."
)
