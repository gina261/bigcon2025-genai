# pip install streamlit plotly kaleido google-genai pandas
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from google import genai

# ─────────────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="범주형 지표(1~6) 일괄 시각화", layout="wide")
st.title("범주형 지표(1~6) 일괄 시각화")

@st.cache_data
def load_df():
    data_path = Path(__file__).parent / "data" / "data2_cleaned.csv"
    return pd.read_csv(data_path)

df = load_df()

# ─────────────────────────────────────────────────────────────
# 1) 1~6 구간 컬럼 자동 탐색
# ─────────────────────────────────────────────────────────────
CATS = [1, 2, 3, 4, 5, 6]

def is_1to6_column(s: pd.Series) -> bool:
    v = pd.Series(s.dropna().unique())
    return v.isin(CATS).all()

candidate_cols = [c for c in df.columns if is_1to6_column(df[c])]

if not candidate_cols:
    st.warning("1~6 값만 갖는 범주형 컬럼을 찾지 못했습니다.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 2) 옵션(본문 상단에 간단히 고정)
# ─────────────────────────────────────────────────────────────
wrap = 3                # 한 줄에 3개 facet
as_percent = True       # 퍼센트 보기
log_y = False           # 로그축 여부
height_per_row = 320    # 행당 높이(px) — 간격 확보를 위해 크게
font_size = 12          # 글꼴 크기

# ─────────────────────────────────────────────────────────────
# 3) 긴 포맷으로 변환
# ─────────────────────────────────────────────────────────────
long_df = df[candidate_cols].melt(var_name="metric", value_name="bin").dropna()
if pd.api.types.is_float_dtype(long_df["bin"]):
    long_df["bin"] = long_df["bin"].astype(int)

metrics = sorted(long_df["metric"].unique())
data_page = long_df[long_df["metric"].isin(metrics)]

# ─────────────────────────────────────────────────────────────
# 4) Facet Grid(작은 배치) 생성
# ─────────────────────────────────────────────────────────────
histnorm = "percent" if as_percent else None
fig = px.histogram(
    data_page,
    x="bin",
    facet_col="metric",
    facet_col_wrap=wrap,
    category_orders={"bin": CATS},
    histfunc="count",
    histnorm=histnorm,
    nbins=6,
    text_auto=True,
    facet_row_spacing=0.16,   # 행 간 간격 ↑
    facet_col_spacing=0.07    # 열 간 간격 약간 ↑
)

# 라벨/레이아웃 정리 (겹침 방지)
fig.update_traces(
    texttemplate="%{y:.2f}%" if as_percent else "%{y}",
    textposition="outside",
    cliponaxis=False,
    marker_line_width=0,
    hovertemplate="구간 %{x}<br>값 %{y}"
)
fig.update_xaxes(title_text="구간 (1~6)", dtick=1)
ylab = "비율(%)" if as_percent else "개수"
fig.update_yaxes(title_text=ylab, rangemode="tozero", type="log" if log_y else "linear")

# 모든 y축 동일 스케일로 매칭
first_yaxis_key = next(k for k in fig.layout if k.startswith("yaxis"))
fig.update_yaxes(matches=first_yaxis_key.replace("axis", ""))  # 'yaxis' → 'y'

# 전체 높이/폰트/여백 등
n_rows = (len(metrics) + wrap - 1) // wrap
fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    height=n_rows * height_per_row,
    bargap=0.25,
    font=dict(size=font_size),
    showlegend=False,
)

# facet 제목 살짝 위로 띄우고 볼드 처리
for ann in fig.layout.annotations:
    ann.font.size = font_size + 1
    ann.yshift = 12
    ann.text = f"<b>{ann.text}</b>"

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ─────────────────────────────────────────────────────────────
# 5) (선택) 한 장 이미지를 Gemini로 보내 요약 받기
# ─────────────────────────────────────────────────────────────
with st.expander("🔎 Gemini 2.5 Flash로 전체 그리드 설명 받기"):
    api_key = st.text_input("Gemini API Key", type="password")
    prompt = st.text_area(
        "프롬프트",
        value=("각 facet(지표)별 1~6 구간 분포 특징을 요약하고, 공통점/차이점을 비교해줘. "
               "극단적으로 치우친 지표도 지적하되, 정밀 수치 확정은 피하라."),
    )
    if st.button("설명 생성"):
        if not api_key:
            st.error("API Key를 입력해 주세요.")
        else:
            try:
                img_bytes = fig.to_image(format="png", scale=2)  # kaleido 필요
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[{
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/png", "data": img_bytes}}
                        ]
                    }]
                )
                st.subheader("🧠 Gemini 설명 결과")
                st.write(resp.text)
            except Exception as e:
                st.error(f"요청 중 오류가 발생했습니다: {e}")
                
                
# ─────────────────────────────────────────────────────────────
# 6) 연속형 지표(비율) 일괄 시각화 섹션
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.header("연속형 지표 일괄 시각화")

# 연속형 후보 목록 (데이터에 실제 존재하는 것만 사용)
CONT_COLS_ALL = [
    "DLV_SAA_RAT",           # 배달매출금액 비율
    "M1_SME_RY_SAA_RAT",     # 동일 업종 매출금액 비율
    "M1_SME_RY_CNT_RAT",     # 동일 업종 매출건수 비율
    "M12_SME_RY_SAA_PCE_RT", # 동일 상권 내 매출 순위 비율
    "M12_SME_RY_ME_MCT_RAT", # 동일 상권 내 매출 가맹점 비중
    "M12_SME_BZN_SAA_PCE_RT",# 동일 상권 내 매출 순위 비중
    "M12_SME_BZN_ME_MCT_RAT" # 동일 상권 내 매출 가맹점 비중
]
cont_cols = [c for c in CONT_COLS_ALL if c in df.columns]
if not cont_cols:
    st.info("연속형 지표 컬럼을 데이터에서 찾지 못했습니다.")
    st.stop()

# 보기 좋은 한글 라벨(있으면 적용)
KOR_LABEL = {
    "DLV_SAA_RAT": "배달매출금액 비율",
    "M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "M12_SME_RY_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "M12_SME_RY_ME_MCT_RAT": "동일 상권 내 매출 가맹점 비중",
    "M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비중",
    "M12_SME_BZN_ME_MCT_RAT": "동일 상권 내 매출 가맹점 비중",
}

# ── 옵션
wrap_cont = 2             # 한 줄 2개
bins = st.slider("Bins(막대 개수)", 10, 80, 40)
q_low, q_high = st.slider("분위수 클리핑(이상치 완화)", 0.0, 0.1, (0.01, 0.99))
# log_x = st.checkbox("로그 X축", value=False)
zscore = st.checkbox("표준화(z-score)", value=False)
height_per_row_cont = 340
font_size_cont = 12

# ── 데이터 전처리
df_cont = df[cont_cols].copy()

# -999999.9 값 제외 (DLV_SAA_RAT, M12_SME_BZN_ME_MCT_RAT)
for bad_col in ["DLV_SAA_RAT", "M12_SME_BZN_ME_MCT_RAT"]:
    if bad_col in df_cont.columns:
        df_cont.loc[df_cont[bad_col] == -999999.9, bad_col] = None

# 분위수 클리핑
for c in cont_cols:
    s = df_cont[c].dropna()
    if len(s) == 0:
        continue
    lo, hi = s.quantile([q_low, q_high])
    df_cont[c] = df_cont[c].clip(lower=lo, upper=hi)

# 표준화 옵션
if zscore:
    df_cont = (df_cont - df_cont.mean(numeric_only=True)) / df_cont.std(numeric_only=True)

# 긴 포맷
long_cont = df_cont.melt(var_name="metric", value_name="val").dropna()
long_cont["metric"] = long_cont["metric"].map(lambda x: KOR_LABEL.get(x, x))

# ── 히스토그램 생성
fig2 = px.histogram(
    long_cont,
    x="val",
    facet_col="metric",
    facet_col_wrap=wrap_cont,
    histnorm="percent",
    nbins=bins,
    opacity=0.95,
    text_auto=True,
    facet_row_spacing=0.16,
    facet_col_spacing=0.08
)
fig2.update_traces(
    texttemplate="%{y:.2f}%",
    textposition="outside",
    cliponaxis=False,
    hovertemplate="값 %{x}<br>비율 %{y:.2f}%"
)
fig2.update_yaxes(title_text="비율(%)", rangemode="tozero")
# if log_x:
    # fig2.update_xaxes(type="log")
fig2.update_xaxes(title_text="값", showgrid=True, zeroline=False)

# 레이아웃
n_rows2 = (len(long_cont["metric"].unique()) + wrap_cont - 1) // wrap_cont
fig2.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    height=n_rows2 * height_per_row_cont,
    bargap=0.05,
    font=dict(size=font_size_cont),
    showlegend=False,
)
for ann in fig2.layout.annotations:
    ann.font.size = font_size_cont + 1
    ann.yshift = 12
    ann.text = f"<b>{ann.text.replace('metric=', '')}</b>"

st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

# ── Gemini 설명 (선택)
with st.expander("🔎 Gemini 2.5 Flash로 연속형 그리드 설명 받기"):
    api_key2 = st.text_input("Gemini API Key (연속형)", type="password", key="k2")
    prompt2 = st.text_area(
        "프롬프트(연속형)",
        value=("각 지표의 분포 모양(대칭/왜도/첨도)과 이상치 가능 구간을 요약해줘. "
               "-999999.9 값은 제외했음을 감안해서 설명해줘."),
    )
    if st.button("설명 생성 (연속형)"):
        if not api_key2:
            st.error("API Key를 입력해 주세요.")
        else:
            try:
                img2 = fig2.to_image(format="png", scale=2)
                client2 = genai.Client(api_key=api_key2)
                resp2 = client2.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[{
                        "role": "user",
                        "parts": [
                            {"text": prompt2},
                            {"inline_data": {"mime_type": "image/png", "data": img2}}
                        ]
                    }]
                )
                st.subheader("🧠 Gemini 설명 결과 (연속형)")
                st.write(resp2.text)
            except Exception as e:
                st.error(f"요청 중 오류가 발생했습니다: {e}")