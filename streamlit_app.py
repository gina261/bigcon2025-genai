import streamlit as st
import pandas as pd
from pathlib import Path
from google import genai
import plotly.express as px
import io

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Bigcon2025 Yvengers',
    page_icon=':earth_americas:',
    layout="wide",
)

@st.cache_data
def get_data():
    DATA_FILENAME = Path(__file__).parent/'data/data2_cleaned.csv'
    raw_df = pd.read_csv(DATA_FILENAME)
    return raw_df

df2 = get_data()

'''
# :earth_americas: Dataset 2 Visualization
'''

# 대상 컬럼
col = "RC_M1_SAA"

# 값별 카운트
counts = df2[col].value_counts().sort_index()

# ─────────────────────────────────────────────────────────────
# (A) Plotly 막대그래프로 렌더 + 이미지 바이트 추출
# ─────────────────────────────────────────────────────────────
counts_df = counts.reset_index()
counts_df.columns = [col, "count"]

fig = px.bar(
    counts_df, x=col, y="count",
    title=f"{col} 값 분포",
)
fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

st.write("### RC_M1_SAA 분포 (매출금액 구간, Plotly)")
st.plotly_chart(fig, use_container_width=True)

# Plotly Figure → PNG bytes (kaleido 필요)
png_bytes = fig.to_image(format="png", scale=2)  # scale로 해상도 업

# ─────────────────────────────────────────────────────────────
# (B) Gemini 2.5 Flash로 이미지 설명 받기
# ─────────────────────────────────────────────────────────────
with st.expander("🔎 Gemini 2.5 Flash에게 그래프 설명 요청 (펼치기)"):
    api_key = st.text_input("Gemini API Key", type="password")
    user_goal = st.text_area(
        "설명 프롬프트(옵션)",
        value=(
            "이 막대그래프를 한국어로 간결히 설명해줘. "
            "전반적 분포 특징(집중/치우침/희소 값), 눈에 띄는 구간, "
            "가능한 인사이트와 주의점(이미지 해상도상 정밀 수치 언급은 자제)을 포함해줘."
        ),
    )
    run = st.button("이미지로 설명 생성")

    if run:
        if not api_key:
            st.error("API Key를 입력해 주세요.")
            st.stop()

        try:
            client = genai.Client(api_key=api_key)

            # 멀티모달 요청: 텍스트 + 이미지(바이트)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{
                    "role": "user",
                    "parts": [
                        {"text": user_goal},
                        {"inline_data": {"mime_type": "image/png", "data": png_bytes}},
                    ],
                }],
                # 필요시 JSON 모드:
                # generation_config={"response_mime_type": "application/json"}
            )

            st.subheader("🧠 Gemini 설명 결과")
            st.write(resp.text)

        except Exception as e:
            st.error(f"요청 중 오류가 발생했습니다: {e}")

# 참고: 원래 st.bar_chart(counts)를 유지하고 싶다면, 시각화는 Plotly로 한 번 더 그리고
#       그 Plotly 이미지를 모델에 보내는 현재 방식이 가장 안정적입니다.