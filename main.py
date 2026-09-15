import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import urllib.request
import io

# ------------------------------------------------------------
# 페이지 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="서울 100년 기온 변화",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"


# ------------------------------------------------------------
# 한글 폰트 설정 (그래프 깨짐 방지)
# ------------------------------------------------------------
def set_korean_font():
    try:
        font_url = "https://raw.githubusercontent.com/googlefonts/nanum-gothic/main/fonts/ttf/NanumGothic-Regular.ttf"
        font_path = "/tmp/NanumGothic-Regular.ttf"
        try:
            urllib.request.urlretrieve(font_url, font_path)
            fm.fontManager.addfont(font_path)
            mpl.rc("font", family=fm.FontProperties(fname=font_path).get_name())
        except Exception:
            # 폰트를 내려받지 못하면 시스템에 있는 한글 폰트를 찾아 사용
            candidates = ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"]
            available = {f.name for f in fm.fontManager.ttflist}
            for c in candidates:
                if c in available:
                    mpl.rc("font", family=c)
                    break
    except Exception:
        pass
    mpl.rcParams["axes.unicode_minus"] = False


set_korean_font()


# ------------------------------------------------------------
# 데이터 불러오기 및 전처리
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data(show_spinner=False)
def compute_yearly(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            최저기온=("최저기온", "mean"),
            최고기온=("최고기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )
    # 관측 일수가 너무 적은(자료 결손이 심한) 해는 제외해서 왜곡을 줄임
    yearly = yearly[yearly["관측일수"] >= 300].reset_index(drop=True)
    return yearly


with st.spinner("서울 기온 데이터를 불러오는 중입니다..."):
    raw_df = load_data()

yearly_df = compute_yearly(raw_df)

min_year = int(yearly_df["연도"].min())
max_year = int(yearly_df["연도"].max())

# ------------------------------------------------------------
# 사이드바
# ------------------------------------------------------------
st.sidebar.header("⚙️ 옵션")

year_range = st.sidebar.slider(
    "연도 범위 선택",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

show_minmax = st.sidebar.checkbox("연평균 최저·최고기온도 함께 보기", value=False)
show_trend = st.sidebar.checkbox("추세선(선형회귀) 표시", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**데이터 출처**  \n"
    "[서울 기온 관측 데이터 (greatsong/modudata)]"
    f"({DATA_URL})"
)

filtered = yearly_df[
    (yearly_df["연도"] >= year_range[0]) & (yearly_df["연도"] <= year_range[1])
].copy()

# ------------------------------------------------------------
# 헤더
# ------------------------------------------------------------
st.title("🌡️ 서울, 100년의 기온 변화")
st.markdown(
    f"서울 기상 관측 데이터를 바탕으로 **{min_year}년부터 {max_year}년까지** "
    "연평균 기온이 어떻게 변해왔는지 한눈에 살펴봅니다."
)

# ------------------------------------------------------------
# 요약 지표
# ------------------------------------------------------------
first_temp = filtered.iloc[0]["평균기온"]
last_temp = filtered.iloc[-1]["평균기온"]
temp_diff = last_temp - first_temp

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"{int(filtered.iloc[0]['연도'])}년 연평균 기온", f"{first_temp:.1f} ℃")
col2.metric(f"{int(filtered.iloc[-1]['연도'])}년 연평균 기온", f"{last_temp:.1f} ℃")
col3.metric("전체 기간 기온 변화", f"{temp_diff:+.1f} ℃")
col4.metric("역대 가장 더웠던 해", f"{int(filtered.loc[filtered['평균기온'].idxmax(), '연도'])}년")

st.markdown("---")

# ------------------------------------------------------------
# 메인 그래프
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    filtered["연도"],
    filtered["평균기온"],
    color="#d62728",
    linewidth=1.8,
    label="연평균 기온",
)

if show_minmax:
    ax.plot(
        filtered["연도"],
        filtered["최고기온"],
        color="#ff9896",
        linewidth=1,
        linestyle="--",
        label="연평균 최고기온",
        alpha=0.8,
    )
    ax.plot(
        filtered["연도"],
        filtered["최저기온"],
        color="#1f77b4",
        linewidth=1,
        linestyle="--",
        label="연평균 최저기온",
        alpha=0.8,
    )

if show_trend and len(filtered) > 1:
    x = filtered["연도"].values.astype(float)
    y = filtered["평균기온"].values.astype(float)
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    ax.plot(
        x,
        trend(x),
        color="black",
        linewidth=2,
        linestyle=":",
        label=f"추세선 (10년당 {coeffs[0]*10:+.2f} ℃)",
    )

ax.set_xlabel("연도")
ax.set_ylabel("기온 (℃)")
ax.set_title("서울 연평균 기온 변화 추이")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# ------------------------------------------------------------
# 추가 설명 / 데이터 표
# ------------------------------------------------------------
with st.expander("📋 연도별 데이터 표 보기"):
    display_df = filtered.rename(
        columns={
            "연도": "연도",
            "평균기온": "연평균 기온(℃)",
            "최저기온": "연평균 최저기온(℃)",
            "최고기온": "연평균 최고기온(℃)",
            "관측일수": "관측일수",
        }
    ).round(1)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(
    "※ 연평균 기온은 해당 연도의 일별 평균기온을 산술평균한 값이며, "
    "관측일수가 300일 미만인 해는 결측이 많아 분석에서 제외했습니다."
)
