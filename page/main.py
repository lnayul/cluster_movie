# main.py — 영화 유형 나누기: 속성을 골라 묶고, 묶음 수도 바꿔 가며 정한다
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.caption("정답 열 없이, 내가 고른 속성이 비슷한 영화끼리 묶습니다.")

MOVIES = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
기호 = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]          # 묶음 번호 대신 사용하는 표시
속성 = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
원래단위 = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]


@st.cache_data
def load_data():
    df = pd.read_csv(MOVIES, dtype={"movieCd": str})
    df = df.sort_values("movieCd").reset_index(drop=True)
    전체편수 = len(df)
    # 네 속성을 만들 수 없는 영화는 뺀다: 값이 없거나 첫 주 관객이 0인 행
    쓸열 = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    df = df.dropna(subset=쓸열 + ["movieNm"])
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0) & (df["first_week_audi"] > 0)]
    df = df.reset_index(drop=True)
    return df, 전체편수


df, 전체편수 = load_data()

# 네 가지 속성을 만든다. 숫자 차이가 너무 큰 둘은 상용로그를 취한다
df["스크린 수"] = df["first_scrn"]
df["누적 관객"] = df["total_audi"]
df["10위권 일수"] = df["days_in_top10"]
df["롱런 지수"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)   # 20을 넘으면 20으로
df["스크린 수(로그)"] = np.log10(df["first_scrn"])
df["누적 관객(로그)"] = np.log10(df["total_audi"])

st.info(f"전체 {전체편수:,}편 가운데 네 속성을 모두 만들 수 있는 {len(df):,}편으로 묶었습니다.")

고른속성 = st.multiselect("묶는 데 사용할 속성 (둘 이상)", 속성, default=속성)
if len(고른속성) < 2:
    st.warning("속성을 둘 이상 골라 주세요.")
    st.stop()

Xs = StandardScaler().fit_transform(df[고른속성])          # 고른 속성을 표준화한다
k = st.slider("몇 묶음으로 나눌까요", 2, 7, 3)


def 나누기(개수):
    """표준화한 값을 k-평균으로 나눈다. 난수를 고정해 다시 실행해도 결과가 같다."""
    return KMeans(n_clusters=개수, n_init=10, random_state=42).fit(Xs)


모델 = 나누기(k)
번호 = 모델.labels_

# 누적 관객 평균이 큰 묶음부터 ㉮·㉯·㉰ 순으로 표시한다
순서 = pd.Series(df["누적 관객"].to_numpy()).groupby(번호).mean().sort_values(ascending=False).index
자리 = {g: i for i, g in enumerate(순서)}
df["묶음"] = [기호[자리[g]] for g in 번호]

st.subheader("묶음 지도 · 2차원")
c1, c2 = st.columns(2)
x축 = c1.selectbox("가로축", 고른속성, index=0)
y축 = c2.selectbox("세로축", 고른속성, index=min(1, len(고른속성) - 1))
fig = px.scatter(df, x=x축, y=y축, color="묶음", hover_name="movieNm",
                 category_orders={"묶음": 기호[:k]})
fig.update_traces(marker=dict(size=9, opacity=0.8))
fig.update_layout(height=460)
st.plotly_chart(fig, width="stretch")
st.caption("점 하나가 영화 한 편입니다. 축을 바꿔 보면 묶음이 나뉘는 축과 섞이는 축이 보입니다.")

st.subheader("묶음 지도 · 3차원")
if len(고른속성) < 3:
    st.info("속성을 셋 이상 고르면 3차원 그림이 표시됩니다.")
else:
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("x축", 고른속성, index=0, key="x3")
    y3 = d2.selectbox("y축", 고른속성, index=1, key="y3")
    z3 = d3.selectbox("z축", 고른속성, index=2, key="z3")
    fig3 = px.scatter_3d(df, x=x3, y=y3, z=z3, color="묶음", hover_name="movieNm",
                         category_orders={"묶음": 기호[:k]})
    fig3.update_traces(marker=dict(size=2, opacity=0.7))
    fig3.update_layout(height=560, legend=dict(orientation="h"))
    st.plotly_chart(fig3, width="stretch")
    st.caption("마우스로 끌면 돌아갑니다. 점에 마우스를 올리면 제목이 보입니다.")

st.subheader("묶음별 편수와 네 속성의 평균")
요약 = df.groupby("묶음")[원래단위].mean().round(2)
요약.insert(0, "편수", df.groupby("묶음").size())
요약 = 요약.reindex(기호[:k])
st.dataframe(요약, width="stretch", column_config={
    "편수": st.column_config.NumberColumn("편수", format="%,d편"),
    "스크린 수": st.column_config.NumberColumn("스크린 수", format="%.1f개"),
    "누적 관객": st.column_config.NumberColumn("누적 관객", format="%,.0f명"),
    "10위권 일수": st.column_config.NumberColumn("10위권 일수", format="%.1f일"),
    "롱런 지수": st.column_config.NumberColumn("롱런 지수", format="%.2f배"),
})
st.caption("평균은 로그를 취하기 전의 원래 단위입니다. 롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이고 20에서 잘랐습니다.")

st.subheader("묶음마다 관객이 많은 다섯 편")
열 = st.columns(k)
for i, 이름 in enumerate(기호[:k]):
    묶음 = df[df["묶음"] == 이름].sort_values("누적 관객", ascending=False).head(5)
    열[i].markdown(f"**{이름} 묶음 ({len(df[df['묶음'] == 이름]):,}편)**")
    열[i].dataframe(pd.DataFrame({"영화": 묶음["movieNm"].to_numpy(),
                                "누적 관객": 묶음["누적 관객"].to_numpy()}),
                   width="stretch", hide_index=True,
                   column_config={"누적 관객": st.column_config.NumberColumn("누적 관객", format="%,d명")})

# 여기부터 ③ — 묶음 수를 몇으로 할지 정하는 화면
st.divider()
st.subheader("묶음 수는 몇이 좋을까")

# 고른 속성으로 다시 계산한다. 묶음 안에서 점들이 중심에서 떨어진 거리의 제곱을 모두 더한 값(k=1~7)
합 = pd.DataFrame({"묶음 수": list(range(1, 8)),
                   "거리 제곱의 합": [round(float(나누기(개수).inertia_), 1) for 개수 in range(1, 8)]})
# 바로 앞 묶음 수에서 줄어든 크기. 맨 첫 줄은 비교할 앞 값이 없다
합["앞보다 줄어든 값"] = ["—" if pd.isna(v) else f"{v:,.1f}" for v in -합["거리 제곱의 합"].diff()]

꺾은선 = px.line(합, x="묶음 수", y="거리 제곱의 합", markers=True)
꺾은선.add_vline(x=k, line_dash="dash", annotation_text=f"지금 고른 묶음 수 {k}")
꺾은선.update_layout(height=380, xaxis_title="묶음 수(k)", yaxis_title="묶음 안 거리 제곱의 합")
st.plotly_chart(꺾은선, width="stretch")
st.caption("묶음 수를 늘리면 값은 반드시 줄어듭니다. 줄어드는 폭이 크게 꺾이는 자리를 찾습니다.")

st.dataframe(합, width="stretch", hide_index=True)

점수 = silhouette_score(Xs, 번호)
st.metric(f"묶음 {k}개일 때의 실루엣 점수", f"{점수:.3f}")
st.caption("실루엣 점수는 -1에서 1 사이이고, 1에 가까울수록 묶음이 뚜렷하다는 뜻입니다. "
           "점수가 가장 높은 묶음 수와 사람이 이해하기 좋은 묶음 수가 늘 같지는 않습니다.")
