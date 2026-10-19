import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==================================================
# 1. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="영화 유형 분석 연구소",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

st.title("🎬 영화 유형 분석 연구소")
st.caption(
    "영화의 흥행 규모와 상영 지속성을 바탕으로 "
    "K-평균 군집 분석을 수행합니다."
)

st.markdown(
    """
    영화의 누적 관객 수만으로는 영화의 흥행 특성을
    모두 설명하기 어렵습니다.

    이 앱에서는 스크린 수, 누적 관객, 10위권 일수,
    롱런 지수를 함께 분석해 영화의 유형을 탐색합니다.
    """
)

# ==================================================
# 2. 데이터 불러오기
# ==================================================
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    required_cols = [
        "movieNm",
        "first_scrn",
        "total_audi",
        "first_week_audi",
        "days_in_top10",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"CSV에서 필요한 열을 찾을 수 없습니다: {missing}"
        )

    total_count = len(df)

    for col in [
        "first_scrn",
        "total_audi",
        "first_week_audi",
        "days_in_top10",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 분석에 필요한 값이 유효한 영화만 사용
    valid = (
        df["first_scrn"].notna()
        & df["total_audi"].notna()
        & df["first_week_audi"].notna()
        & df["days_in_top10"].notna()
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
        & (df["first_week_audi"] > 0)
        & (df["days_in_top10"] >= 0)
    )

    data = df.loc[valid].copy()
    data = data.reset_index(drop=True)

    # 원래 단위의 값 보존
    data["스크린 수(원래값)"] = data["first_scrn"]
    data["누적 관객(원래값)"] = data["total_audi"]
    data["10위권 일수(원래값)"] = data["days_in_top10"]

    # 흥행 규모의 큰 편차를 줄이기 위한 로그 변환
    data["스크린 수(로그10)"] = np.log10(data["first_scrn"])
    data["누적 관객(로그10)"] = np.log10(data["total_audi"])

    data["10위권 일수"] = data["days_in_top10"]

    # 첫 주 관객 대비 누적 관객의 비율
    # 극단적으로 큰 값은 20으로 제한
    data["롱런 지수"] = (
        data["total_audi"] / data["first_week_audi"]
    ).clip(upper=20)

    data["movieNm"] = data["movieNm"].fillna("제목 미상").astype(str)

    return data, total_count


try:
    data, total_count = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("데이터 주소와 CSV 파일의 열 이름을 확인해 주세요.")
    st.stop()

if len(data) < 3:
    st.error("군집 분석을 하려면 유효한 영화가 최소 3편 필요합니다.")
    st.stop()

FEATURES = [
    "스크린 수(로그10)",
    "누적 관객(로그10)",
    "10위권 일수",
    "롱런 지수",
]

# ==================================================
# 3. 데이터 현황
# ==================================================
st.subheader("1. 데이터 현황")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 데이터", f"{total_count:,}편")
c2.metric("분석 가능 영화", f"{len(data):,}편")
c3.metric("분석 제외", f"{total_count - len(data):,}편")
c4.metric("분석 속성", "4개")

with st.expander("분석 속성의 의미 알아보기"):
    st.markdown(
        """
        - **스크린 수(로그10):** 영화가 개봉할 때 확보한 스크린 수의 로그값
        - **누적 관객(로그10):** 누적 관객 수의 로그값
        - **10위권 일수:** 일별 박스오피스 10위권에 포함된 일수
        - **롱런 지수:** 누적 관객 수를 첫 주 관객 수로 나눈 값

        로그 변환은 흥행 규모가 매우 큰 영화와 작은 영화 사이의
        수치 차이를 줄이기 위해 사용합니다.

        롱런 지수는 상영 기간 자체가 아니라 첫 주 관객 대비
        누적 관객 규모를 나타내는 참고 지표입니다.
        """
    )

# ==================================================
# 4. 분석 설정
# ==================================================
st.divider()
st.subheader("2. 군집 분석 설정")

col1, col2 = st.columns([1, 2])

with col1:
    n_clusters = st.slider(
        "분류할 영화 유형 수",
        min_value=2,
        max_value=min(6, len(data) - 1),
        value=min(3, len(data) - 1),
        step=1,
        help="군집 수가 많아지면 영화 유형이 더 세분화됩니다.",
    )

with col2:
    selected_features = st.multiselect(
        "군집 분석에 사용할 속성",
        options=FEATURES,
        default=FEATURES,
        help="최소 2개를 선택하세요.",
    )

if len(selected_features) < 2:
    st.warning("분석 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# ==================================================
# 5. K-평균 분석
# ==================================================
X = data[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if np.unique(X_scaled, axis=0).shape[0] < n_clusters:
    st.error(
        "서로 다른 속성 조합의 영화가 부족합니다. "
        "군집 수를 줄이거나 분석 속성을 변경해 주세요."
    )
    st.stop()

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=20,
)

data["군집번호"] = kmeans.fit_predict(X_scaled)

# 군집 번호 자체에는 순위 의미가 없으므로
# 누적 관객 평균이 높은 순서대로 유형 이름을 붙임
cluster_order = (
    data.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

CIRCLE_NAMES = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳"]

cluster_label = {
    cluster_id: CIRCLE_NAMES[i]
    for i, cluster_id in enumerate(cluster_order)
}

data["유형"] = data["군집번호"].map(cluster_label)

CLUSTER_COLORS = {
    "㉮": "#E45756",
    "㉯": "#4C78A8",
    "㉰": "#54A24B",
    "㉱": "#F2CF5B",
    "㉲": "#B279A2",
    "㉳": "#72B7B2",
}

plot_data = data.copy()
plot_data["유형"] = pd.Categorical(
    plot_data["유형"],
    categories=CIRCLE_NAMES[:n_clusters],
    ordered=True,
)

# 실루엣 점수: 군집이 얼마나 잘 구분되는지 참고
unique_labels = np.unique(data["군집번호"])

if 1 < len(unique_labels) < len(data):
    silhouette = silhouette_score(X_scaled, data["군집번호"])
else:
    silhouette = np.nan

st.success("군집 분석이 완료되었습니다.")

m1, m2, m3 = st.columns(3)
m1.metric("분류된 유형", f"{n_clusters}개")
m2.metric(
    "실루엣 점수",
    f"{silhouette:.3f}" if np.isfinite(silhouette) else "계산 불가",
)
m3.metric("분석 영화", f"{len(data):,}편")

with st.expander("실루엣 점수 해석하기"):
    st.write(
        """
        실루엣 점수는 각 영화가 자기 군집 안에서는 얼마나 비슷하고,
        다른 군집과는 얼마나 구별되는지를 나타내는 지표입니다.

        - 1에 가까울수록 군집 구분이 잘 된 편입니다.
        - 0에 가까우면 군집 사이의 경계가 겹칠 수 있습니다.
        - 음수이면 일부 영화가 다른 군집에 더 가까울 가능성이 있습니다.

        점수가 높다고 해서 영화 유형의 해석이 반드시 옳다는 뜻은 아닙니다.
        """
    )

# ==================================================
# 6. 군집 수별 실루엣 점수 비교
# ==================================================
st.divider()
st.subheader("3. 군집 수 비교")

max_k = min(6, len(data) - 1)
score_rows = []

for k in range(2, max_k + 1):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(X_scaled)

    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
        score_rows.append(
            {"군집 수": k, "실루엣 점수": score}
        )

if score_rows:
    score_df = pd.DataFrame(score_rows)

    fig_score = px.line(
        score_df,
        x="군집 수",
        y="실루엣 점수",
        markers=True,
        title="군집 수에 따른 실루엣 점수",
    )
    fig_score.update_layout(
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_score, use_container_width=True)

    best_row = score_df.loc[score_df["실루엣 점수"].idxmax()]
    st.info(
        f"현재 선택한 속성에서는 군집 수 "
        f"{int(best_row['군집 수'])}개일 때 실루엣 점수가 "
        f"가장 높았습니다. 이것은 참고 결과이며, "
        f"최종 군집 수는 해석 가능성도 함께 고려해야 합니다."
    )

# ==================================================
# 7. 2차원 산점도
# ==================================================
st.divider()
st.subheader("4. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        key="scatter_x",
    )

with col_y:
    y_default = 1 if selected_features[0] == x_feature else 0
    y_options = [f for f in selected_features if f != x_feature]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        key="scatter_y",
    )

fig2d = px.scatter(
    plot_data,
    x=x_feature,
    y=y_feature,
    color="유형",
    color_discrete_map=CLUSTER_COLORS,
    hover_name="movieNm",
    hover_data={
        "유형": True,
        "군집번호": False,
        "total_audi": ":,",
        "first_scrn": ":,",
        "days_in_top10": True,
    },
    labels={
        x_feature: x_feature,
        y_feature: y_feature,
        "유형": "영화 유형",
        "total_audi": "누적 관객",
        "first_scrn": "스크린 수",
        "days_in_top10": "10위권 일수",
    },
    title=f"{x_feature}와 {y_feature}에 따른 영화 유형",
)

fig2d.update_traces(
    marker=dict(size=8, opacity=0.75)
)
fig2d.update_layout(
    legend_title_text="영화 유형",
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig2d, use_container_width=True)

st.caption(
    "각 점은 영화 한 편을 의미합니다. 같은 색의 점은 같은 군집에 속하며, "
    "점 사이의 거리는 선택한 두 속성에서의 차이를 나타냅니다."
)

# ==================================================
# 8. 3차원 산점도
# ==================================================
st.divider()
st.subheader("5. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 그래프를 보려면 분석 속성을 세 개 이상 선택하세요.")
else:
    col3x, col3y, col3z = st.columns(3)

    with col3x:
        x3 = st.selectbox(
            "X축",
            selected_features,
            index=0,
            key="x3",
        )

    with col3y:
        y3 = st.selectbox(
            "Y축",
            [f for f in selected_features if f != x3],
            index=0,
            key="y3",
        )

    with col3z:
        z_options = [
            f for f in selected_features
            if f not in [x3, y3]
        ]
        z3 = st.selectbox(
            "Z축",
            z_options,
            index=0,
            key="z3",
        )

    fig3d = px.scatter_3d(
        plot_data,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        color_discrete_map=CLUSTER_COLORS,
        hover_name="movieNm",
        hover_data={
            "total_audi": ":,",
            "first_scrn": ":,",
            "days_in_top10": True,
        },
        title="영화 유형별 3차원 분포",
    )

    fig3d.update_traces(
        marker=dict(size=4, opacity=0.8)
    )
    fig3d.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        legend_title_text="영화 유형",
    )

    st.plotly_chart(fig3d, use_container_width=True)

# ==================================================
# 9. 유형별 요약
# ==================================================
st.divider()
st.subheader("6. 유형별 분석 결과")

summary_rows = []

for label in CIRCLE_NAMES[:n_clusters]:
    group = data[data["유형"] == label]

    if group.empty:
        continue

    summary_rows.append({
        "영화 유형": label,
        "영화 수": len(group),
        "비율(%)": len(group) / len(data) * 100,
        "평균 스크린 수": group["first_scrn"].mean(),
        "평균 누적 관객": group["total_audi"].mean(),
        "평균 10위권 일수": group["days_in_top10"].mean(),
        "평균 롱런 지수": group["롱런 지수"].mean(),
    })

summary = pd.DataFrame(summary_rows)

display_summary = summary.copy()
display_summary["비율(%)"] = display_summary["비율(%)"].round(1)
display_summary["평균 스크린 수"] = (
    display_summary["평균 스크린 수"].round(1)
)
display_summary["평균 누적 관객"] = (
    display_summary["평균 누적 관객"]
    .round(0)
    .map(lambda x: f"{x:,.0f}명")
)
display_summary["평균 10위권 일수"] = (
    display_summary["평균 10위권 일수"].round(1)
)
display_summary["평균 롱런 지수"] = (
    display_summary["평균 롱런 지수"].round(2)
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)

# 유형별 영화 수
fig_count = px.bar(
    summary,
    x="영화 유형",
    y="영화 수",
    color="영화 유형",
    color_discrete_map=CLUSTER_COLORS,
    text="영화 수",
    title="유형별 영화 수 비교",
)
fig_count.update_layout(showlegend=False)
st.plotly_chart(fig_count, use_container_width=True)

# 유형별 평균 누적 관객
fig_audience = px.bar(
    summary,
    x="영화 유형",
    y="평균 누적 관객",
    color="영화 유형",
    color_discrete_map=CLUSTER_COLORS,
    title="유형별 평균 누적 관객 비교",
)
fig_audience.update_layout(showlegend=False)
st.plotly_chart(fig_audience, use_container_width=True)

# ==================================================
# 10. 유형별 특징 자동 해석
# ==================================================
st.divider()
st.subheader("7. 영화 유형별 특징 해석")

overall_screen = data["first_scrn"].mean()
overall_audience = data["total_audi"].mean()
overall_top10 = data["days_in_top10"].mean()
overall_longrun = data["롱런 지수"].mean()

for label in CIRCLE_NAMES[:n_clusters]:
    group = data[data["유형"] == label]

    if group.empty:
        continue

    avg_screen = group["first_scrn"].mean()
    avg_audience = group["total_audi"].mean()
    avg_top10 = group["days_in_top10"].mean()
    avg_longrun = group["롱런 지수"].mean()

    if (
        avg_audience >= overall_audience
        and avg_screen >= overall_screen
    ):
        title = "대규모 흥행 경향"
    elif (
        avg_top10 >= overall_top10
        and avg_longrun >= overall_longrun
    ):
        title = "상영 지속성 경향"
    elif (
        avg_audience < overall_audience
        and avg_screen < overall_screen
    ):
        title = "상대적으로 작은 흥행 규모"
    else:
        title = "혼합 특성"

    with st.expander(
        f"{label} 유형 · {title} · {len(group):,}편",
        expanded=True,
    ):
        st.write(
            f"- 평균 스크린 수: {avg_screen:,.1f}개"
        )
        st.write(
            f"- 평균 누적 관객: {avg_audience:,.0f}명"
        )
        st.write(
            f"- 평균 10위권 일수: {avg_top10:.1f}일"
        )
        st.write(
            f"- 평균 롱런 지수: {avg_longrun:.2f}"
        )
        st.write(
            f"**해석:** 이 유형은 전체 영화 평균과 비교했을 때 "
            f"'{title}'의 경향을 보입니다. "
            "이는 선택한 데이터와 속성에 따른 상대적인 분류이며, "
            "흥행의 원인을 직접 증명하는 것은 아닙니다."
        )

# ==================================================
# 11. 유형별 관객 상위 영화
# ==================================================
st.divider()
st.subheader("8. 유형별 누적 관객 상위 영화")

for label in CIRCLE_NAMES[:n_clusters]:
    group = (
        data[data["유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    if group.empty:
        continue

    st.markdown(f"**{label} 유형의 대표 영화**")

    top_movies = group[
        ["movieNm", "first_scrn", "total_audi",
         "days_in_top10", "롱런 지수"]
    ].copy()

    top_movies.columns = [
        "영화 제목",
        "스크린 수",
        "누적 관객",
        "10위권 일수",
        "롱런 지수",
    ]

    top_movies["스크린 수"] = (
        top_movies["스크린 수"].map(lambda x: f"{x:,.0f}개")
    )
    top_movies["누적 관객"] = (
        top_movies["누적 관객"].map(lambda x: f"{x:,.0f}명")
    )
    top_movies["롱런 지수"] = (
        top_movies["롱런 지수"].round(2)
    )

    st.dataframe(
        top_movies,
        use_container_width=True,
        hide_index=True,
    )

# ==================================================
# 12. 결과 다운로드
# ==================================================
st.divider()
st.subheader("9. 분석 데이터 다운로드")

export_cols = [
    "movieNm",
    "유형",
    "first_scrn",
    "total_audi",
    "days_in_top10",
    "롱런 지수",
] + FEATURES

export_data = data[export_cols].copy()
export_data = export_data.rename(columns={
    "movieNm": "영화 제목",
    "first_scrn": "스크린 수(원래값)",
    "total_audi": "누적 관객(원래값)",
    "days_in_top10": "10위권 일수(원래값)",
})

csv_bytes = export_data.to_csv(
    index=False,
    encoding="utf-8-sig",
).encode("utf-8-sig")

st.download_button(
    label="📥 군집 분석 결과 CSV 다운로드",
    data=csv_bytes,
    file_name="영화_유형_군집분석.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "분석 방법: 선택한 속성 표준화 → K-평균 군집 분석 → "
    "누적 관객 평균에 따른 유형 이름 지정. "
    "군집 결과는 선택한 속성, 군집 수, 데이터 구성에 따라 달라질 수 있습니다."
)
