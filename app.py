import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import gzip
import gdown
import os

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="응급실 중증도 예측 시스템",
    page_icon="🏥",
    layout="centered"
)

# 2. 모델 로드 함수
@st.cache_resource
def load_model():
    file_path = 'final_model.pgz'
    
    # 파일이 없으면 구글 드라이브에서 다운로드
    if not os.path.exists(file_path):
        # ⚠️ 여기에 아까 복사한 본인의 구글 드라이브 파일 ID를 넣으세요!
        file_id = '1ZTVpFYYFL7QOJFjGSMcNvnXwelCKjFSj' 
        
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_path, quiet=False)

    # 모델 로드
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 헤더 및 설명
st.title("🏥 응급실 중증도 예측 AI")
st.markdown("환자의 **8가지 활력 징후**를 입력하면 AI가 중증도(Triage Level)를 분석합니다.")
st.markdown("---")

# 4. 입력 폼 구성 (사용자 입력)
with st.form("patient_form"):
    st.subheader("📋 환자 정보 입력")
    col1, col2 = st.columns(2)

    with col1:
        anchor_age = st.number_input("1. 나이 (Age)", min_value=0, max_value=120, value=50)
        temperature_y = st.number_input("2. 체온 (Temp)", min_value=30.0, max_value=45.0, value=36.5, format="%.1f")
        heartrate_y = st.number_input("3. 심박수 (Heart Rate)", min_value=0, max_value=300, value=80)
        resprate_y = st.number_input("4. 호흡수 (Resp Rate)", min_value=0, max_value=100, value=20)

    with col2:
        o2sat_y = st.number_input("5. 산소포화도 (O2 Sat)", min_value=0, max_value=100, value=98)
        sbp_y = st.number_input("6. 수축기 혈압 (SBP)", min_value=1, max_value=300, value=120, help="0이 될 수 없습니다.")
        dbp_y = st.number_input("7. 이완기 혈압 (DBP)", min_value=0, max_value=200, value=80)
        pain_y_numeric = st.slider("8. 통증 점수 (Pain 0-10)", 0, 10, 0)

    submit = st.form_submit_button("🚀 중증도 예측하기", type="primary")

# 5. 예측 및 결과 출력 로직
if submit:
    # (1) 데이터프레임 생성
    input_dict = {
        'anchor_age': [anchor_age],
        'temperature_y': [temperature_y],
        'heartrate_y': [heartrate_y],
        'resprate_y': [resprate_y],
        'o2sat_y': [o2sat_y],
        'sbp_y': [sbp_y],
        'dbp_y': [dbp_y],
        'pain_y_numeric': [pain_y_numeric]
    }
    df_new = pd.DataFrame(input_dict)

    # (2) 파생변수 생성 (Shock Index) - 로직 반영
    if sbp_y == 0:
        df_new['shock_index_y'] = 0 
    else:
        df_new['shock_index_y'] = df_new['heartrate_y'] / df_new['sbp_y']
    
    # Inf 처리
    df_new['shock_index_y'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # (3) 컬럼 순서 정렬 (모델 학습시와 동일하게)
    predictor_cols = [
        'anchor_age', 'temperature_y', 'heartrate_y', 'resprate_y',
        'o2sat_y', 'sbp_y', 'dbp_y', 'pain_y_numeric', 'shock_index_y'
    ]
    df_final = df_new[predictor_cols]

    # 디버깅용 데이터 확인 (필요시 주석 해제)
    # st.write("입력 데이터:", df_final)

    try:
        # 예측 수행
        prediction = model.predict(df_final)
        proba = model.predict_proba(df_final)

        st.markdown("---")
        st.subheader("📢 분석 결과")

        # 결과 텍스트 출력
        pred_level = prediction[0]
        st.success(f"**최종 예측 중증도: Level {pred_level}**")
        st.info(f"계산된 쇼크 인덱스: {df_final['shock_index_y'][0]:.2f}")

        # (4) 시각화: Plotly 가로형 막대 차트 (의료 모니터 스타일)
        st.write("### 📊 레벨별 확률 분석")
        
        levels = model.classes_
        probabilities = proba[0]
        
        # 색상 팔레트 (위험=빨강 ~ 안전=파랑/초록)
        # 클래스 개수에 맞춰서 색상을 자동으로 가져옵니다.
        color_palette = ['#FF4B4B', '#FF8C00', '#FFD700', '#90EE90', '#1E90FF'] # Red, Orange, Yellow, Green, Blue
        
        fig = go.Figure(go.Bar(
            x=probabilities,
            y=[f"Level {l}" for l in levels],
            orientation='h',
            marker=dict(color=color_palette[:len(levels)]), # 클래스 개수만큼 색상 사용
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='auto',
            hoverinfo='text+y'
        ))

        fig.update_layout(
            xaxis_title="확률 (Probability)",
            yaxis_title="중증도 단계",
            plot_bgcolor='rgba(0,0,0,0)', # 배경 투명하게
            height=350,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"예측 중 에러 발생: {e}")
        st.warning("입력 데이터의 형태가 모델 학습 데이터와 일치하지 않을 수 있습니다.")