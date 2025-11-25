import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import gzip
import os
import gdown
from datetime import datetime # 환자 도착 시간 기록용

# 1. 페이지 설정
st.set_page_config(
    page_title="응급실 중증도 우선순위 시스템",
    page_icon="🚨",
    layout="wide" # 화면 넓게 쓰기
)

# 2. 환자 리스트를 저장할 '기억 장소' 만들기 (세션 스테이트)
if 'patient_list' not in st.session_state:
    st.session_state.patient_list = []

# 3. 모델 로드 함수 (구글 드라이브 연동)
@st.cache_resource
def load_model():
    file_path = 'final_model.pgz'
    if not os.path.exists(file_path):
        # 본인의 구글 드라이브 파일 ID를 입력하세요 (이전과 동일)
        file_id = '1ZTVpFYYFL7QOJFjGSMcNvnXwelCKjFSj' # <-- 아까 쓰신 ID 그대로!
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_path, quiet=False)
    
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"모델 로딩 중 오류: {e}")
    st.stop()

# 4. 화면 구성 (왼쪽: 입력 / 오른쪽: 대기열)
st.title("🚨 AI 응급환자 우선순위(Triage) 대시보드")

col_input, col_queue = st.columns([1, 1.5]) # 왼쪽 1 : 오른쪽 1.5 비율

# === [왼쪽] 환자 정보 입력 ===
with col_input:
    st.subheader("📝 신규 환자 등록")
    with st.form("patient_form"):
        # 환자 구분을 위한 이름 추가
        p_name = st.text_input("환자 이름 (Name)", placeholder="예: 홍길동")
        
        c1, c2 = st.columns(2)
        with c1:
            anchor_age = st.number_input("나이", 0, 120, 50)
            temperature_y = st.number_input("체온", 30.0, 45.0, 36.5, format="%.1f")
            heartrate_y = st.number_input("심박수", 0, 300, 80)
            resprate_y = st.number_input("호흡수", 0, 100, 20)
        with c2:
            o2sat_y = st.number_input("산소포화도", 0, 100, 98)
            sbp_y = st.number_input("수축기 혈압", 1, 300, 120)
            dbp_y = st.number_input("이완기 혈압", 0, 200, 80)
            pain_y_numeric = st.slider("통증 점수", 0, 10, 0)

        submit = st.form_submit_button("환자 등록 및 분석", type="primary")

# === 로직: 점수 계산 및 리스트 추가 ===
if submit:
    if not p_name:
        st.warning("환자 이름을 입력해주세요!")
    else:
        # 1. 데이터 준비
        input_data = pd.DataFrame({
            'anchor_age': [anchor_age], 'temperature_y': [temperature_y],
            'heartrate_y': [heartrate_y], 'resprate_y': [resprate_y],
            'o2sat_y': [o2sat_y], 'sbp_y': [sbp_y],
            'dbp_y': [dbp_y], 'pain_y_numeric': [pain_y_numeric]
        })
        
        # 쇼크 인덱스 계산
        input_data['shock_index_y'] = input_data['heartrate_y'] / input_data['sbp_y']
        
        # 컬럼 순서 맞추기
        cols = ['anchor_age', 'temperature_y', 'heartrate_y', 'resprate_y',
                'o2sat_y', 'sbp_y', 'dbp_y', 'pain_y_numeric', 'shock_index_y']
        final_data = input_data[cols]

        # 2. 모델 예측
        pred_level = model.predict(final_data)[0]
        proba = model.predict_proba(final_data)[0] # 확률 배열 [P_Lv1, P_Lv2, ...]

        # 3. 🔥 응급 점수(Risk Score) 계산 (100점 만점) 🔥
        # Level 1(가장 위험)에 높은 가중치를 둬서 100점 스케일로 변환
        # 가정: 모델의 classes_가 [1, 2, 3, 4, 5] 순서라고 가정
        # 1급:100점, 2급:80점, 3급:60점, 4급:40점, 5급:20점 가중치 부여
        
        # 클래스별 가중치 (위험할수록 고득점)
        weights = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
        
        risk_score = 0
        for idx, level_class in enumerate(model.classes_):
            # level_class가 1.0, 2.0 실수형일 수 있으므로 int로 변환해서 매칭
            lvl = int(level_class)
            if lvl in weights:
                risk_score += proba[idx] * weights[lvl]
        
        # 4. 리스트에 추가
        new_patient = {
            "이름": p_name,
            "도착시간": datetime.now().strftime("%H:%M:%S"),
            "예측단계": f"Level {int(pred_level)}",
            "응급점수": round(risk_score, 1), # 소수점 1자리
            "나이": anchor_age,
            "주증상": f"통증 {pain_y_numeric}, 열 {temperature_y}"
        }
        st.session_state.patient_list.append(new_patient)
        st.success(f"✅ {p_name} 환자 등록 완료! (응급 점수: {risk_score:.1f}점)")

# === [오른쪽] 실시간 대기열 (점수순 정렬) ===
with col_queue:
    st.subheader("📋 실시간 응급 환자 대기열")
    
    if len(st.session_state.patient_list) > 0:
        # 🔥 핵심: 응급 점수가 높은 순서대로 정렬 (내림차순)
        sorted_list = sorted(st.session_state.patient_list, key=lambda x: x['응급점수'], reverse=True)
        
        # 데이터프레임으로 변환하여 보여주기
        df_display = pd.DataFrame(sorted_list)
        
        # 가장 급한 환자 강조 표시 (1등)
        top_patient = sorted_list[0]
        st.error(f"🚨 **치료 1순위:** {top_patient['이름']} (Level {top_patient['예측단계'][-1]} / {top_patient['응급점수']}점)")
        
        # 테이블 스타일링 (점수가 높으면 배경색 진하게)
        st.dataframe(
            df_display,
            column_config={
                "응급점수": st.column_config.ProgressColumn(
                    "응급도 (100점 만점)",
                    help="점수가 높을수록 위급합니다.",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 리스트 초기화 버튼
        if st.button("대기열 초기화"):
            st.session_state.patient_list = []
            st.rerun()
            
    else:
        st.info("현재 대기 중인 환자가 없습니다.")

# 디버깅용 (필요 없으면 삭제)
# st.write(model.classes_)