### ✅ Abstract

- **목표**: 중환자실(ICU) 입원 환자의 사망 위험을 시계열 데이터를 기반으로 예측
- **데이터**: MIMIC-IV (Vitals, Lab, Demographics 포함 다변수 시계열)
    - **특징(feature)**:
        - 9가지 vital sign (예: 호흡수, 심박수, SBP, DBP, 체온 등)
        - 16가지 lab result (예: ALT, AST, creatinine, hemoglobin, 등)
    - **레이블**: `dead_in_hosp` (원내 사망 여부)

### 🧪 Method

- **모델 구조**: LSTM (3-layer) + LightGBM soft voting 앙상블
    - Feature-wise Fully Connected Layer → LSTM (3층) → Fully Connected Layer (5층, ReLU 적용) → Output (Binary)
- **데이터 전처리**: 환자별 시계열 정렬, 결측치 보간, 주요 변수 선정
- **최적화 도구**: Optuna 기반 하이퍼파라미터 튜닝 (LightGBM)

### 🔄 Process

1. **데이터 정제**: ICU 기간 중 주요 vital sign/lab 항목 추출
2. **특성 선택 및 분석**: SHAP 시각화를 통한 feature 영향도 분석
3. **모델 구성**:
    - 시계열 LSTM 모델 (5 fully-connected + 3 LSTM layer)
    - LightGBM 단독 모델, soft voting 적용 (동일 가중치)
4. **성능 평가**:
    - Accuracy, Precision, Recall 등 비교

### 📊 **Result & Limitation**

- **최종 성능**: 앙상블 모델 Accuracy 0.85+ (Test set)
    
    
    | Model | Accuracy | AUROC | AUPRC |
    | --- | --- | --- | --- |
    | LSTM | 0.9267 | 0.9783 | 0.9776 |
    | LightGBM | 0.9450 | 0.9868 | 0.9868 |
    | Ensemble | 0.9267 | 0.9807 | 0.9804 |
- **주요 인자**: Creatinine, Age, Heart rate 등이 사망 예측에 높은 영향
- **한계**
    - 원 논문은 MIMIC-3, eICU, UMCdb 등 다양한 병원 데이터를 코호트로 사용했으나, 본 구현은 MIMIC-IV만을 train/test dataset으로 사용하여, 원 논문에서 채택한 일부 feature 불포함 (예: APTT, 연명치료 거부 변수 등)
    - 과적합의 가능성
    - 추가적인 외부 코호트(eICU 등)에서의 모델 성능 검증 필요
