구현 전체 코드: https://github.com/yk-Jeong/iMORS/blob/main/code.ipynb `last update: 24.07.23`

- [ ]  논문 정보
- 논문명 : Real-time machine learning model to predict short-term mortality in critically ill patients: development and international validation
- 저널명 : Critical Care
- 게재일 : 2024년 3월 14일
- 원문 : https://ccforum.biomedcentral.com/articles/10.1186/s13054-024-04866-7#Sec11

### **data**

- selected feature:
    - **hosp schema - admissions table**: ‘deathtime’, 'hospital_expire_flag', ’race’, **patients table**: 'gender', 'age'
    - **9 vital signs**: 'espiratory_rate’, 'heartrate', 'sbp', 'dbp', 'temperature', ‘SpO2’, 'gcs_eye', 'gcs_verbal', 'gcs_motor’
    - **16 laboratory/chart results**: 'alt', 'ast', 'albumin', 'bun', 'bilirubin', 'crp', 'chloride', 'creatinine', 'glucose', 'hemoglobin', 'prothrombin_time', 'platelets', 'potassium', 'sodium', 'wbc'
        
        ※원 논문은 학습 코호트로 자체 병원 데이터 및 MIMIC-3을 사용 → feature의 차이가 존재(연명치료 거부자 등)
        
- dataset:
    - **stay_id** ICU 입실 고유코드(index)
    **race** 인종 → 6개 카테고리로 재분류
    **gender** 성별 
    **age**	입원 당시 연령(39세 이하 제외)
    **los** ICU 체류시간(60일 이상 제외)
    **CU type** 중환자실 종류(마지막 기준)
    **charttime** 데이터 측정 시간
    **item_id** 측정 대상값
    **value** 측정 수치
    **dead_in_hosp** 원내 사망 여부(0/1)
    - 17696 case, 424704 측정건(negative downsampled)을 train:valid:test=7:2:1 로 분리
    
    <aside>
    💡 **dataset 개선사항 (last update: 07/23)**
    
    1. 측정시점 데이터로 storetime 폐기, **charttime 채택**
    2. **결측치 처리**: stay_id별로 forward fill 적용, 첫 행이 결측치일 경우 전체의 median 값 채택
    3. **los**(length of stay): charttime(측정시각)-intime(입실시각)으로 수정
    4. negative 값 downsampling 방식 변경: random 시점 선택 후 **해당 시점 전후 24시간** 값만 채택
    5. negative random downsampling 비율을 **4배**로 고정: positive-negative 비율 6:4(valid, test set에서도 동일하게 유지)
    </aside>
    
    ⇒ 24 sequence, 128 batch size, 44 feature 구조로 모델에 입력
    

### model

```markdown
1. Feature-wise fully connected layers: 각 특성(예: 체온, 맥박, 혈압 등)에 대해 별도의 fully connected 층을 적용
2. LSTM layers: 시간 순서대로 정렬된 환자 데이터의 시간적 의존성을 캡처하기 위해 3개의 LSTM 층을 사용(이 층들은 환자의 건강 상태 변화를 시간에 따라 추적)
3. Fully connected layers with ReLU: LSTM의 마지막 출력을 받아 5개의 fully connected 층을 통과. 각 층 후에는 ReLU 활성화 함수를 적용하여 비선형성을 도입
4. output layer: 마지막 fully connected 층은 사망을 이진 분류로 예측하는 단일 출력.
```

LSTM(5 FC+3 LSTM layer) + LightGBM ensemble(soft voting: 모델간 가중치 없음)

- LSTM hyperparameter(selected value)
    
    ![Untitled](01.png)
    

- LightGBM hyperparameter(Optuna로 최적화)
    
    `{'num_leaves': 36,
    'learning_rate': 0.09888460855095497,
    'feature_fraction': 0.8874536057773371,
    'bagging_fraction': 0.6685948940175749,
    'bagging_freq': 3}`
    

### evaluation

- LSTM: epoch 20, lr=0.0001, weight decay=0.01
- 평가 지표 : Accuracy, AUROC, AUPRC
    
    
    |  | LSTM | LightGBM | ensemble  |
    | --- | --- | --- | --- |
    | Accuracy | 0.5668 | 0.5668 | 0.5668 |
    | AUROC | 0.5196 | 0.4891 | 0.4934 |
    | AUPRC | 0.4452 | 0.4356 | 0.4306 |

![Untitled](02.png)

### limitation

- 논문에서는 cohort로 MIMIC-3, eICU-crd, UMCdb 및 자체 데이터(서울대병원) 사용하였으나, 구현에는 MIMIC-4 단독으로 사용 → dataest의 차이로 일부 feature 반영할 수 없었음
- 원 연구의 AUROC 0.964(SNUH)~0.870(UMCdb)에 비해 낮은 성능

### furthermore

- another cohort dataset validation (EICU 등)