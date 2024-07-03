# iMORS
 Real-time machine learning model to predict short-term mortality in critically ill patients: development and international validation(2024)
- 논문명 : Real-time machine learning model to predict short-term mortality in critically ill patients: development and international validation
- 저널명 : Critical Care
- 게재일 : 2024년 3월 14일
- 원문 : https://ccforum.biomedcentral.com/articles/10.1186/s13054-024-04866-7#Sec11

### **data**

- selected feature:
    - **hosp schema - admissions table**: ‘hadm_id’, ‘deathtime’, 'hospital_expire_flag', ’race’, **patients table**: 'gender', 'age'
    - **9 vital signs**: 'espiratory_rate’, 'heartrate', 'sbp', 'dbp', 'temperature', ‘SpO2’, 'gcs_eye', 'gcs_verbal', 'gcs_motor’
    - **16 laboratory/chart results**: 'alt', 'ast', 'albumin', 'bun', 'bilirubin', 'crp', 'chloride', 'creatinine', 'glucose', 'hemoglobin', 'prothrombin_time', 'platelets', 'potassium', 'sodium', 'wbc'
        
        ※원 논문은 학습 코호트로 자체 병원 데이터 및 MIMIC-3을 사용 → feature의 차이가 존재
        
- dataset
    - **death or disch time** 사망/퇴원 시각 (index)
    **race** 인종 → 6개 카테고리로 재분류
    **gender** 성별 
    **age**	입원 당시 연령(18~39세 제외)
    **los** ICU 체류시간(60일 이상 제외)
    **CU type** 중환자실 종류(마지막 기준)
    **storetime** 데이터 측정 시간
    **item_id** 측정값(약어)
    **value** 측정 수치
    **dead_in_hosp** 원내 사망 여부(0/1)
    - 결측치 처리: forward fill → 최초로 등장하는 행이 결측치인 경우 median fill 순서 (논문)
    최초로 등장하는 행이 결측치인 경우 median fill → forward fill (실제 구현)
    - 52783 case, 858263 측정건을 train:valid:test=7:2:1 로 분리(unbalanced sample: 1/10로 random downsampling하여 dead 18.4% alive 81.6% 비율 유지)

### model

```markdown
1. **Feature-wise fully connected layers**: 각 특성(예: 체온, 맥박, 혈압 등)에 대해 별도의 fully connected 층을 적용
2. **LSTM layers**: 시간 순서대로 정렬된 환자 데이터의 시간적 의존성을 캡처하기 위해 3개의 LSTM 층을 사용(이 층들은 환자의 건강 상태 변화를 시간에 따라 추적)
3. **Fully connected layers with ReLU**: LSTM의 마지막 출력을 받아 5개의 fully connected 층을 통과. 각 층 후에는 ReLU 활성화 함수를 적용하여 비선형성을 도입
4. **output layer**: 마지막 fully connected 층은 사망을 이진 분류로 예측하는 단일 출력.
```

LSTM(5 FC+3 LSTM layer) + LightGBM ensemble(soft voting: 모델간 가중치 없음)

- LSTM hyperparameter
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ca6c517-4a39-4006-964b-1cdb47c9cea6/002931b6-d93a-4540-addb-f51e052e1d75/Untitled.png)
    
- LightGBM hyperparameter(Optuna로 최적화)
    
    `'num_leaves': 50, 'learning_rate': 0.08174132149788077, 'feature_fraction': 0.6526524514956806`
    

### evaluation

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9ca6c517-4a39-4006-964b-1cdb47c9cea6/27f71f28-8c1c-4714-b3d2-9d94f5808bce/Untitled.png)

- 평가 지표 : Accuracy, AUROC, AUPRC
    
    
    |  | LSTM | LightGBM | ensemble  |
    | --- | --- | --- | --- |
    | Accuracy | 0.8652 | 0.8937 | 0.8848 |
    | AUROC | 0.8324 | 0.8908 | 0.8798 |
    | AUPRC | 0.5747 | 0.7081 | 0.6771 |

### futhermore

- another cohort dataset (EICU 등)
