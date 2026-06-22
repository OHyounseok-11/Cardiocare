# CardioCare (기계학습)

UCI Heart Disease 데이터셋을 이용한 심장병 예측 ML 시스템입니다.  
CardioCare는 단순 의사결정을 보조하는 도구이며, 완벽한 진단을 내려주는 모델이 아닙니다. 

## 재현 절차

```bash
# 1. 클론 
git clone <https://github.com/OHyounseok-11/CardioCare-.git>
cd CardioCare
pip install -r requirements.txt

# 2. EDA 
jupyter notebook notebooks/01_eda_preprocessing.ipynb

# 3. 모델 학습 + MLflow 기록
python src/train.py

# 4. 단위 테스트
python -m unittest discover -s tests -v

# 5. 배치 추론
python src/inference.py --input data/sample_batch.csv --output outputs/predictions.csv

# 6. Docker 빌드 및 실행
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0

# 7. 드리프트 모니터링
python src/monitor.py
```

## MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

브라우저에서 3개 이상 모델 계열(logistic_regression, svc, random_forest)과 튜닝 run을 확인할 수 있습니다.

## 피처 스토어 / 모델 레지스트리 (논리 설계)

| 항목 | 후보 | 이유 |
|------|------|------|
| **피처 스토어** | `chol` (콜레스테롤) | 드리프트에 민감하고 생활습관·식이 변화를 반영해 재학습 트리거로 적합 |
| **레지스트리 메타데이터** | `training_data_hash` + `class_balance` | 데이터 버전 추적 및 클래스 불균형 변화 감지에 필수 |

## AI 도구 사용 공개

본 프로젝트 작성 시 AI 도구(Cursor/Claude)를 **보일러플레이트 코드 생성 및 디버깅**에만 사용했습니다.  
모든 실험 결과·모델 선택·보고서 해석은 작성자가 검증했습니다.

## 제출물

- `report.pdf` — 6~10쪽 최종 보고서 (별도 작성)
- GitHub public 저장소 — CI green 상태 유지

## 라이선스 / 인용

UCI Heart Disease 데이터 사용 시 원 논문 및 기관 기여자를 인용해야 합니다. 자세한 내용은 `data/heart-disease.names`를 참고하세요.
