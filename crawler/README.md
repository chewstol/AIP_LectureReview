# Lecture Review Data Pipeline

`crawler/`는 강의평가 데이터를 수집하고, 모델 실험에 사용할 수 있는 정규화 CSV와 기본 feature 데이터를 만드는 파이프라인을 담고 있습니다.

이 폴더의 핵심 역할은 원본 API 응답을 그대로 분석 코드에 넣는 것이 아니라, 이후 `model/`, `persona_maker/`, `evaluation/`에서 재사용할 수 있는 구조화된 데이터로 바꾸는 것입니다.

## 파일 구조

```text
crawler/
├── README.md
├── crawler_README.md
├── requirements.txt
├── .gitignore
├── crawler_src/
│   ├── api_client.py
│   ├── collect.py
│   └── config.py
├── scripts/
│   ├── normalize_raw.py
│   ├── build_lecture_nodes.py
│   ├── build_text_features.py
│   ├── run_small_portion_experiment.py
│   ├── run_full_cv_experiment.py
│   ├── recommend_topk.py
│   └── make_cv_visuals.py
└── data/
    ├── normalized/
    ├── model/
    ├── experiments/
    └── recommendations/
```

## 주요 구성

- `crawler_src/`: 강의 상세 정보와 강의평가 글 목록을 API로 수집하는 코드입니다.
- `scripts/normalize_raw.py`: 원본 JSON을 CSV 형태로 정규화합니다.
- `scripts/build_lecture_nodes.py`: 강의 단위 node/feature 데이터를 만듭니다.
- `scripts/build_text_features.py`: 강의평가 텍스트 기반 feature를 생성합니다.
- `scripts/recommend_topk.py`: feature vector 기반 Top-K 추천을 실행합니다.
- `data/normalized/`: 정규화된 강의, 강의평가, 시험, 교재 CSV입니다.
- `data/model/`: 모델 입력으로 쓰는 강의 feature CSV입니다.
- `data/experiments/`: 기본 실험 결과와 시각화 파일입니다.
- `data/recommendations/`: 추천 결과 CSV와 설명 파일입니다.

API 수집 방식 자체에 대한 자세한 설명은 `crawler_README.md`를 참고합니다.

## 실행 방법

`crawler/` 폴더에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

원본 JSON을 정규화하고 feature를 만드는 기본 흐름은 다음과 같습니다.

```powershell
python scripts\normalize_raw.py
python scripts\build_lecture_nodes.py
python scripts\build_text_features.py
```

기본 모델 실험과 추천 결과를 다시 만들려면 다음 명령을 실행합니다.

```powershell
python scripts\run_full_cv_experiment.py
python scripts\make_cv_visuals.py
python scripts\recommend_topk.py --preset low_workload --top-k 10
```

## 산출물

- `data/normalized/lecture_articles.csv`: 강의평가 글 단위 정규화 데이터입니다.
- `data/normalized/lecture_details.csv`: 강의 상세 정보 정규화 데이터입니다.
- `data/model/lecture_nodes.csv`: 강의 단위 기본 feature 데이터입니다.
- `data/model/lecture_nodes_with_text.csv`: 텍스트 feature가 결합된 모델 입력 데이터입니다.
- `data/experiments/full_cv/`: cross-validation 결과와 발표용 그래프입니다.
- `data/recommendations/`: 선호 preset별 추천 결과입니다.

## 포함하지 않는 것

원본 raw JSON 전체는 공개/공유용 산출물에 포함하지 않는 것을 전제로 합니다. 원본 데이터는 용량이 크고, 서비스 약관 및 민감 정보 이슈가 있을 수 있으므로 필요한 경우 별도 경로로 관리합니다.

## 주의사항

API 수집에는 인증 쿠키가 필요할 수 있습니다. 쿠키나 토큰을 코드에 직접 저장하지 말고 환경 변수로 전달해야 합니다.

현재 데이터에는 과목명, 교수명, 학생별 수강 이력 같은 metadata가 제한적입니다. 따라서 이 폴더의 산출물은 모델 학습과 실험 재현을 위한 입력 데이터로 보는 것이 적절합니다.
