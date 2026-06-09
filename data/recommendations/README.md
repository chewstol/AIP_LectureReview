# Top-K Lecture Recommendation

이 폴더는 전체 데이터에서 만든 강의 feature 벡터를 바탕으로, 사용자 선호 벡터에 가장 가까운 강의를 Top-K로 정렬한 결과입니다.

## Recommendation Logic

추천 점수는 아래 두 값을 섞어 계산합니다.

```text
recommendation_score = 0.7 * preference_similarity + 0.3 * predicted_quality
```

- `preference_similarity`: 사용자 선호 벡터와 강의 feature 벡터의 cosine similarity를 0~1 범위로 변환한 값
- `predicted_quality`: 전체 데이터로 학습한 Ridge regression(alpha=10)이 예측한 강의 평점 품질
- `rating_average`: 실제 수집 데이터의 평균 별점으로, 추천 점수 계산의 직접 입력이 아니라 결과 해석용으로 함께 출력

## Presets

- `low_workload`: 과제 적음, 팀플 적음, 학점 후함, 시험 부담 낮음
- `learning_quality`: 강의력/설명 관련 긍정 텍스트와 긍정 리뷰 비율 중시
- `exam_light`: 시험 부담 낮은 강의 선호
- `no_team_project`: 팀플 부담이 낮고 전체 평가도 괜찮은 강의 선호
- `challenging_but_good`: 과제/시험 부담은 있어도 강의력이 좋은 강의 선호

## Generated Files

- `topk_low_workload_10.csv`
- `topk_learning_quality_10.csv`
- `topk_exam_light_10.csv`
- `topk_no_team_project_10.csv`
- `topk_challenging_but_good_10.csv`

각 시나리오별 `_all.csv`는 753개 강의 전체 순위이고, `_summary.md`는 발표/공유용 요약입니다.

## How To Run

```powershell
& 'C:\Users\jungseobum\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\recommend_topk.py --preset low_workload --top-k 10
```

사용자 선호를 직접 넣고 싶으면 JSON 형태로 feature 가중치를 줄 수 있습니다.

```powershell
& 'C:\Users\jungseobum\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\recommend_topk.py --weights-json '{ "assignment_low_score": 1.0, "exam_light_score": 0.8, "text_positive_ratio": 0.6 }'
```

## Limitation

현재 raw 데이터에는 과목명/교수명 메타데이터가 거의 없어서 추천 결과가 `lecture_id` 중심으로 나옵니다. 실제 서비스형 추천으로 보이게 하려면 `lecture_id -> 과목명/교수명/학과` 매핑 테이블을 추가로 붙이는 단계가 필요합니다.
