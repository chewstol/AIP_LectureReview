# Top-K Recommendation Summary

- scenario: `low_workload`
- score: `0.70 * preference_similarity + 0.30 * predicted_quality`
- quality model: Ridge regression, alpha=10.0
- face-validity (mean z-score on specified features, higher is better): +1.2966
- intra-list similarity (lower is more diverse): 0.8586
- limitation: current raw data has lecture_id only, so course/professor names must be merged later.

## Preference Weights

- `assignment_low_score`: 1.0
- `attendance_light_score`: 0.6
- `exam_light_score`: 0.9
- `grading_generous_score`: 0.9
- `teamwork_low_score`: 1.0
- `text_positive_ratio`: 0.5

## Top 10

1. lecture_id `120875` | score 0.8680 | similarity 0.8217 | predicted 4.88/5 | actual avg 4.87/5 | keywords: 연극, 꿀강, 아캠, 보면, 들으세요, 1학점, 보고, 남으면, 면담, 연극이, 꿀강입니다, 교수님이랑, 한번, 짧게, 감상문, 출석만, 번만, 영화, 한번만, 
2. lecture_id `2768543` | score 0.8638 | similarity 0.8172 | predicted 4.86/5 | actual avg 4.99/5 | keywords: 1학점, 하나, 비교과프로그램, 채우기, 들으면, pf입니다, 군학점으로, 1학점짜리, 꿀과목, 낭낭하게, 하나만, 과제, 학점, 온라인으로, 쉬운, 강의입니다,
3. lecture_id `2768545` | score 0.8622 | similarity 0.8153 | predicted 4.86/5 | actual avg 4.99/5 | keywords: 상담, 프로그램, 1학점, 진로, 비교과, 상담도, 수강하고, 신청만, 아무거나, 꿀인, 남으면, 하긴, 하라는, 귀찮긴, 개꿀, 되고, 않음, 듣기, 도움이, 
4. lecture_id `2768544` | score 0.8577 | similarity 0.8068 | predicted 4.88/5 | actual avg 4.98/5 | keywords: 1학점, 편하게, 내면, 포트폴리오, 취업, 비교과, 프로그램, 쉽게, 귀찮긴, 마지막에, 꿀강입니다, 과제도, 온라인으로, 보고서, 돼요, 있는, 대체, 좋았어
5. lecture_id `2249342` | score 0.8472 | similarity 0.7972 | predicted 4.82/5 | actual avg 4.84/5 | keywords: 월요일, 창업한, 창업에, 누구나, 2학점, 이야기를, 아침, 부담없이, 좋고, 않은거라, 채우기도, 사람들한테도, 9시라는게, 갓벽한, 스타트업, ceo가, 구
6. lecture_id `2376792` | score 0.8460 | similarity 0.7882 | predicted 4.90/5 | actual avg 4.90/5 | keywords: 박람회, 꿀강입니다, 다녀오면, 전시회, 하나만, 제출하면, 들으면, 됩니다, 영상만, 할것도, 강의도, 레포트, 한번, 좋습니다, 대체과제, 없고, 보고서, 돼
7. lecture_id `2249333` | score 0.8457 | similarity 0.7897 | predicted 4.88/5 | actual avg 4.94/5 | keywords: 채우면, 70점만, 70점, 기한, 채우기, 점수, 확인하시고, 되는, 것도, 됩니다, 미리미리, 하시면, 자동으로, 엘씨분들과, 친하다면, 놀면됨, 혼자할, 직
8. lecture_id `2892987` | score 0.8450 | similarity 0.7864 | predicted 4.91/5 | actual avg 4.48/5 | keywords: 시간도, 라틴어를, 라틴어, 인문학, 인생을, 강의실이, 불편한, 학교에서, 이야기도, 아니라, 인생에, 내용도, 와서, 그럼에도, 좋습니다, 분입니다, 대한, 
9. lecture_id `2087310` | score 0.8445 | similarity 0.7960 | predicted 4.79/5 | actual avg 4.63/5 | keywords: 영화, 한국, 영화를, 영화에, 그래도, 쉬워서, 감독, 바라보는, 시험은, 어떻게, 기말고사, 대해, 좋아하면, 영어, 얻는, 흥미롭게, 전반적인, 시험이, 지
10. lecture_id `2748934` | score 0.8439 | similarity 0.7882 | predicted 4.87/5 | actual avg 4.96/5 | keywords: 갓병섭, 10시에, 보내면, 잡으세요, 들어본, 잡을, 9시, 교양, 있으면, 주시고, 높은데는, 아프면, 잡느라, 빠질뻔, 듣고있어요, 수강신청이었습니다, 학문
