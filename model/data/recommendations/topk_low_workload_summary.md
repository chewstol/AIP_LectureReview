# Top-K Recommendation Summary

- scenario: `low_workload`
- score: `0.70 * preference_similarity + 0.30 * predicted_quality`
- quality model: Ridge regression, alpha=10.0
- limitation: current raw data has lecture_id only, so course/professor names must be merged later.

## Preference Weights

- `assignment_low_score`: 1.0
- `attendance_light_score`: 0.6
- `exam_light_score`: 0.9
- `grading_generous_score`: 0.9
- `teamwork_low_score`: 1.0
- `text_positive_ratio`: 0.5

## Top 10

1. lecture_id `120875` | score 0.8752 | similarity 0.8320 | predicted 4.88/5 | actual avg 4.87/5 | keywords: 연극, 꿀강, 아캠, 보면, 들으세요, 1학점, 보고, 남으면, 면담, 연극이, 꿀강입니다, 교수님이랑, 한번, 짧게, 감상문, 출석만, 번만, 영화, 한번만, 
2. lecture_id `2768543` | score 0.8719 | similarity 0.8288 | predicted 4.86/5 | actual avg 4.99/5 | keywords: 1학점, 하나, 비교과프로그램, 채우기, 들으면, pf입니다, 군학점으로, 1학점짜리, 꿀과목, 낭낭하게, 하나만, 과제, 학점, 온라인으로, 쉬운, 강의입니다,
3. lecture_id `2768545` | score 0.8708 | similarity 0.8276 | predicted 4.86/5 | actual avg 4.99/5 | keywords: 상담, 프로그램, 1학점, 진로, 비교과, 상담도, 수강하고, 신청만, 아무거나, 꿀인, 남으면, 하긴, 하라는, 귀찮긴, 개꿀, 되고, 않음, 듣기, 도움이, 
4. lecture_id `2768544` | score 0.8688 | similarity 0.8226 | predicted 4.88/5 | actual avg 4.98/5 | keywords: 1학점, 편하게, 내면, 포트폴리오, 취업, 비교과, 프로그램, 쉽게, 귀찮긴, 마지막에, 꿀강입니다, 과제도, 온라인으로, 보고서, 돼요, 있는, 대체, 좋았어
5. lecture_id `2249333` | score 0.8607 | similarity 0.8111 | predicted 4.88/5 | actual avg 4.94/5 | keywords: 채우면, 70점만, 70점, 기한, 채우기, 점수, 확인하시고, 되는, 것도, 됩니다, 미리미리, 하시면, 자동으로, 엘씨분들과, 친하다면, 놀면됨, 혼자할, 직
6. lecture_id `991009` | score 0.8581 | similarity 0.8100 | predicted 4.85/5 | actual avg 4.89/5 | keywords: 어깨동무, 채울, 엘씨원들이랑, 1학점, 먹을, 됩니다, 영상만, pass, 날먹, 패스, 열심히, 신경, 영상, 같이, 하고싶어서, 하는걸로, 3일만에, 채움,
7. lecture_id `2249342` | score 0.8530 | similarity 0.8054 | predicted 4.82/5 | actual avg 4.84/5 | keywords: 월요일, 창업한, 창업에, 누구나, 2학점, 이야기를, 아침, 부담없이, 좋고, 않은거라, 채우기도, 사람들한테도, 9시라는게, 갓벽한, 스타트업, ceo가, 구
8. lecture_id `2086919` | score 0.8522 | similarity 0.8031 | predicted 4.83/5 | actual avg 4.80/5 | keywords: 대체, 갓조일, 중간, 발표, 대체과제, 분량이, 아방가르드, 안잡으심, 제출이고, 기말, 시험은, 10페이지, 중간기말, 매년, 예술에, 뭐라, 같아요, 작성하
9. lecture_id `2376792` | score 0.8516 | similarity 0.7962 | predicted 4.90/5 | actual avg 4.90/5 | keywords: 박람회, 꿀강입니다, 다녀오면, 전시회, 하나만, 제출하면, 들으면, 됩니다, 영상만, 할것도, 강의도, 레포트, 한번, 좋습니다, 대체과제, 없고, 보고서, 돼
10. lecture_id `2748934` | score 0.8513 | similarity 0.7988 | predicted 4.87/5 | actual avg 4.96/5 | keywords: 갓병섭, 10시에, 보내면, 잡으세요, 들어본, 잡을, 9시, 교양, 있으면, 주시고, 높은데는, 아프면, 잡느라, 빠질뻔, 듣고있어요, 수강신청이었습니다, 학문
