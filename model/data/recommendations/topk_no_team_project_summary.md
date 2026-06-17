# Top-K Recommendation Summary

- scenario: `no_team_project`
- score: `0.70 * preference_similarity + 0.30 * predicted_quality`
- quality model: Ridge regression, alpha=10.0
- limitation: current raw data has lecture_id only, so course/professor names must be merged later.

## Preference Weights

- `assignment_low_score`: 0.7
- `exam_light_score`: 0.5
- `grading_generous_score`: 0.5
- `teamwork_low_score`: 1.0
- `text_positive_ratio`: 0.5

## Top 10

1. lecture_id `120875` | score 0.8535 | similarity 0.8010 | predicted 4.88/5 | actual avg 4.87/5 | keywords: 연극, 꿀강, 아캠, 보면, 들으세요, 1학점, 보고, 남으면, 면담, 연극이, 꿀강입니다, 교수님이랑, 한번, 짧게, 감상문, 출석만, 번만, 영화, 한번만, 
2. lecture_id `2748934` | score 0.8482 | similarity 0.7944 | predicted 4.87/5 | actual avg 4.96/5 | keywords: 갓병섭, 10시에, 보내면, 잡으세요, 들어본, 잡을, 9시, 교양, 있으면, 주시고, 높은데는, 아프면, 잡느라, 빠질뻔, 듣고있어요, 수강신청이었습니다, 학문
3. lecture_id `2249342` | score 0.8465 | similarity 0.7962 | predicted 4.82/5 | actual avg 4.84/5 | keywords: 월요일, 창업한, 창업에, 누구나, 2학점, 이야기를, 아침, 부담없이, 좋고, 않은거라, 채우기도, 사람들한테도, 9시라는게, 갓벽한, 스타트업, ceo가, 구
4. lecture_id `1772127` | score 0.8437 | similarity 0.7889 | predicted 4.86/5 | actual avg 4.67/5 | keywords: 1학점, 남는다면, 꿀교양, 둘다, 편한, 과제도, 꿀강, 많아요, 없어요, 좋은, 없이, 다른수업, 죄송스러운, 가져가게, 출석하기가, 차마시고, 오면됨, 1핫
5. lecture_id `120992` | score 0.8374 | similarity 0.7817 | predicted 4.84/5 | actual avg 4.93/5 | keywords: 끝내주시고, 정병섭, 짧고, 일찍, 무조건, 들으세요, 중간1, 편의를, 굵게, 좋았음, 성논은, 드리면, 시험도, 항상, 없고, 최고의, 하나도, 편안한, 잡으
6. lecture_id `2768543` | score 0.8339 | similarity 0.7745 | predicted 4.86/5 | actual avg 4.99/5 | keywords: 1학점, 하나, 비교과프로그램, 채우기, 들으면, pf입니다, 군학점으로, 1학점짜리, 꿀과목, 낭낭하게, 하나만, 과제, 학점, 온라인으로, 쉬운, 강의입니다,
7. lecture_id `2768545` | score 0.8330 | similarity 0.7737 | predicted 4.86/5 | actual avg 4.99/5 | keywords: 상담, 프로그램, 1학점, 진로, 비교과, 상담도, 수강하고, 신청만, 아무거나, 꿀인, 남으면, 하긴, 하라는, 귀찮긴, 개꿀, 되고, 않음, 듣기, 도움이, 
8. lecture_id `2087310` | score 0.8304 | similarity 0.7760 | predicted 4.79/5 | actual avg 4.63/5 | keywords: 영화, 한국, 영화를, 영화에, 그래도, 쉬워서, 감독, 바라보는, 시험은, 어떻게, 기말고사, 대해, 좋아하면, 영어, 얻는, 흥미롭게, 전반적인, 시험이, 지
9. lecture_id `2376792` | score 0.8273 | similarity 0.7616 | predicted 4.90/5 | actual avg 4.90/5 | keywords: 박람회, 꿀강입니다, 다녀오면, 전시회, 하나만, 제출하면, 들으면, 됩니다, 영상만, 할것도, 강의도, 레포트, 한번, 좋습니다, 대체과제, 없고, 보고서, 돼
10. lecture_id `2768544` | score 0.8271 | similarity 0.7631 | predicted 4.88/5 | actual avg 4.98/5 | keywords: 1학점, 편하게, 내면, 포트폴리오, 취업, 비교과, 프로그램, 쉽게, 귀찮긴, 마지막에, 꿀강입니다, 과제도, 온라인으로, 보고서, 돼요, 있는, 대체, 좋았어
