# Top-K Recommendation Summary

- scenario: `exam_light`
- score: `0.70 * preference_similarity + 0.30 * predicted_quality`
- quality model: Ridge regression, alpha=10.0
- limitation: current raw data has lecture_id only, so course/professor names must be merged later.

## Preference Weights

- `assignment_low_score`: 0.6
- `exam_light_score`: 1.0
- `grading_generous_score`: 0.7
- `teamwork_low_score`: 0.5
- `text_exam_positive_tfidf`: 0.4

## Top 10

1. lecture_id `2748951` | score 0.8589 | similarity 0.8153 | predicted 4.80/5 | actual avg 4.83/5 | keywords: 힐링되는, 2학점, 인생, 모두, 없고, 중간, 없어요, 기말, 배우는, 대체과제외에, pf에, pf인데, 마음건강을, 다입니다, f이기도, 명상, 째거나, 남으
2. lecture_id `1772126` | score 0.8387 | similarity 0.7853 | predicted 4.82/5 | actual avg 4.78/5 | keywords: 리트, 1학점, 대체과제, 꿀강입니당, 중간, 기말, 대체, 학점, 번만, 편해요, 됩니다, 되는데, 없고, 좋은, 무난하게, 편하게, 없이, 준비하기에, 산출됩
3. lecture_id `2086919` | score 0.8261 | similarity 0.7659 | predicted 4.83/5 | actual avg 4.80/5 | keywords: 대체, 갓조일, 중간, 발표, 대체과제, 분량이, 아방가르드, 안잡으심, 제출이고, 기말, 시험은, 10페이지, 중간기말, 매년, 예술에, 뭐라, 같아요, 작성하
4. lecture_id `2249342` | score 0.8220 | similarity 0.7612 | predicted 4.82/5 | actual avg 4.84/5 | keywords: 월요일, 창업한, 창업에, 누구나, 2학점, 이야기를, 아침, 부담없이, 좋고, 않은거라, 채우기도, 사람들한테도, 9시라는게, 갓벽한, 스타트업, ceo가, 구
5. lecture_id `1772127` | score 0.8133 | similarity 0.7456 | predicted 4.86/5 | actual avg 4.67/5 | keywords: 1학점, 남는다면, 꿀교양, 둘다, 편한, 과제도, 꿀강, 많아요, 없어요, 좋은, 없이, 다른수업, 죄송스러운, 가져가게, 출석하기가, 차마시고, 오면됨, 1핫
6. lecture_id `2748934` | score 0.8098 | similarity 0.7396 | predicted 4.87/5 | actual avg 4.96/5 | keywords: 갓병섭, 10시에, 보내면, 잡으세요, 들어본, 잡을, 9시, 교양, 있으면, 주시고, 높은데는, 아프면, 잡느라, 빠질뻔, 듣고있어요, 수강신청이었습니다, 학문
7. lecture_id `1325151` | score 0.8063 | similarity 0.7520 | predicted 4.67/5 | actual avg 4.82/5 | keywords: 무조건, 송인욱교수님, 파생, 투자론은, 채권, 절대평가에, 투자론, 문제, 다만, 가끔, 위주로, 도움되는, 없이, 들으세요, 족보, 퀴즈, 학점도, 학생들, 
8. lecture_id `2249337` | score 0.7990 | similarity 0.7471 | predicted 4.60/5 | actual avg 4.69/5 | keywords: 독후감, 수학을, 오픈북, 시험입니다, 조선, 교양, 좋은, 아직, 흥미있게, 들었습니다, 중간은, 수학, 강의를, 기말은, 채우기, 들을, 오프라인으로, 내용,
9. lecture_id `120875` | score 0.7981 | similarity 0.7218 | predicted 4.88/5 | actual avg 4.87/5 | keywords: 연극, 꿀강, 아캠, 보면, 들으세요, 1학점, 보고, 남으면, 면담, 연극이, 꿀강입니다, 교수님이랑, 한번, 짧게, 감상문, 출석만, 번만, 영화, 한번만, 
10. lecture_id `2376792` | score 0.7970 | similarity 0.7182 | predicted 4.90/5 | actual avg 4.90/5 | keywords: 박람회, 꿀강입니다, 다녀오면, 전시회, 하나만, 제출하면, 들으면, 됩니다, 영상만, 할것도, 강의도, 레포트, 한번, 좋습니다, 대체과제, 없고, 보고서, 돼
