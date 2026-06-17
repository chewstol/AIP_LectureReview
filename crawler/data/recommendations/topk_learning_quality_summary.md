# Top-K Recommendation Summary

- scenario: `learning_quality`
- score: `0.70 * preference_similarity + 0.30 * predicted_quality`
- quality model: Ridge regression, alpha=10.0
- limitation: current raw data has lecture_id only, so course/professor names must be merged later.

## Preference Weights

- `grading_generous_score`: 0.3
- `text_positive_ratio`: 0.8
- `text_teaching_positive_tfidf`: 1.0
- `text_teaching_tfidf`: 1.0

## Top 10

1. lecture_id `44167` | score 0.9456 | similarity 0.9514 | predicted 4.66/5 | actual avg 4.56/5 | keywords: 화학, 출석이, 들어도, 과제는, 잘해주십니다, 정말정말, 굉장히, 강의력, 오프라인으로, 맞춰서, 최고, 시험도, 좋았음, 어떤, 이해, 오프라인, 있는, 설명
2. lecture_id `2748975` | score 0.9365 | similarity 0.9414 | predicted 4.63/5 | actual avg 4.43/5 | keywords: 합의하신, 칠판, 복습이, 절대평가, 설명, 배점이, 봤고, 학생들, 복습, 기준, 친절하게, 복수전공자인, 조정해주십니다, 싶어져요, 일주일씩, 예제에서만, 출
3. lecture_id `991606` | score 0.9097 | similarity 0.9174 | predicted 4.46/5 | actual avg 4.54/5 | keywords: 기말, 피피티, ppt, 교안, 중간, 교안이, 좋은, 퀄이, 문제, 아쉬웠어요, 공부하기에, 상위, 깔끔한, 끝날, 시험은, 잘주심, 시험문제는, 기출, 좋아서
4. lecture_id `353911` | score 0.9048 | similarity 0.9125 | predicted 4.43/5 | actual avg 4.50/5 | keywords: 강의력, 강의를, 강의력도, 늦게, 좋고, 풀어보고, 공지도, 교안, 올려주시는, 시험, 없고, 한국어로, 온라인으로, 중요한, 없이, 올라오는거, 5주치, 올려
5. lecture_id `226710` | score 0.9035 | similarity 0.8837 | predicted 4.75/5 | actual avg 4.86/5 | keywords: 중간은, 족보를, 타며, 강의력, 영어로, 최고십니다, 같아요, 족보, 기말은, 정도, 강의력이, 시험은, 좋으시고, 엄청, 들으시면, 족보랑, 설명, 이해하기,
6. lecture_id `993442` | score 0.8998 | similarity 0.8759 | predicted 4.78/5 | actual avg 4.85/5 | keywords: 교안에, 공식이, 황금철, 기반한, 필요없는, 의미를, 아니라, 좋으십니다, 공식, 단순, 내용으로, 중요하게, 가르쳐, 좋습니다, 강의력, 달달, 개념을, 과목
7. lecture_id `488935` | score 0.8915 | similarity 0.8829 | predicted 4.56/5 | actual avg 4.68/5 | keywords: 강의력, goat, 과제가, xv6는, 강의력이, 재밌게, 한데, 어렵긴, 관심, 좋으셔서, 재밌고, 시험공부하지, 맞지않더라도, 토크쇼, 증원이나, 되었으면, 
8. lecture_id `78169` | score 0.8879 | similarity 0.8775 | predicted 4.56/5 | actual avg 4.64/5 | keywords: 출튀, 화학을, 잡으심, 다시, 어려운, 한번도, 교수님이, 부분은, 한국어로, 받음, 절대, 이해하기, 하시고, 열심히, 충분히, 화학적, 지식에, 벽을느끼는,
9. lecture_id `77947` | score 0.8866 | similarity 0.8676 | predicted 4.65/5 | actual avg 4.58/5 | keywords: 매트랩, 강의력, 매우, 과제가, 있으면, 영어, 신시, 한국어로, 좋으시고, 과제, 시험, 도움이, 어려웠는데, 되면, 정도만, 금방, 오래, 교수님이, 하지만
10. lecture_id `45314` | score 0.8814 | similarity 0.8746 | predicted 4.49/5 | actual avg 4.64/5 | keywords: 설명, 다시, 쉽게, 시험문제는, 예제를, 무난무난, 연습문제에서, 책에, 해주심, 시험문제가, 과제에서, 잘해주시고, 과제, 예제, 거의, 교수님이, 한국어로,
