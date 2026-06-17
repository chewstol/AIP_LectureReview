# Generated Persona Prediction Evaluation

## Scope

- Input personas: `data/generated_personas.json`
- Ground truth used here: each persona's `selected_reviews[].rate` on a 1-5 scale
- Quality model: Ridge regression reproduced from `recommend_topk.py`, trained on `data/model/lecture_nodes_with_text.csv`
- Ridge alpha: `10.0`
- Ranking score: `0.70 * preference_similarity + 0.30 * predicted_quality`
- Note: this evaluates how well generated persona examples align with the existing model; it is not a true held-out user study.

## Overall Results

| Vector field | Personas | Reviews | MAE | RMSE | Lecture avg MAE | Bias | Within 0.5 | Within 1.0 | Hit@10 | Hit@30 | Hit@100 | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `initial_preference_vector` | 20 | 200 | 0.651 | 0.803 | 0.637 | -0.096 | 48.0% | 84.5% | 27.0% | 39.5% | 68.0% | 62.8 |
| `aggregated_review_vector` | 20 | 200 | 0.651 | 0.803 | 0.637 | -0.096 | 48.0% | 84.5% | 21.0% | 31.5% | 58.0% | 101.7 |

## Per-Persona Results

| Vector field | Persona | Preset | Reviews | MAE | RMSE | Bias | Within 1.0 | Hit@30 | Median rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `aggregated_review_vector` | 1 | `low_workload` | 10 | 0.570 | 0.727 | -0.084 | 90.0% | 20.0% | 180.0 |
| `aggregated_review_vector` | 2 | `learning_quality` | 10 | 0.761 | 0.841 | -0.488 | 70.0% | 70.0% | 4.0 |
| `aggregated_review_vector` | 3 | `balanced` | 10 | 0.629 | 0.701 | 0.100 | 90.0% | 40.0% | 54.5 |
| `aggregated_review_vector` | 4 | `grade_focused` | 10 | 0.536 | 0.576 | -0.514 | 100.0% | 10.0% | 98.5 |
| `aggregated_review_vector` | 5 | `balanced` | 10 | 0.859 | 1.071 | -0.186 | 80.0% | 10.0% | 230.0 |
| `aggregated_review_vector` | 6 | `grade_focused` | 10 | 0.609 | 0.812 | 0.049 | 90.0% | 20.0% | 96.0 |
| `aggregated_review_vector` | 7 | `learning_quality` | 10 | 0.740 | 0.908 | -0.303 | 70.0% | 40.0% | 49.5 |
| `aggregated_review_vector` | 8 | `low_workload` | 10 | 0.263 | 0.319 | -0.238 | 100.0% | 40.0% | 65.5 |
| `aggregated_review_vector` | 9 | `learning_quality` | 10 | 0.586 | 0.661 | -0.539 | 90.0% | 40.0% | 54.0 |
| `aggregated_review_vector` | 10 | `grade_focused` | 10 | 0.794 | 0.996 | 0.093 | 80.0% | 10.0% | 88.0 |
| `aggregated_review_vector` | 11 | `low_workload` | 10 | 0.542 | 0.598 | -0.542 | 90.0% | 40.0% | 111.5 |
| `aggregated_review_vector` | 12 | `low_workload` | 10 | 0.416 | 0.520 | -0.337 | 90.0% | 40.0% | 102.5 |
| `aggregated_review_vector` | 13 | `grade_focused` | 10 | 0.573 | 0.733 | -0.191 | 90.0% | 20.0% | 59.0 |
| `aggregated_review_vector` | 14 | `low_workload` | 10 | 0.424 | 0.470 | -0.236 | 100.0% | 40.0% | 97.5 |
| `aggregated_review_vector` | 15 | `balanced` | 10 | 0.608 | 0.763 | -0.154 | 90.0% | 30.0% | 73.5 |
| `aggregated_review_vector` | 16 | `grade_focused` | 10 | 0.660 | 0.848 | 0.165 | 80.0% | 10.0% | 124.0 |
| `aggregated_review_vector` | 17 | `balanced` | 10 | 0.767 | 1.053 | 0.243 | 80.0% | 70.0% | 26.0 |
| `aggregated_review_vector` | 18 | `grade_focused` | 10 | 0.841 | 1.058 | 0.289 | 70.0% | 10.0% | 86.0 |
| `aggregated_review_vector` | 19 | `learning_quality` | 10 | 0.865 | 1.117 | 0.483 | 70.0% | 40.0% | 260.0 |
| `aggregated_review_vector` | 20 | `balanced` | 10 | 0.983 | 1.288 | 0.474 | 70.0% | 30.0% | 174.0 |
| `initial_preference_vector` | 1 | `low_workload` | 10 | 0.570 | 0.727 | -0.084 | 90.0% | 10.0% | 202.0 |
| `initial_preference_vector` | 2 | `learning_quality` | 10 | 0.761 | 0.841 | -0.488 | 70.0% | 70.0% | 3.0 |
| `initial_preference_vector` | 3 | `balanced` | 10 | 0.629 | 0.701 | 0.100 | 90.0% | 50.0% | 33.5 |
| `initial_preference_vector` | 4 | `grade_focused` | 10 | 0.536 | 0.576 | -0.514 | 100.0% | 40.0% | 39.5 |
| `initial_preference_vector` | 5 | `balanced` | 10 | 0.859 | 1.071 | -0.186 | 80.0% | 20.0% | 131.5 |
| `initial_preference_vector` | 6 | `grade_focused` | 10 | 0.609 | 0.812 | 0.049 | 90.0% | 30.0% | 42.0 |
| `initial_preference_vector` | 7 | `learning_quality` | 10 | 0.740 | 0.908 | -0.303 | 70.0% | 40.0% | 62.0 |
| `initial_preference_vector` | 8 | `low_workload` | 10 | 0.263 | 0.319 | -0.238 | 100.0% | 50.0% | 33.0 |
| `initial_preference_vector` | 9 | `learning_quality` | 10 | 0.586 | 0.661 | -0.539 | 90.0% | 40.0% | 46.5 |
| `initial_preference_vector` | 10 | `grade_focused` | 10 | 0.794 | 0.996 | 0.093 | 80.0% | 40.0% | 56.0 |
| `initial_preference_vector` | 11 | `low_workload` | 10 | 0.542 | 0.598 | -0.542 | 90.0% | 40.0% | 94.0 |
| `initial_preference_vector` | 12 | `low_workload` | 10 | 0.416 | 0.520 | -0.337 | 90.0% | 60.0% | 15.0 |
| `initial_preference_vector` | 13 | `grade_focused` | 10 | 0.573 | 0.733 | -0.191 | 90.0% | 40.0% | 42.0 |
| `initial_preference_vector` | 14 | `low_workload` | 10 | 0.424 | 0.470 | -0.236 | 100.0% | 40.0% | 62.5 |
| `initial_preference_vector` | 15 | `balanced` | 10 | 0.608 | 0.763 | -0.154 | 90.0% | 40.0% | 45.0 |
| `initial_preference_vector` | 16 | `grade_focused` | 10 | 0.660 | 0.848 | 0.165 | 80.0% | 30.0% | 53.5 |
| `initial_preference_vector` | 17 | `balanced` | 10 | 0.767 | 1.053 | 0.243 | 80.0% | 50.0% | 25.0 |
| `initial_preference_vector` | 18 | `grade_focused` | 10 | 0.841 | 1.058 | 0.289 | 70.0% | 30.0% | 39.0 |
| `initial_preference_vector` | 19 | `learning_quality` | 10 | 0.865 | 1.117 | 0.483 | 70.0% | 40.0% | 159.0 |
| `initial_preference_vector` | 20 | `balanced` | 10 | 0.983 | 1.288 | 0.474 | 70.0% | 30.0% | 72.5 |

## Largest Rating Errors

| Vector field | Persona | Preset | Lecture | Actual | Predicted | Lecture avg | Error | Rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `initial_preference_vector` | 17 | `balanced` | 2086884 | 1.000 | 3.447 | 3.650 | 2.447 | 164 |
| `initial_preference_vector` | 20 | `balanced` | 2086884 | 1.000 | 3.447 | 3.650 | 2.447 | 185 |
| `aggregated_review_vector` | 17 | `balanced` | 2086884 | 1.000 | 3.447 | 3.650 | 2.447 | 296 |
| `aggregated_review_vector` | 20 | `balanced` | 2086884 | 1.000 | 3.447 | 3.650 | 2.447 | 273 |
| `initial_preference_vector` | 5 | `balanced` | 1688981 | 1.000 | 3.340 | 3.780 | 2.340 | 292 |
| `initial_preference_vector` | 19 | `learning_quality` | 1688981 | 1.000 | 3.340 | 3.780 | 2.340 | 366 |
| `initial_preference_vector` | 20 | `balanced` | 1688981 | 1.000 | 3.340 | 3.780 | 2.340 | 297 |
| `aggregated_review_vector` | 5 | `balanced` | 1688981 | 1.000 | 3.340 | 3.780 | 2.340 | 370 |
| `aggregated_review_vector` | 19 | `learning_quality` | 1688981 | 1.000 | 3.340 | 3.780 | 2.340 | 356 |
| `aggregated_review_vector` | 20 | `balanced` | 1688981 | 1.000 | 3.340 | 3.780 | 2.340 | 303 |

## Interpretation

- Rating MAE/RMSE compares predicted lecture quality to individual selected review ratings. Individual reviews are noisier than lecture averages, so this is stricter than the original lecture-average CV task.
- Positive bias means the model tends to predict higher than the selected review rating; negative bias means it tends to underpredict.
- Hit@K checks whether the persona vector places the selected-review lectures inside the recommender's Top-K list.
- In this run, `initial_preference_vector` ranked the selected-review lectures better than `aggregated_review_vector` overall. That suggests the generated initial persona vectors are closer to the recommender's feature space than the post-sampled aggregate vectors.
- `initial_preference_vector` is the cleaner cold-start persona input. `aggregated_review_vector` is useful as an observed-review profile, but it should not automatically be assumed to improve ranking.
