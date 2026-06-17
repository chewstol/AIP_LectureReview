# Extended Model Experiment

전체 753개 강의에 대해 동일한 5-fold cross-validation으로 표형 데이터 모델을 비교했습니다.

## Feature Sets

- `all_31`: 31개 feature. 별점 기반 감성 feature 포함.
- `leakage_safe`: 16개 feature. 별점으로 만든 감성 비율 및 긍정/부정 TF-IDF 제외.

`all_31` 성능은 rating-derived feature 때문에 낙관적으로 측정될 수 있으므로, 최종 연구 결과에는 `leakage_safe`를 함께 보고해야 합니다.

## Best Results: all_31

| Rank | Model | Parameters | MSE | RMSE | MAE |
|---:|---|---|---:|---:|---:|
| 1 | Ridge | `ridge__alpha=10.0` | 0.003234 | 0.056647 | 0.042048 |
| 2 | Histogram Gradient Boosting | `l2_regularization=1.0;learning_rate=0.05;max_iter=100;max_leaf_nodes=15` | 0.003333 | 0.057376 | 0.041920 |
| 3 | Kernel Ridge-RBF | `kernelridge__alpha=0.01;kernelridge__gamma=0.001` | 0.003357 | 0.057819 | 0.042217 |
| 4 | Random Forest | `max_depth=12;min_samples_leaf=3` | 0.003386 | 0.057814 | 0.042159 |
| 5 | Gradient Boosting | `learning_rate=0.05;loss=squared_error;max_depth=2;n_estimators=300` | 0.003481 | 0.058672 | 0.042711 |
| 6 | Extra Trees | `max_depth=8;min_samples_leaf=6` | 0.003484 | 0.058574 | 0.042564 |
| 7 | SVR-RBF | `svr__C=0.3;svr__epsilon=0.03;svr__gamma=scale` | 0.004034 | 0.063225 | 0.046583 |
| 8 | Stacking Ensemble | `` | 0.004700 | 0.068080 | 0.052470 |

## Best Results: leakage_safe

| Rank | Model | Parameters | MSE | RMSE | MAE |
|---:|---|---|---:|---:|---:|
| 1 | Kernel Ridge-RBF | `kernelridge__alpha=0.01;kernelridge__gamma=0.001` | 0.007226 | 0.084306 | 0.061689 |
| 2 | Ridge | `ridge__alpha=10.0` | 0.007292 | 0.084741 | 0.062210 |
| 3 | Histogram Gradient Boosting | `l2_regularization=0.0;learning_rate=0.05;max_iter=100;max_leaf_nodes=7` | 0.007656 | 0.086674 | 0.062757 |
| 4 | Random Forest | `max_depth=6;min_samples_leaf=3` | 0.007666 | 0.086671 | 0.063571 |
| 5 | Extra Trees | `max_depth=8;min_samples_leaf=3` | 0.007735 | 0.087158 | 0.063220 |
| 6 | Gradient Boosting | `learning_rate=0.05;loss=huber;max_depth=2;n_estimators=100` | 0.007819 | 0.087399 | 0.062753 |
| 7 | SVR-RBF | `svr__C=0.3;svr__epsilon=0.05;svr__gamma=scale` | 0.008245 | 0.090054 | 0.066561 |
| 8 | Stacking Ensemble | `` | 0.009028 | 0.094290 | 0.072643 |

