# Final Six-model Comparison

- input: structured 10 + KoBERT PCA32
- validation: 5-fold cross-validation

| Rank | Model | MSE | RMSE | MAE |
|---:|---|---:|---:|---:|
| 1 | Ridge | 0.006933 | 0.082837 | 0.060989 |
| 2 | Gradient Boosting | 0.007483 | 0.085815 | 0.062967 |
| 3 | Random Forest | 0.007643 | 0.086617 | 0.063650 |
| 4 | SVR-RBF | 0.007932 | 0.088387 | 0.066300 |
| 5 | Content KNN | 0.011368 | 0.106021 | 0.081763 |
| 6 | MLP | 0.013176 | 0.114041 | 0.087872 |
