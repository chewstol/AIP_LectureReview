# Leakage-safe KNN and Graph MLP

- lectures: 753
- features: 16
- validation: 5-fold cross-validation
- excluded: rating-derived sentiment ratios and positive/negative TF-IDF
- Graph MLP evaluation: fixed final epoch prediction; validation-best epoch was not used

| Rank | Model | Best parameters | MSE | RMSE | MAE |
|---:|---|---|---:|---:|---:|
| 1 | Content KNN | `k=20` | 0.007856 | 0.087770 | 0.064634 |
| 2 | Graph-augmented MLP | `hidden_dim=16;learning_rate=0.003;epochs=500` | 0.042775 | 0.206095 | 0.158103 |
