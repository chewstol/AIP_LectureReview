# KoBERT PCA32 + Structured 10: All Models

- lectures: 753
- input: structured features 10 + KoBERT PCA components 32
- PCA/scaling: fitted within each training fold
- validation: 5-fold cross-validation

| Rank | Model | Best parameters | MSE | RMSE | MAE |
|---:|---|---|---:|---:|---:|
| 1 | Ridge | `alpha=10.0` | 0.006933 | 0.082837 | 0.060989 |
| 2 | XGBoost | `colsample_bytree=0.8;learning_rate=0.03;max_depth=2;min_child_weight=8;n_estimators=500;reg_lambda=1.0;subsample=0.8` | 0.007329 | 0.084969 | 0.062306 |
| 3 | Extra Trees | `max_depth=None;max_features=1.0;min_samples_leaf=3` | 0.007451 | 0.085564 | 0.062329 |
| 4 | Gradient Boosting | `learning_rate=0.05;loss=squared_error;max_depth=2;n_estimators=300` | 0.007483 | 0.085815 | 0.062967 |
| 5 | Histogram Gradient Boosting | `l2_regularization=1.0;learning_rate=0.05;max_iter=100;max_leaf_nodes=7` | 0.007498 | 0.085839 | 0.062504 |
| 6 | LightGBM | `colsample_bytree=0.8;learning_rate=0.03;max_depth=3;min_child_samples=10;n_estimators=200;num_leaves=15;reg_lambda=1.0;subsample=0.8` | 0.007572 | 0.086318 | 0.062999 |
| 7 | Kernel Ridge-RBF | `alpha=0.01;gamma=0.0001` | 0.007640 | 0.087002 | 0.064194 |
| 8 | Random Forest | `max_depth=None;max_features=0.7;min_samples_leaf=3` | 0.007643 | 0.086617 | 0.063650 |
| 9 | SVR-RBF | `C=0.3;epsilon=0.05;gamma=0.001` | 0.007932 | 0.088387 | 0.066300 |
| 10 | Content KNN | `n_neighbors=20` | 0.011368 | 0.106021 | 0.081763 |
