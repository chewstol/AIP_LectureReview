# TF-IDF 16-feature Model Comparison

- input: structured 10 + category TF-IDF 6
- validation: 5-fold cross-validation
- final model: Kernel Ridge-RBF

| Rank | Model | Best parameters | MSE | RMSE | MAE |
|---:|---|---|---:|---:|---:|
| 1 | Kernel Ridge-RBF | `kernelridge__alpha=0.01;kernelridge__gamma=0.001` | 0.007226 | 0.084306 | 0.061689 |
| 2 | Ridge | `ridge__alpha=10.0` | 0.007292 | 0.084741 | 0.062210 |
| 3 | Random Forest | `max_depth=6;min_samples_leaf=3` | 0.007666 | 0.086671 | 0.063571 |
| 4 | Gradient Boosting | `learning_rate=0.05;loss=huber;max_depth=2;n_estimators=100` | 0.007819 | 0.087399 | 0.062753 |
| 5 | Content KNN | `k=20` | 0.007856 | 0.087770 | 0.064634 |
| 6 | MLP | `activation=tanh;alpha=0.1;hidden_layer_sizes=(32, 16);learning_rate_init=0.003` | 0.008007 | 0.088816 | 0.065256 |
| 7 | SVR-RBF | `svr__C=0.3;svr__epsilon=0.05;svr__gamma=scale` | 0.008245 | 0.090054 | 0.066561 |
