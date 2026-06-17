# Evidence for Ridge Baseline Selection

## Correlation With Average Rating

| Feature | Pearson r | Univariate linear R2 | Direction |
|---|---:|---:|---|
| Grading generous | +0.632 | 0.400 | Positive |
| Grading strict | -0.632 | 0.400 | Negative |
| Low assignment load | +0.410 | 0.168 | Positive |
| High assignment load | -0.410 | 0.168 | Negative |
| Low teamwork load | +0.270 | 0.073 | Positive |
| High teamwork load | -0.270 | 0.073 | Negative |

Grading and assignment features show clear monotonic linear trends with average lecture rating. Attendance and exam features have weaker individual correlations.

## Model-level Evidence

For the final feature set (`structured 10 + KoBERT PCA32`), Ridge achieved the best 5-fold CV RMSE:

| Model | RMSE |
|---|---:|
| Ridge | 0.082837 |
| XGBoost | 0.084969 |
| Extra Trees | 0.085564 |
| Gradient Boosting | 0.085815 |
| Kernel Ridge-RBF | 0.087002 |

## Recommended Wording

The data does not prove that every relationship is linear. A defensible conclusion is:

> Exploratory analysis showed clear linear trends between several structured features, especially grading and assignment-related features, and the average rating. Ridge was therefore selected as an interpretable baseline, while nonlinear models were also evaluated to verify whether more complex relationships improved performance.
