# Ridge Alpha Diagnostics

Input:

- structured features: 10
- KoBERT embedding: PCA 32
- validation: 5-fold cross-validation

## Key Values

| Alpha | Train RMSE | Validation RMSE | Coefficient L2 norm |
|---:|---:|---:|---:|
| 0 | 0.077559 | 0.082871 | 2186.945 |
| 1 | 0.077799 | 0.082857 | 0.0695 |
| 3 | 0.077813 | 0.082841 | 0.0666 |
| 10 | 0.077885 | **0.082837** | 0.0596 |
| 30 | 0.078098 | 0.082926 | 0.0507 |
| 100 | 0.078552 | 0.083189 | 0.0432 |
| 1000 | 0.083336 | 0.087158 | 0.0275 |

## Interpretation

- With little or no regularization, the coefficient solution is unstable because of correlated input features.
- Validation RMSE gradually improves until `alpha=10`.
- `alpha=3~10` forms a broad, stable optimum rather than a sharply separated minimum.
- After `alpha=30`, both train and validation RMSE increase.
- At `alpha=1000`, excessive regularization causes clear underfitting.

Recommended presentation wording:

> We varied the Ridge regularization strength under the same 5-fold cross-validation setting. Validation RMSE reached its minimum at alpha=10. The difference around alpha=3~10 was small, indicating a stable optimal range, while stronger regularization increased both train and validation error.
