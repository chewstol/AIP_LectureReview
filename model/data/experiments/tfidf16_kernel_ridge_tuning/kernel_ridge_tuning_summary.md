# Kernel Ridge Hyperparameter Tuning

## Setup

- input: structured features 10 + category TF-IDF features 6
- model: Kernel Ridge with RBF kernel
- validation: 5-fold cross-validation
- alpha candidates: 11 values from `1e-6` to `0.1`
- gamma candidates: 10 values from `1e-6` to `0.03`

## Result

```text
Best alpha = 0.000001
Best gamma = 0.00001
CV RMSE = 0.084056
```

The previous coarse-grid setting (`alpha=0.01`, `gamma=0.001`) produced RMSE `0.084306`. The refined search improved RMSE by about 0.30%.

## Interpretation

- The heatmap shows a broad low-error valley rather than a sharply isolated optimum.
- Small gamma values produce a smooth decision function that is close to a linear relationship.
- Very large gamma values make the model react too locally to individual lectures and increase validation error.
- The numerical minimum occurs at very weak regularization, but nearby combinations have nearly identical performance.
- Because fold-to-fold RMSE standard deviation is about `0.011`, the practical difference between the best and the previous setting is small.

Recommended presentation wording:

> We jointly searched the Kernel Ridge regularization strength and RBF kernel width using 5-fold cross-validation. The lowest RMSE was obtained at alpha=1e-6 and gamma=1e-5. The heatmap showed a broad stable region at small alpha and gamma values rather than a sharp optimum.
