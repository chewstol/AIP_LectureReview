# KoBERT + Tabular Model Experiment

## Setup

- lectures: 753
- reviews embedded: 13,979
- text encoder: `skt/kobert-base-v1`
- review embedding: attention-mask mean pooling, 768 dimensions
- lecture embedding: mean of review embeddings
- dimensionality reduction: PCA fitted only on each training fold
- structured features: 10 objective lecture evaluation features
- validation: 5-fold cross-validation
- target: normalized average lecture rating

Individual review ratings were not used as input features.

The review text and target ratings come from the same set of existing reviews. This is appropriate for estimating the quality of a lecture that already has reviews, but it does not evaluate cold-start prediction for a new lecture with no reviews.

## Best Combined Results

| Rank | Features | Model | MSE | RMSE | MAE |
|---:|---|---|---:|---:|---:|
| 1 | Structured 10 + KoBERT PCA32 | Ridge | 0.006933 | 0.082837 | 0.060989 |
| 2 | Structured 10 + KoBERT PCA16 | Ridge | 0.006975 | 0.083018 | 0.061290 |
| 3 | Structured 10 + KoBERT PCA16 | XGBoost | 0.007082 | 0.083495 | 0.060629 |
| 4 | Structured 10 + KoBERT PCA16 | LightGBM | 0.007262 | 0.084422 | 0.061949 |

XGBoost produced the lowest MAE, but Ridge produced the best MSE and RMSE.

## Ablation

| Features | Best Ridge alpha | MSE | RMSE | MAE |
|---|---:|---:|---:|---:|
| Structured 10 only | 1 | 0.007384 | 0.085304 | 0.062752 |
| KoBERT PCA32 only | 100 | 0.011834 | 0.108486 | 0.083234 |
| Structured 10 + KoBERT PCA32 | 10 | 0.006933 | 0.082837 | 0.060989 |

KoBERT alone was weaker than the structured features. Combining both improved RMSE by about 2.9% over structured-only Ridge, indicating that the text embedding adds complementary information.

Compared with the previous leakage-safe Kernel Ridge result (`RMSE=0.084306`), the combined KoBERT-Ridge model improved RMSE by about 1.7%.

## Interpretation

- The multimodal feature combination was useful.
- The small dataset still favored a strongly regularized linear model.
- XGBoost and LightGBM did not outperform Ridge on RMSE.
- KoBERT was used as a frozen encoder; it was not fine-tuned on the lecture reviews.
- A sentence-similarity-trained Korean SBERT model may be more suitable for persona-to-lecture cosine similarity than base KoBERT.
