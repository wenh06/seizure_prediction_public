# seizure_prediction

Seizure prediction using clinical data.

This is a copy of a private repository, but **without** the raw data and trained models.

## Data distribution

<details>
<summary>Click to expand!</summary>

  Age distribution         |  Sex distribution
:-------------------------:|:-------------------------:
![Age distribution](images/age_distribution.svg) | ![Sex distribution](images/sex_distribution.svg)

:point_right: [Back to top](#seizure_prediction)

</details>

## Models

- Logistic regression
- ~~Support vector classifier~~
- Multi-layer perceptron
- Random forest
- Bagging classifier
- Gradient boosting classifier
- XGBoost classifier

## Grid search

Execute the following command for a complete grid search over all the [models](#models)

```bash
nohup python grid_search.py > /dev/null 2>&1 & echo $! > ./log/gs.pid
```

### Aggregation of all grid search experiments

<details>
<summary>Click to expand!</summary>

  BIO NA drop              |  BIO NA keep
:-------------------------:|:-------------------------:
![BIO NA drop](images/grid_search_agg_all_BIO_NA_drop.svg) | ![BIO NA keep](images/grid_search_agg_all_BIO_NA_keep.svg)

:point_right: [Back to top](#seizure_prediction)

</details>

### Aggregation of grid search experiments on different feature sets

<details>
<summary>Click to expand!</summary>

  TD              |  TDS             |  TDB             |  TDSB
:----------------:|:----------------:|:----------------:|:----------------:
![TD](images/grid_search_agg_TD.svg) | ![TDS](images/grid_search_agg_TDS.svg) | ![TDS](images/grid_search_agg_TDB.svg) | ![TDS](images/grid_search_agg_TDSB.svg)

:point_right: [Back to top](#seizure_prediction)

</details>

## Feature importance analysis

### `SHAP` summary (top 10 features)

<details>
<summary>Click to expand!</summary>

![Dot plot](images/SHAP-summary-dot-top10-rf_TDSB_drop.svg)

  Violin plot              |  Bar plot
:-------------------------:|:-------------------------:
![Violin plot](images/SHAP-summary-violin-top10-rf_TDSB_drop.svg) | ![Bar plot](images/SHAP-summary-bar-top10-rf_TDSB_drop.svg)

:point_right: [Back to top](#seizure_prediction)

</details>

### Permutation importance

<details>
<summary>Click to expand!</summary>

  Run 1                    |  Run 2
:-------------------------:|:-------------------------:
![Run 1](images/permutation-importance-rf_TDSB_drop-1.svg) | ![Run 2](images/permutation-importance-rf_TDSB_drop-2.svg)

:point_right: [Back to top](#seizure_prediction)

</details>
