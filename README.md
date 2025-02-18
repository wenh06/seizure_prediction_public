# seizure_prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seizure-prediction.streamlit.app/)
[![autotest](https://github.com/wenh06/seizure_prediction_public/actions/workflows/run-pytest.yml/badge.svg?branch=autotest)](https://github.com/wenh06/seizure_prediction_public/actions/workflows/run-pytest.yml)

Seizure prediction using clinical data.

This is a copy of a private repository, but **without** the raw data and trained models.

[Visualization](images) were done using functions in [`viz.py`](viz.py)

Risk differences analysis were conducted with [`diff-binom-confint`](https://pypi.org/project/diff-binom-confint/), which is also hosted on [Github](https://github.com/DeepPSP/DBCI/).

Serving API: http://101.43.135.121:11111/seizure_prediction (POST only)

Online APP: http://101.43.135.121:8501/ | https://seizure-prediction.streamlit.app/

The [Test action](.github/workflows/run-pytest.yml) runs a minimal test (demo) of the code using Github Actions.

## File/folder description

- [`app`](app) - Streamlit app.
- [`data`](data) - Data folder, **excluding** the raw data.
- [`images`](images) - Image folder containing visualizations of the data and experiment results.
- [`nn`](nn) - Model factory for neural networks (mainly MLP).
- [`config.py`](config.py) - Configuration file, including the parameters/hyperparameters for the models.
- [`data_processing.py`](data_processing.py) - Data processing functions.
- [`feature_selection.py`](feature_selection.py) - Feature selection functions.
- [`grid_search.py`](grid_search.py) - Grid search for hyperparameters.
- [`metrics.py`](metrics.py) - Evaluation metrics.
- [`models.py`](models.py) - Model factory for non-neural network models.
- [`risk_diff.py`](risk_diff.py) - Binomial risk difference analysis.
- [`service.py`](service.py) - Flask service for serving the model.
- [`utils.py`](utils.py) - Utility functions.
- [`viz.py`](viz.py) - Visualization functions.

There are also docker files for building the containers for different purposes, and requirements files along with them.

## Data distribution

<details>
<summary>Click to expand!</summary>

  Age distribution         |  Gender distribution
:-------------------------:|:-------------------------:
![Age distribution](images/age_distribution.svg) | ![Gender distribution](images/sex_distribution.svg)

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

### ROC of one typical experiment

<details>
<summary>Click to expand!</summary>

<img src="./images/roc_curve_example_no_over_sampling.svg" alt="ROC" width=600>

:point_right: [Back to top](#seizure_prediction)

</details>

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
