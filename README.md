# seizure_prediction

Seizure prediction using clinical data.

This is a copy of a private repository, but **without** the raw data and trained models.

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
