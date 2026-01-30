""" """

import math

from sklearn.model_selection import ParameterGrid

from config import CFG, FeatureConfig, GridSearchConfig
from grid_search import GridSearch


def simplify_grid_search_config(config: CFG) -> CFG:
    """Simplify the grid search configuration by retaining
    only half of each parameter's values.
    """
    simple_config = CFG()

    for attr_name in ["rf", "xgb", "gdbt", "svc", "lr", "bagging", "sk_mlp"]:
        if hasattr(config, attr_name):
            param_grid = getattr(config, attr_name)
            original_params = param_grid.param_grid[0]

            simplified_params = {k: v[: max(1, math.ceil(len(v) / 2))] for k, v in original_params.items()}

            # fix potential inconsistencies in simplified parameters
            if "warm_start" in simplified_params:
                simplified_params["warm_start"] = [True, False]

            setattr(simple_config, attr_name, ParameterGrid(simplified_params))

    return simple_config


def test_grid_search():
    SimpleGridSearchConfig = simplify_grid_search_config(GridSearchConfig)
    feature_config = dict(BIO_na_strategy="keep", binarize_variables=False)  # drop, keep
    grid_search = GridSearch(feature_config=feature_config, grid_search_config=SimpleGridSearchConfig)
    for feature_set in FeatureConfig.sets:
        for strategy in ["keep", "drop"]:
            feature_config["BIO_na_strategy"] = strategy
            grid_search.update_feature_config(config=feature_config)
            for model_name in ["lr", "rf", "gdbt", "bagging", "xgb", "svc"]:
                if model_name == "lr":
                    feature_config["binarize_variables"] = True
                else:
                    feature_config["binarize_variables"] = False
                grid_search.update_feature_config(config=feature_config)
                result = grid_search.search(model_name=model_name, feature_set=feature_set)
                print("*" * 80)
                print(f"feature_set: {feature_set}\nBIO_na_strategy: {strategy}\nmodel: {model_name}")
                print(f"scores: {result[2:]}")
                print("*" * 80)


if __name__ == "__main__":
    try:
        test_grid_search()
    except KeyboardInterrupt:
        print("Test cancelled by user.")
