"""
"""

from config import FeatureConfig
from grid_search import GridSearch


def test_grid_search():
    feature_config = dict(BIO_na_strategy="keep", binarize_variables=False)  # drop, keep
    grid_search = GridSearch(feature_config=feature_config)
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
    test_grid_search()
