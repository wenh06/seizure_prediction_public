""" """

import warnings
from copy import deepcopy
from typing import List, Optional

import torch
from torch.utils.data import TensorDataset

from config import CFG, DataPreprocessConfig, FeatureConfig
from data_processing import get_training_data
from utils import ReprMixin


class SeizureDataset(TensorDataset, ReprMixin):
    def __init__(
        self,
        preprocess_config: Optional[CFG] = None,
        feature_config: Optional[CFG] = None,
        feature_set: str = "TDSB",
        training: bool = True,
    ) -> None:
        """ """
        if preprocess_config is None:
            self.preprocess_config = deepcopy(DataPreprocessConfig)
        else:
            _preprocess_config = deepcopy(preprocess_config)
            self.preprocess_config = deepcopy(DataPreprocessConfig)
            self.preprocess_config.update(_preprocess_config)

        if feature_config is None:
            self.feature_config = deepcopy(FeatureConfig)
        else:
            _feature_config = deepcopy(feature_config)
            self.feature_config = deepcopy(FeatureConfig)
            self.feature_config.update(_feature_config)
        if not self.feature_config.binarize_variables:
            warnings.warn("Binarize variables should be True. Changed to True.")
            self.feature_config.binarize_variables = True
        self.feature_set = feature_set
        self.training = training

        (
            X_train,
            y_train,
            X_test,
            y_test,
            self.__feature_list,
        ) = get_training_data(self.preprocess_config, self.feature_config, self.feature_set)

        if self.training:
            super().__init__(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        else:
            super().__init__(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    @property
    def feature_list(self) -> List[str]:
        return self.__feature_list

    @property
    def n_features(self) -> int:
        return len(self.feature_list)

    def extra_repr_keys(self) -> List[str]:
        return ["feature_set", "training"]
