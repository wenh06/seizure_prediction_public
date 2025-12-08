""" """

from typing import Optional, Sequence, Union

import torch.nn as nn

from .utils import CLFMixin, SizeMixin, get_activation

__all__ = ["SeizureMLP"]


class SeizureMLP(nn.Sequential, CLFMixin, SizeMixin):
    """multi-layer perceptron for seizure prediction"""

    __name__ = "SeizureMLP"

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim_hidden: Optional[Union[int, Sequence[int]]] = [128, 64],
        activations: Union[str, Sequence[str]] = "mish",
        activation_kw: Optional[Union[dict, Sequence[dict]]] = None,
        dropouts: Union[float, Sequence[float]] = 0.2,
    ) -> None:
        """ """
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        dims = []
        if dim_hidden is not None:
            if isinstance(dim_hidden, int):
                dims = [
                    dim_hidden,
                ]
            else:
                dims = list(dim_hidden)
        if isinstance(activations, str):
            activations = [activations] * len(dims)
        if isinstance(activation_kw, dict) or activation_kw is None:
            activation_kw = [activation_kw or {}] * len(dims)
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(dims)
        dim_in = n_features
        for i, dim in enumerate(dims):
            self.add_module(f"linear_{i+1}", nn.Linear(dim_in, dim))
            self.add_module(
                f"{activations[i]}_{i+1}",
                get_activation(activations[i], activation_kw[i]),
            )
            if dropouts[i] > 0:
                self.add_module(f"dropout_{i+1}", nn.Dropout(dropouts[i]))
            dim_in = dim
        self.add_module(f"linear_{len(dims)+1}", nn.Linear(dim_in, n_classes))
