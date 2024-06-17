"""
"""

from inspect import isclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import CFG

__all__ = [
    "CLFMixin",
    "SizeMixin",
    "get_activation",
]


def compute_module_size(module: nn.Module, human: bool = False, dtype: str = "float32") -> Union[int, str]:
    """
    compute the size (number of parameters) of a module

    Parameters
    ----------
    module: Module,
        a torch Module
    human: bool, default False,
        return size in a way that is easy to read by a human,
        by appending a suffix corresponding to the unit (K, M, G, T, P)
    dtype: str, default "float32",
        data type of the module parameters, one of "float16", "float32", "float64"

    Returns
    -------
    n_params: int,
        size (number of parameters) of this torch module

    """
    module_parameters = filter(lambda p: p.requires_grad, module.parameters())
    n_params = sum([np.prod(p.size()) for p in module_parameters])
    if human:
        n_params = n_params * {"float16": 2, "float32": 4, "float64": 8}[dtype.lower()] / 1024
        div_count = 0
        while n_params >= 1024:
            n_params /= 1024
            div_count += 1
        # cvt_dict = {0:"K", 1:"M", 2:"G", 3:"T", 4:"P"}
        cvt_dict = {c: u for c, u in enumerate(list("KMGTP"))}
        n_params = f"""{n_params:.1f}{cvt_dict[div_count]}"""
    return n_params


class SizeMixin:
    """
    mixin class for size related methods
    """

    __name__ = "SizeMixin"

    @property
    def module_size(self) -> int:
        return compute_module_size(
            self,
            dtype=self.dtype_,
        )

    @property
    def module_size_(self) -> str:
        return compute_module_size(
            self,
            human=True,
            dtype=self.dtype_,
        )

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32
        except Exception as err:
            raise err  # unknown error

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
        except Exception as err:
            raise err  # unknown error

    @property
    def dtype_(self) -> str:
        return str(self.dtype).replace("torch.", "")

    @property
    def device_(self) -> str:
        return str(self.device)


class CLFMixin:
    """
    mixin for classifiers
    """

    __name__ = "CLFMixin"

    def predict_proba(self, input: Union[Tensor, np.ndarray], multi_label: bool = False) -> np.ndarray:
        """ """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).float().to(self.device)
        output = self.forward(input)
        if multi_label:
            return torch.sigmoid(output).cpu().detach().numpy()
        return torch.softmax(output, dim=-1).cpu().detach().numpy()

    def predict(
        self,
        input: Union[Tensor, np.ndarray],
        thr: Optional[float] = None,
        class_map: Optional[Dict[int, str]] = None,
    ) -> list:
        """ """
        proba = self.predict_proba(input, multi_label=thr is not None)
        if thr is None:
            output = proba.argmax(axis=-1).tolist()
            if class_map is not None:
                output = [class_map[i] for i in output]
            return output
        output = [[] for _ in range(input.shape[0])]
        indices = np.where(proba > thr)
        if len(indices) > 2:
            raise ValueError("multi-label classification is not supported for output of 3 dimensions or more")
        for i, j in zip(*indices):
            output[i].append(j)
        for idx in range(len(output)):
            if len(output[idx]) == 0:
                output[idx] = [proba[idx].argmax()]
        if class_map is not None:
            output = [[class_map[i] for i in item] for item in output]
        return output


try:
    Mish = nn.Mish  # pytorch added in version 1.9
    Mish.__name__ = "Mish"
except Exception:

    class Mish(nn.Module):
        """The Mish activation"""

        __name__ = "Mish"

        def __init__(self):
            """ """
            super().__init__()

        def forward(self, input: Tensor) -> Tensor:
            """ """
            return input * (torch.tanh(F.softplus(input)))


try:
    Swish = nn.SiLU  # pytorch added in version 1.7
    Swish.__name__ = "Swish"
    SiLU = nn.SiLU
except Exception:

    class Swish(nn.Module):
        """The Swish activation"""

        __name__ = "Swish"

        def __init__(self):
            """ """
            super().__init__()

        def forward(self, input: Tensor) -> Tensor:
            """ """
            return input * F.sigmoid(input)

    SiLU = Swish
    SiLU.__name__ = "SiLU"


try:
    Hardswish = nn.Hardswish  # pytorch added in version 1.6
    Hardswish.__name__ = "Hardswish"
except Exception:

    class Hardswish(nn.Module):
        r"""Applies the hardswish function, element-wise, as described in the paper:
        `Searching for MobileNetV3`_.
        .. math::
            \text{Hardswish}(x) = \begin{cases}
                0 & \text{if~} x \le -3, \\
                x & \text{if~} x \ge +3, \\
                x \cdot (x + 3) /6 & \text{otherwise}
            \end{cases}
        .. _`Searching for MobileNetV3`:
            https://arxiv.org/abs/1905.02244
        """

        __name__ = "Hardswish"

        def __init__(self):
            """ """
            super().__init__()

        def forward(self, input: Tensor) -> Tensor:
            """ """
            return torch.clamp(input * (3 + input) / 6, min=0, max=input)


Activations = CFG()
Activations.mish = Mish
Activations.swish = Swish
Activations.hardswish = Hardswish
Activations.hard_swish = Hardswish
Activations.relu = nn.ReLU
Activations.relu6 = nn.ReLU6
Activations.rrelu = nn.RReLU
Activations.leaky = nn.LeakyReLU
Activations.leaky_relu = Activations.leaky
Activations.gelu = nn.GELU
Activations.silu = SiLU
Activations.elu = nn.ELU
Activations.celu = nn.CELU
Activations.selu = nn.SELU
Activations.glu = nn.GLU
Activations.prelu = nn.PReLU
Activations.tanh = nn.Tanh
Activations.hardtanh = nn.Hardtanh
Activations.sigmoid = nn.Sigmoid
Activations.hardsigmoid = nn.Hardsigmoid
Activations.softmax = nn.Softmax


def get_activation(act: Union[str, nn.Module, type(None)], kw_act: Optional[dict] = None) -> Optional[nn.Module]:
    """

    Parameters
    ----------
    act: str or nn.Module or None,
        name or the class or an instance of the activation, or None
    kw_act: dict, optional,
        keyword arguments for the activation

    Returns
    -------
    nn.Module or None,
        the class of the activation or an instance of the activation, or None
    """
    if act is None:
        return act
    if isclass(act):
        _act = act
    elif isinstance(act, str):
        if act.lower() not in Activations:
            raise ValueError(f"activation `{act}` not supported")
        _act = Activations[act.lower()]
    elif isinstance(act, nn.Module):
        return act
    else:
        raise ValueError(f"activation `{act}` not supported")
    if kw_act is None:
        return _act
    return _act(**kw_act)
