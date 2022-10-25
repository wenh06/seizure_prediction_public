"""
"""

import re
import warnings
import inspect
import signal
from copy import deepcopy
from collections import Counter
from contextlib import contextmanager
from functools import reduce
from typing import Sequence, Tuple, Callable, Dict, List, Any

import numpy as np
import pandas as pd

from config import DEFAULTS


__all__ = [
    "list_sum",
    "stratified_train_test_split",
    "add_docstring",
    "get_kwargs",
    "ReprMixin",
    "separate_by_capital_letters",
    "timeout",
]


def list_sum(lst: Sequence[list]) -> list:
    """

    Parameters
    ----------
    lst: sequence of list,
        the sequence of lists to obtain the summation

    Returns
    -------
    l_sum: list,
        sum of `lst`,
        i.e. if lst = [list1, list2, ...], then l_sum = list1 + list2 + ...

    """
    l_sum = reduce(lambda a, b: a + b, lst, [])
    return l_sum


def stratified_train_test_split(
    df: pd.DataFrame,
    stratified_cols: Sequence[str],
    test_ratio: float = 0.2,
    reset_index: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Do stratified train-test split on the dataframe,

    Parameters
    ----------
    df: pd.DataFrame,
        dataframe to be split
    stratified_cols: sequence of str,
        columns to be stratified, assuming each column is a categorical variable
        each class in any of the columns will be
        split into train and test sets with an approximate ratio of `test_ratio`
    test_ratio: float, default 0.2,
        ratio of test set to the whole dataframe
    reset_index: bool, default True,
        whether to reset the index of the dataframes

    Returns
    -------
    df_train: pd.DataFrame,
        the dataframe of the train set
    df_test: pd.DataFrame,
        the dataframe of the test set

    For example,
    if one has a dataframe with columns `sex`, `nationality`, etc.,
    assuming `sex` includes `male`, `female`; `nationality` includes `Chinese`, `American`,
    and sets `stratified_cols = ["sex", "nationality"]` with `test_ratio = 0.2`,
    then approximately 20% of the male and 20% of the female subjects
    will be put into the test set,
    and **at the same time**, approximately 20% of the Chinese and 20% of the Americans
    lie in the test set as well.

    """
    invalid_cols = [
        col
        for col in stratified_cols
        if not all([v > 1 for v in Counter(df[col]).values()])
    ]
    if len(invalid_cols) > 0:
        warnings.warn(
            f"invalid columns: {invalid_cols}, "
            "each of which has classes with only one member (row), "
        )
    stratified_cols = [col for col in stratified_cols if col not in invalid_cols]
    df_inspection = df[stratified_cols].copy()
    for item in stratified_cols:
        all_entities = df_inspection[item].unique().tolist()
        entities_dict = {e: str(i) for i, e in enumerate(all_entities)}
        df_inspection[item] = df_inspection[item].apply(lambda e: entities_dict[e])

    inspection_col_name = "Inspection" * (
        max([len(c) for c in stratified_cols]) // 10 + 1
    )
    df_inspection[inspection_col_name] = ""
    for idx, row in df_inspection.iterrows():
        cn = "-".join([row[sc] for sc in stratified_cols])
        df_inspection.loc[idx, inspection_col_name] = cn
    item_names = df_inspection[inspection_col_name].unique().tolist()
    item_indices = {
        n: df_inspection.index[df_inspection[inspection_col_name] == n].tolist()
        for n in item_names
    }
    for n in item_names:
        DEFAULTS.RNG.shuffle(item_indices[n])

    test_indices = []
    for n in item_names:
        item_test_indices = item_indices[n][: round(test_ratio * len(item_indices[n]))]
        test_indices += item_test_indices
    df_test = df.loc[df.index.isin(test_indices)]
    df_train = df.loc[~df.index.isin(test_indices)]
    if reset_index:
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
    return df_train, df_test


def add_docstring(doc: str, mode: str = "replace") -> Callable:
    """
    decorator to add docstring to a function or a class

    Parameters
    ----------
    doc: str,
        the docstring to be added
    mode: str, default "replace",
        the mode of the docstring,
        can be "replace", "append" or "prepend",
        case insensitive

    """

    def decorator(func_or_cls: Callable) -> Callable:
        """ """

        pattern = "(\\s^\n){1,}"
        if mode.lower() == "replace":
            func_or_cls.__doc__ = doc
        elif mode.lower() == "append":
            tmp = re.sub(pattern, "", func_or_cls.__doc__)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", doc)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ += new_lines + doc
        elif mode.lower() == "prepend":
            tmp = re.sub(pattern, "", doc)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", func_or_cls.__doc__)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ = doc + new_lines + func_or_cls.__doc__
        else:
            raise ValueError(f"mode {mode} is not supported")
        return func_or_cls

    return decorator


def get_kwargs(func_or_cls: Callable, kwonly: bool = False) -> Dict[str, Any]:
    """
    get the kwargs of a function or class

    Parameters
    ----------
    func_or_cls: Callable,
        the function or class to get the kwargs of
    kwonly: bool, default False,
        whether to get the kwonly kwargs of the function or class

    Returns
    -------
    kwargs: Dict[str, Any],
        the kwargs of the function or class

    """
    fas = inspect.getfullargspec(func_or_cls)
    kwargs = {}
    if fas.kwonlydefaults is not None:
        kwargs = deepcopy(fas.kwonlydefaults)
    if kwonly:
        return kwargs
    if fas.defaults is not None:
        kwargs.update(
            {k: v for k, v in zip(fas.args[-len(fas.defaults) :], fas.defaults)}
        )
    return kwargs


def default_class_repr(c: object, align: str = "center", depth: int = 1) -> str:
    """

    Parameters
    ----------
    c: object,
        the object to be represented
    align: str, default "center",
        the alignment of the class arguments
    depth: int, default 1,
        the depth of the class arguments to display

    Returns
    -------
    str,
        the representation of the class

    """
    indent = 4 * depth * " "
    closing_indent = 4 * (depth - 1) * " "
    if not hasattr(c, "extra_repr_keys"):
        return repr(c)
    elif len(c.extra_repr_keys()) > 0:
        max_len = max([len(k) for k in c.extra_repr_keys()])
        extra_str = (
            "(\n"
            + ",\n".join(
                [
                    f"""{indent}{k.ljust(max_len, " ") if align.lower() in ["center", "c"] else k} = {default_class_repr(eval(f"c.{k}"),align,depth+1)}"""
                    for k in c.__dir__()
                    if k in c.extra_repr_keys()
                ]
            )
            + f"{closing_indent}\n)"
        )
    else:
        extra_str = ""
    return f"{c.__class__.__name__}{extra_str}"


class ReprMixin(object):
    """
    Mixin for enhanced __repr__ and __str__ methods.

    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """ """
        return []


def separate_by_capital_letters(string: str) -> str:
    """
    separate a string by capital letters,
    e.g. "HelloWorld" -> "Hello World"
    """
    separated = re.findall("[A-Z][^A-Z]*", string)
    if len(separated) > 0:
        return " ".join(separated)
    else:
        return string


@contextmanager
def timeout(duration: float):
    """
    A context manager that raises a `TimeoutError` after a specified time.

    Parameters
    ----------
    duration: float,
        the time duration in seconds,
        should be non-negative,
        0 for no timeout

    References
    ----------
    https://stackoverflow.com/questions/492519/timeout-on-a-function-call

    """
    if np.isinf(duration):
        duration = 0
    elif duration < 0:
        raise ValueError("duration must be non-negative")
    elif duration > 0:  # granularity is 1 second, so round up
        duration = max(1, int(duration))

    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)
