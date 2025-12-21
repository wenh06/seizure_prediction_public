""" """

import inspect
import os
import pickle
import re
import threading
import warnings
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import IO, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
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
    "make_serializable",
    "CrossPlatformUnpickler",
    "compute_iqr",
    "compute_iqr_fences",
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
    invalid_cols = [col for col in stratified_cols if not all([v > 1 for v in Counter(df[col]).values()])]
    if len(invalid_cols) > 0:
        warnings.warn(f"invalid columns: {invalid_cols}, " "each of which has classes with only one member (row), ")
    stratified_cols = [col for col in stratified_cols if col not in invalid_cols]
    df_inspection = df[stratified_cols].copy()
    for item in stratified_cols:
        all_entities = df_inspection[item].unique().tolist()
        entities_dict = {e: str(i) for i, e in enumerate(all_entities)}
        df_inspection[item] = df_inspection[item].apply(lambda e: entities_dict[e])

    inspection_col_name = "Inspection" * (max([len(c) for c in stratified_cols]) // 10 + 1)
    df_inspection[inspection_col_name] = ""
    for idx, row in df_inspection.iterrows():
        cn = "-".join([row[sc] for sc in stratified_cols])
        df_inspection.loc[idx, inspection_col_name] = cn  # type: ignore
    item_names = df_inspection[inspection_col_name].unique().tolist()
    item_indices = {n: df_inspection.index[df_inspection[inspection_col_name] == n].tolist() for n in item_names}
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
            tmp = re.sub(pattern, "", func_or_cls.__doc__ or "")
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", doc)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ += new_lines + doc  # type: ignore
        elif mode.lower() == "prepend":
            tmp = re.sub(pattern, "", doc)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", func_or_cls.__doc__ or "")
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ = doc + new_lines + (func_or_cls.__doc__ or "")
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
        kwargs.update({k: v for k, v in zip(fas.args[-len(fas.defaults) :], fas.defaults)})
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


def separate_by_capital_letters(
    string: str,
    capitalize: bool = True,
    drop: Optional[Union[str, Sequence[str]]] = None,
) -> str:
    """
    separate a string by capital letters,
    e.g. "HelloWorld" -> "Hello World",
    or -> "Hello world" if `capitalize` is True
    """
    if drop is None:
        drop = []
    elif isinstance(drop, str):
        drop = [drop]
    separated = re.findall("[A-Z][^A-Z]*", string)
    separated = list(filter(lambda w: w.lower() not in drop or w == "", separated))
    if len(separated) > 0:
        new_string = " ".join(separated)
    else:
        new_string = string
    if capitalize:
        new_string = new_string.capitalize()
    return new_string


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

    if duration <= 0:
        yield
        return

    timeout_event = threading.Event()
    timeout_occurred = [False]

    def timeout_handler():
        timeout_occurred[0] = True
        timeout_event.set()

    timer = threading.Timer(duration, timeout_handler)
    timer.daemon = True
    timer.start()

    try:
        yield
    finally:
        timer.cancel()
        if timeout_occurred[0]:
            raise TimeoutError(f"block timed out after {duration} seconds")


def make_serializable(
    x: Any, drop_unserializable: bool = True, drop_paths: bool = False
) -> Optional[Union[list, dict, str, int, float, bool]]:
    """Recursively convert object into JSON-serializable form.

    Rules
    -----
    - NDArray → list
    - np.generic → Python scalar
    - dict → new dict with only serializable values
    - list/tuple → list with only serializable values
    - str/int/float/bool/None → kept
    - if drop_unserializable:
          anything else (like Path, custom classes) → dropped (return None)
      else:
          fallback to str(x)

    Parameters
    ----------
    x : Any
        Input object to be converted.
    drop_unserializable : bool, default=True
        Whether to drop unserializable objects (return None),
        or convert them to string with str(x).

        .. versionadded:: 0.0.32
    drop_paths : bool, default=False
        If True, drop all filesystem paths (Path objects and strings
        that look like paths).

        .. versionadded:: 0.0.32

    Returns
    -------
    Optional[Union[list, dict, str, int, float, bool]]
        A JSON-serializable object, or None if dropped.

    Examples
    --------
    >>> import numpy as np
    >>> make_serializable(np.array([1, 2, 3]))
    [1, 2, 3]
    >>> make_serializable({"a": np.float64(3.14), "b": Path("file.txt")})
    {'a': 3.14, 'b': 'file.txt'}
    >>> make_serializable({"a": np.float64(3.14), "b": Path("file.txt")}, drop_paths=True)
    {'a': 3.14}

    """

    if isinstance(x, np.ndarray):
        return make_serializable(x.tolist(), drop_unserializable=drop_unserializable, drop_paths=drop_paths)

    elif isinstance(x, np.generic):
        return x.item()

    elif isinstance(x, dict):
        result = {}
        for k, v in x.items():
            v_serial = make_serializable(v, drop_unserializable=drop_unserializable, drop_paths=drop_paths)
            if v_serial is not None:
                result[k] = v_serial
        return result if result else None

    elif isinstance(x, (list, tuple)):
        result = []
        for v in x:
            v_serial = make_serializable(v, drop_unserializable=drop_unserializable, drop_paths=drop_paths)
            if v_serial is not None:
                result.append(v_serial)
        return result if result else None

    elif isinstance(x, (str, int, float, bool, type(None))):
        if isinstance(x, str) and drop_paths and _is_pathlike_string(x):
            return None
        return x

    elif isinstance(x, Path):
        if drop_paths:
            return None
        return str(x)

    else:
        if drop_unserializable:
            return None
        else:
            return str(x)


def _is_pathlike_string(s: str) -> bool:
    """Heuristically check if a string looks like a filesystem path."""
    if not isinstance(s, str):
        return False

    p = PurePath(s)
    if os.sep in s or (os.altsep and os.altsep in s):
        return True
    if s.startswith((".", "~")) or p.is_absolute():
        return True
    if p.suffix != "":
        return True
    if len(s) > 2 and s[1] == ":" and s[0].isalpha() and s[2] in ("/", "\\"):
        return True
    return False


class CrossPlatformUnpickler(pickle.Unpickler):
    """
    Custom unpickler for handling cross-platform path objects.

    This unpickler solves the issue of loading pickle files created on Windows
    (containing WindowsPath objects) on Linux/Mac systems, and vice versa.
    It provides options to convert path objects to strings or platform-specific
    Path objects during deserialization.
    """

    def __init__(self, file: IO, path_resolution: str = "string"):
        """
        Initialize the cross-platform unpickler.

        Parameters
        ----------
        file : file-like object
            File object to unpickle from.
        path_resolution : str, default "string"
            How to handle path objects during deserialization:

            - "string": Convert all path objects to strings
            - "posix": Convert to PosixPath (for Linux/Mac)
            - "windows": Convert to WindowsPath (will fall back to string on Linux)

        Raises
        ------
        ValueError
            If an invalid path_resolution option is provided.
        """
        super().__init__(file)
        if path_resolution not in ("string", "posix", "windows"):
            raise ValueError(f"Invalid path_resolution: {path_resolution}. " "Must be one of: 'string', 'posix', 'windows'")
        self.path_resolution = path_resolution

    def find_class(self, module: str, name: str):
        """
        Find and return the class for the given module and name.

        This method is called during unpickling to locate classes. It intercepts
        path-related classes and returns appropriate handlers to ensure
        cross-platform compatibility.

        Parameters
        ----------
        module : str
            Module name where the class is defined.
        name : str
            Class name.

        Returns
        -------
        Any
            The class or factory function to use during unpickling.

        Notes
        -----
        - For WindowsPath, PosixPath, and Path classes, returns a factory function
          that creates the appropriate object based on path_resolution setting.
        - For PurePath classes, returns the actual class as they are platform-independent.
        - For all other classes, uses the default unpickler behavior.
        """
        # Handle Path-related classes
        if module == "pathlib":
            if name in ("WindowsPath", "PosixPath", "Path"):
                # Return a factory function that creates path objects or strings
                def create_path(*args, **kwargs):
                    """
                    Factory function for creating path objects or strings.

                    Parameters
                    ----------
                    *args : tuple
                        Arguments passed to Path constructor.
                    **kwargs : dict
                        Keyword arguments passed to Path constructor.

                    Returns
                    -------
                    Union[str, Path]
                        Either a string representation of the path or a Path object,
                        depending on the path_resolution setting.
                    """
                    if self.path_resolution == "string":
                        # Convert to string
                        if args and len(args) == 1 and isinstance(args[0], str):
                            return args[0]
                        return str(Path(*args, **kwargs))
                    elif self.path_resolution == "posix":
                        # Convert to PosixPath
                        return Path(*args, **kwargs)
                    elif self.path_resolution == "windows":
                        # On Linux, we cannot create WindowsPath, so return string
                        # On Windows, this would return a WindowsPath
                        try:
                            from pathlib import WindowsPath

                            return WindowsPath(*args, **kwargs)
                        except (ImportError, NotImplementedError):
                            return str(Path(*args, **kwargs))

                # Set attributes to make pickle think this is a class
                create_path.__name__ = name
                create_path.__module__ = module
                return create_path

            # Handle PurePath classes (platform-independent)
            if name in ("PureWindowsPath", "PurePosixPath", "PurePath"):
                # PurePath classes can be created on all platforms
                if name == "PureWindowsPath":
                    return PureWindowsPath
                elif name == "PurePosixPath":
                    return PurePosixPath
                else:
                    return PurePath

        # Default behavior for all other classes
        return super().find_class(module, name)


def compute_iqr(arr: Union[Iterable[float], npt.NDArray[np.float64]]) -> Tuple[float, float]:
    """
    Calculate the Interquartile Range (IQR) interval of a numerical array, returned as a 2-tuple (Q1, Q3).

    The IQR "range" (interval) refers to the range between the first quartile (Q1, 25th percentile)
    and the third quartile (Q3, 75th percentile) of the data. This interval contains the middle 50%
    of the values and is robust to outliers. Note that the *IQR value* (a single number) is Q3 - Q1,
    while this function returns the interval (Q1, Q3) as requested.

    Parameters
    ----------
    arr : array-like
        Input numerical array (can be a Python list/tuple or numpy ndarray) containing
        integer or float values.

    Returns
    -------
    Tuple[float, float]
        A 2-tuple (Q1, Q3) representing the interquartile range interval:
        - First element: Q1 (25th percentile of the input array)
        - Second element: Q3 (75th percentile of the input array)

    Raises
    ------
    ValueError
        1. If the input array is empty (size = 0)
        2. If the input array contains non-numerical elements (e.g., strings, booleans)

    """
    try:
        np_arr = np.array(arr, dtype=np.float64)
    except ValueError as e:
        raise ValueError("Input array must contain only numerical values (int/float)") from e

    if np_arr.size == 0:
        raise ValueError("Input array cannot be empty")

    q1 = np.percentile(np_arr, 25).item()
    q3 = np.percentile(np_arr, 75).item()

    return (q1, q3)


def compute_iqr_fences(arr: Union[Iterable[float], npt.NDArray[np.float64]]) -> Tuple[float, float]:
    """
    Calculate Tukey's fences (IQR fences) for outlier detection, returned as (lower_fence, upper_fence).

    Tukey's fences are defined as:
    - Lower fence: Q1 - 1.5 × IQR
    - Upper fence: Q3 + 1.5 × IQR
    where IQR = Q3 - Q1 (Interquartile Range value). Data points outside this range are considered outliers.

    Parameters
    ----------
    arr : array-like
        Input numerical array (can be a Python list/tuple or numpy ndarray) containing
        integer or float values.

    Returns
    -------
    Tuple[float, float]
        A 2-tuple (lower_fence, upper_fence) representing Tukey's fences:
        - First element: Lower fence (Q1 - 1.5 × IQR)
        - Second element: Upper fence (Q3 + 1.5 × IQR)

    Raises
    ------
    ValueError
        1. If the input array is empty (size = 0)
        2. If the input array contains non-numerical elements (e.g., strings, booleans)

    """
    # Get Q1 and Q3 from the IQR interval function
    q1, q3 = compute_iqr(arr)

    # Calculate IQR value (single number: Q3 - Q1)
    iqr_value = q3 - q1

    # Compute Tukey's fences
    lower_fence = q1 - 1.5 * iqr_value
    upper_fence = q3 + 1.5 * iqr_value

    return (lower_fence, upper_fence)
