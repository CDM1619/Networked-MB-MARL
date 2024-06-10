# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import abc
import dataclasses
from typing import Any, cast, Type, TypeVar, Dict, List

import istype
import numpy as np

__all__ = ['required', 'abstract_class_property', 'checked_cast', 'shallow_asdict', 'cluster_into_random_sized_groups',
           'integer_partitions']

_T = TypeVar('_T')


def required() -> _T:
    def required_err() -> Any:
        raise ValueError('Missing required field')

    return cast(_T, dataclasses.field(default_factory=required_err))


def abstract_class_property() -> _T:
    @abc.abstractmethod  # type: ignore
    def inner() -> Any:
        raise NotImplementedError

    return cast(_T, inner)


def checked_cast(type: Type[_T], obj: Any) -> _T:
    """
    Method for executing a safe cast in python
    """
    assert istype.isinstanceof(obj, type)
    return obj  # type: ignore


def shallow_asdict(x: Any) -> Dict[str, Any]:
    assert dataclasses.is_dataclass(x)
    return {field.name: getattr(x, field.name) for field in dataclasses.fields(x)}


def cluster_into_random_sized_groups(orig_list: List[int],
                                     min_group_size: int,
                                     max_group_size: int,
                                     numpy_rng: np.random.RandomState) -> List[List[int]]:
    final_list = []
    cnt = 0
    while cnt < len(orig_list):
        size = numpy_rng.randint(min_group_size, max_group_size + 1)
        final_list.append(orig_list[cnt: cnt + size])
        cnt += size
    return final_list


def integer_partitions(x: int, n_partitions: int) -> List[int]:
    _x = x // n_partitions
    return [_x + 1 if i < x % n_partitions else _x for i in range(n_partitions)]
