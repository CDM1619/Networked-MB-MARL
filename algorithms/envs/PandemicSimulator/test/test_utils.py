# Confidential, Copyright 2021, Sony Corporation of America, All rights reserved.
import itertools

import pytest

import pandemic_simulator as ps


@pytest.mark.parametrize(['x', 'n'],
                         [[i, j] for i, j in itertools.product(range(1, 10), range(1, 10))])
def test_integer_partitions(x: int, n: int) -> None:
    res = ps.utils.integer_partitions(x, n)
    assert sum(res) == x
    assert max(res) in [min(res) + 1, min(res)]
