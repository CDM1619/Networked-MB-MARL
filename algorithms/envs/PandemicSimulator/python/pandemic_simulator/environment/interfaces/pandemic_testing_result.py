# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from enum import IntEnum

__all__ = ['PandemicTestResult']


class PandemicTestResult(IntEnum):
    UNTESTED = 0
    NEGATIVE = 1
    POSITIVE = 2
    CRITICAL = 3
    DEAD = 4
