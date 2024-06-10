# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from abc import ABC
from typing import Any, Optional, Union

import numpy as np

from ...environment import PandemicObservation

__all__ = ['ExperimentDataSaver']


class ExperimentDataSaver(ABC):
    """An interface for an experiment data saver."""

    def begin(self, obs: PandemicObservation) -> None:
        """Begin a saving episode"""
        pass

    def record(self, obs: PandemicObservation, reward: Optional[Union[np.ndarray, float]] = None) -> None:
        """Record data from obs and optionally a reward"""
        pass

    def finalize(self, **kwargs: Any) -> bool:
        """Finalize saving the episode and return True if successful else False"""
        pass

    def close(self) -> None:
        """Perform any closing operations."""
        pass
