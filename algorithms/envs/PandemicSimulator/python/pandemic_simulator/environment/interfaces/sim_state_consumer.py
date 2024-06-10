# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from abc import ABC, abstractmethod
from typing import Any

from .regulation import PandemicRegulation
from .sim_state import PandemicSimState

__all__ = ['SimStateConsumer']


class SimStateConsumer(ABC):

    @abstractmethod
    def consume_begin(self, sim_state: PandemicSimState) -> None:
        """Begin the consumer. For example, the start of each episode"""
        pass

    @abstractmethod
    def consume_state(self, sim_state: PandemicSimState, regulation: PandemicRegulation) -> None:
        pass

    @abstractmethod
    def finalize(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs any final operations (if needed). For example, finalize and return a metric.

        :param args: Any args for finalizing the consumer
        :param kwargs: Any kwargs for finalizing the consumer
        :return: Any value returned after finalizing
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset consumer"""

    def close(self) -> None:
        """Performs any closing operation. For example, saves and closes file objects etc."""
        pass
