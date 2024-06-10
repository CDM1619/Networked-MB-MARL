# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from .pandemic_testing_result import PandemicTestResult
from .infection_model import InfectionSummary
from .person import PersonState

__all__ = ['PandemicTesting', 'GlobalTestingState']


@dataclass
class GlobalTestingState:
    summary: Dict[InfectionSummary, int]
    num_tests: int


class PandemicTesting(ABC):
    """An interface for pandemic testing."""

    @abstractmethod
    def admit_person(self, person_state: PersonState) -> bool:
        """
        Return a bool whether to admit a person for testing.

        :param person_state: Person's state
        :return: bool
        """

    @abstractmethod
    def test_person(self, person_state: PersonState) -> PandemicTestResult:
        """
        Test the given person and return the test result
        :param person_state: Person's state
        :return: PandemicTestResult instance
        """
