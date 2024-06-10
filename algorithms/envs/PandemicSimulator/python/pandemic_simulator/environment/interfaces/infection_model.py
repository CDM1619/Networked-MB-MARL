# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ['IndividualInfectionState', 'InfectionModel', 'InfectionSummary', 'Risk', 'sorted_infection_summary']


class InfectionSummary(Enum):
    NONE = 'none (N)'
    INFECTED = 'infected (I)'
    CRITICAL = 'critical (C)'
    RECOVERED = 'recovered (R)'
    DEAD = 'dead (D)'


sorted_infection_summary = sorted(InfectionSummary, key=lambda x: x.value)


class Risk(Enum):
    LOW = 0
    HIGH = 1


@dataclass(frozen=True)
class IndividualInfectionState:
    """State of the infection."""
    summary: InfectionSummary
    spread_probability: float
    exposed_rnb: float = -1.
    is_hospitalized: bool = False
    shows_symptoms: bool = False


class InfectionModel(ABC):
    """Model of the spreading of the infection."""

    @abstractmethod
    def step(self, subject_infection_state: Optional[IndividualInfectionState], subject_age: int,
             subject_risk: Risk, infection_probability: float) -> IndividualInfectionState:
        """
        This method implements the evolution model for the infection.
        :param subject_infection_state: Current state of the infection for the subject. If None, a base state is used.
        :param subject_age: Age of the subject.
        :param subject_risk: Health risk for the subject.
        :param infection_probability: Probability of getting infected.

        :return: New infection state of the subject.
        """
        pass

    @abstractmethod
    def needs_contacts(self, subject_infection_state: Optional[IndividualInfectionState]) -> bool:
        """
        This method returns True if the current state needs contacts to be computed in order to step.
        :param subject_infection_state: Current state of the infection for the subject. If None, a base state is used.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the infection model"""
