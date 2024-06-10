# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import cast

import numpy as np

from ..interfaces import PersonState, InfectionSummary, IndividualInfectionState, PandemicTestResult, PandemicTesting, \
    globals

__all__ = ['RandomPandemicTesting']


class RandomPandemicTesting(PandemicTesting):
    """Implements random pandemic testing based on the specified probabilities."""

    _spontaneous_testing_rate: float
    _symp_testing_rate: float
    _critical_testing_rate: float
    _testing_false_positive_rate: float
    _testing_false_negative_rate: float
    _retest_rate: float
    _numpy_rng: np.random.RandomState

    def __init__(self,
                 spontaneous_testing_rate: float = 1.,
                 symp_testing_rate: float = 1.,
                 critical_testing_rate: float = 1.,
                 testing_false_positive_rate: float = 0.01,
                 testing_false_negative_rate: float = 0.01,
                 retest_rate: float = 0.033):
        """
        :param spontaneous_testing_rate: Testing rate for non symptomatic population.
        :param symp_testing_rate: Testing rate for symptomatic population.
        :param critical_testing_rate: Testing rate for critical population.
        :param testing_false_negative_rate: False negative rate of testing
        :param testing_false_positive_rate: False positive rate of testing
        :param retest_rate: Rate to retest a peron
        """
        self._spontaneous_testing_rate = spontaneous_testing_rate
        self._symp_testing_rate = symp_testing_rate
        self._critical_testing_rate = critical_testing_rate
        self._testing_false_positive_rate = testing_false_positive_rate
        self._testing_false_negative_rate = testing_false_negative_rate
        self._retest_rate = retest_rate
        self._numpy_rng = globals.numpy_rng

    def admit_person(self, person_state: PersonState) -> bool:
        infection_state = cast(IndividualInfectionState, person_state.infection_state)

        if person_state.test_result == PandemicTestResult.DEAD:
            # A person is not tested if he/she is dead
            return False

        elif infection_state.summary == InfectionSummary.DEAD:
            return True

        rnd = self._numpy_rng.uniform()
        test_person = (
                # if the person is in a hospital, then retest deterministically
                infection_state.is_hospitalized or

                # if the person was tested before, then retest based on retest-probability (independent of symptoms)
                (person_state.test_result in {PandemicTestResult.CRITICAL,
                                              PandemicTestResult.POSITIVE} and rnd < self._retest_rate) or

                # if the person shows symptoms, then test based on critical/symptomatic-probability
                (infection_state.shows_symptoms and (
                        (infection_state.summary == InfectionSummary.CRITICAL and rnd < self._critical_testing_rate) or
                        (infection_state.summary != InfectionSummary.CRITICAL and rnd < self._symp_testing_rate))) or

                # if the person does not show symptoms, then test based on spontaneous-probability
                (not infection_state.shows_symptoms and rnd < self._spontaneous_testing_rate)
        )
        return test_person

    def test_person(self, person_state: PersonState) -> PandemicTestResult:
        positive_states = {InfectionSummary.INFECTED, InfectionSummary.CRITICAL}
        infection_state = cast(IndividualInfectionState, person_state.infection_state)

        if infection_state.summary == InfectionSummary.DEAD:
            return PandemicTestResult.DEAD

        test_outcome = infection_state.summary in positive_states

        # account for testing uncertainty
        rnd = self._numpy_rng.uniform()
        if test_outcome and rnd < self._testing_false_negative_rate:
            test_outcome = False
        elif not test_outcome and rnd < self._testing_false_positive_rate:
            test_outcome = True

        critical = infection_state.summary == InfectionSummary.CRITICAL
        test_result = (PandemicTestResult.CRITICAL if test_outcome and critical
                       else PandemicTestResult.POSITIVE if test_outcome else PandemicTestResult.NEGATIVE)

        return test_result
