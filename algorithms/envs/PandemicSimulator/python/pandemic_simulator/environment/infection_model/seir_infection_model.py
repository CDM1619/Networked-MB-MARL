# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from scipy.stats import truncnorm

from ..interfaces import IndividualInfectionState, InfectionModel, InfectionSummary, Risk, globals
from ...utils import required

__all__ = ['SEIRInfectionState', 'SEIRModel', 'SpreadProbabilityParams']


class _SEIRLabel(Enum):
    susceptible = 'susceptible'
    exposed = 'exposed'
    pre_asymp = 'pre_asymp'
    pre_symp = 'pre_symp'
    asymp = 'asymp'
    symp = 'symp'
    needs_hospitalization = 'needs_hospitalization'
    hospitalized = 'hospitalized'
    recovered = 'recovered'
    deceased = 'deceased'


class _AgeLimit(Enum):
    _4 = 4
    _17 = 17
    _49 = 49
    _64 = 64
    _200 = 200


_DEFAULT_HOSP_RATE_SYMP = {
    (_AgeLimit._4, Risk.LOW): 0.0279,
    (_AgeLimit._17, Risk.LOW): 0.0215,
    (_AgeLimit._49, Risk.LOW): 1.3215,
    (_AgeLimit._64, Risk.LOW): 2.8563,
    (_AgeLimit._200, Risk.LOW): 3.3873,
    (_AgeLimit._4, Risk.HIGH): 0.2791,
    (_AgeLimit._17, Risk.HIGH): 0.2146,
    (_AgeLimit._49, Risk.HIGH): 13.2154,
    (_AgeLimit._64, Risk.HIGH): 28.5634,
    (_AgeLimit._200, Risk.HIGH): 33.8733,
}

_DEFAULT_DEATH_RATE_NEEDS_HOSP = {
    (_AgeLimit._4, Risk.LOW): 0.2390,
    (_AgeLimit._17, Risk.LOW): 0.3208,
    (_AgeLimit._49, Risk.LOW): 0.2304,
    (_AgeLimit._64, Risk.LOW): 0.3049,
    (_AgeLimit._200, Risk.LOW): 0.4269,
    (_AgeLimit._4, Risk.HIGH): 0.2390,
    (_AgeLimit._17, Risk.HIGH): 0.3208,
    (_AgeLimit._49, Risk.HIGH): 0.2304,
    (_AgeLimit._64, Risk.HIGH): 0.3049,
    (_AgeLimit._200, Risk.HIGH): 0.4269,
}

_DEFAULT_DEATH_RATE_HOSP = {
    (_AgeLimit._4, Risk.LOW): 0.0390,
    (_AgeLimit._17, Risk.LOW): 0.1208,
    (_AgeLimit._49, Risk.LOW): 0.0304,
    (_AgeLimit._64, Risk.LOW): 0.1049,
    (_AgeLimit._200, Risk.LOW): 0.2269,
    (_AgeLimit._4, Risk.HIGH): 0.0390,
    (_AgeLimit._17, Risk.HIGH): 0.1208,
    (_AgeLimit._49, Risk.HIGH): 0.0304,
    (_AgeLimit._64, Risk.HIGH): 0.1049,
    (_AgeLimit._200, Risk.HIGH): 0.2269,
}


def _get_age_limit_from_age(age: int) -> _AgeLimit:
    value = _AgeLimit._200

    for a in _AgeLimit:
        if age <= a.value:
            value = a
            break

    return value


@dataclass(frozen=True)
class SEIRInfectionState(IndividualInfectionState):
    """State of the infection according to SEIR."""
    label: _SEIRLabel = required()


@dataclass(frozen=True)
class SpreadProbabilityParams:
    """Parameters for individual spread probabilities."""
    mean: float = 0.03
    sigma: float = 0.03


_TransitionProbability = Dict[_SEIRLabel, float]
_ModelDescriptionValue = Dict[Optional[Tuple[_AgeLimit, Risk]], _TransitionProbability]
_ModelDescription = Dict[_SEIRLabel, _ModelDescriptionValue]


class SEIRModel(InfectionModel):
    """Model of the spreading of the infection."""

    _model: _ModelDescription
    _seir_to_summary: Dict[_SEIRLabel, InfectionSummary] = {
        _SEIRLabel.susceptible: InfectionSummary.NONE,
        _SEIRLabel.exposed: InfectionSummary.NONE,
        _SEIRLabel.pre_asymp: InfectionSummary.INFECTED,
        _SEIRLabel.pre_symp: InfectionSummary.INFECTED,
        _SEIRLabel.asymp: InfectionSummary.INFECTED,
        _SEIRLabel.symp: InfectionSummary.INFECTED,
        _SEIRLabel.needs_hospitalization: InfectionSummary.CRITICAL,
        _SEIRLabel.hospitalized: InfectionSummary.CRITICAL,
        _SEIRLabel.recovered: InfectionSummary.RECOVERED,
        _SEIRLabel.deceased: InfectionSummary.DEAD
    }
    _spread_probability: Any
    _numpy_rng: np.random.RandomState
    _pandemic_started_counter: int
    _pandemic_start_limit: int

    def __init__(self,
                 symp_proportion: float = 0.57,
                 exposed_rate: Optional[float] = None,
                 pre_asymp_rate: float = 1. / 2.3,
                 pre_symp_rate: float = 1. / 2.3,
                 recovery_rate_asymp: Optional[float] = None,
                 recovery_rate_symp_non_treated: Optional[float] = None,
                 recovery_rate_needs_hosp: float = 0.0214286,
                 recovery_rate_hosp: Optional[float] = None,
                 hosp_rate_symp: Optional[Dict[Tuple[_AgeLimit, Risk], float]] = None,
                 death_rate_hosp: Optional[Dict[Tuple[_AgeLimit, Risk], float]] = None,
                 death_rate_needs_hosp: Optional[Dict[Tuple[_AgeLimit, Risk], float]] = None,
                 from_symp_to_hosp_rate: float = 0.1695,
                 from_needs_hosp_to_death_rate: float = 0.3,
                 from_hosp_to_death_rate: Optional[float] = None,
                 spread_probability_params: Optional[SpreadProbabilityParams] = None,
                 pandemic_start_limit: int = 5):
        self._numpy_rng = globals.numpy_rng
        assert self._numpy_rng, 'No numpy rng found. Either pass a rng or set the default repo wide rng.'

        exposed_rate = 1. / self._numpy_rng.triangular(1.9, 2.9, 3.9) if exposed_rate is None else exposed_rate
        recovery_rate_asymp = 1. / self._numpy_rng.triangular(3.0, 4.0, 5.0) if recovery_rate_asymp is None else \
            recovery_rate_asymp
        recovery_rate_symp_non_treated = 1. / self._numpy_rng.triangular(3.0, 4.0, 5.0) \
            if recovery_rate_symp_non_treated is None else recovery_rate_symp_non_treated

        recovery_rate_hosp = 1. / self._numpy_rng.triangular(9.4, 10.7, 12.8) if recovery_rate_hosp is None \
            else recovery_rate_hosp
        from_hosp_to_death_rate = 1. / self._numpy_rng.triangular(5.2, 8.1, 10.1) if from_hosp_to_death_rate is None \
            else from_hosp_to_death_rate

        hosp_rate_symp = hosp_rate_symp if hosp_rate_symp else _DEFAULT_HOSP_RATE_SYMP
        hosp_rate_symp = defaultdict(lambda: 1., hosp_rate_symp)

        death_rate_hosp = death_rate_hosp if death_rate_hosp else _DEFAULT_DEATH_RATE_HOSP
        death_rate_hosp = defaultdict(lambda: 1., death_rate_hosp)

        death_rate_needs_hosp = death_rate_needs_hosp if death_rate_needs_hosp else _DEFAULT_DEATH_RATE_NEEDS_HOSP
        death_rate_needs_hosp = defaultdict(lambda: 1., death_rate_needs_hosp)

        symp_transition = {
            (a, r): {
                _SEIRLabel.needs_hospitalization: from_symp_to_hosp_rate * self._get_go_to_hospital_rate(
                    hosp_rate_symp[(a, r)],
                    recovery_rate_symp_non_treated,
                    from_symp_to_hosp_rate),
                _SEIRLabel.recovered: recovery_rate_symp_non_treated * (1. - self._get_go_to_hospital_rate(
                    hosp_rate_symp[(a, r)],
                    recovery_rate_symp_non_treated,
                    from_symp_to_hosp_rate)),
            } for a in _AgeLimit for r in Risk
        }

        needs_hosp_transition = {
            (a, r): {
                _SEIRLabel.recovered: (1. - death_rate_needs_hosp[(a, r)]) * recovery_rate_needs_hosp,
                _SEIRLabel.deceased: death_rate_needs_hosp[(a, r)] * from_needs_hosp_to_death_rate
            } for a in _AgeLimit for r in Risk
        }

        hosp_transition = {
            (a, r): {
                _SEIRLabel.recovered: (1. - death_rate_hosp[(a, r)]) * recovery_rate_hosp,
                _SEIRLabel.deceased: death_rate_hosp[(a, r)] * from_hosp_to_death_rate
            } for a in _AgeLimit for r in Risk
        }

        self._model = {
            _SEIRLabel.exposed: self._create_default(_SEIRLabel.exposed, {
                _SEIRLabel.exposed: 1. - exposed_rate,
                _SEIRLabel.pre_asymp: exposed_rate * (1. - symp_proportion),
                _SEIRLabel.pre_symp: exposed_rate * symp_proportion
            }),
            _SEIRLabel.pre_asymp: self._create_default(_SEIRLabel.pre_asymp, {
                _SEIRLabel.pre_asymp: 1. - pre_asymp_rate,
                _SEIRLabel.asymp: pre_asymp_rate
            }),
            _SEIRLabel.pre_symp: self._create_default(_SEIRLabel.pre_symp, {
                _SEIRLabel.pre_symp: 1. - pre_symp_rate,
                _SEIRLabel.symp: pre_symp_rate
            }),
            _SEIRLabel.asymp: self._create_default(_SEIRLabel.asymp, {
                _SEIRLabel.asymp: 1. - recovery_rate_asymp,
                _SEIRLabel.recovered: recovery_rate_asymp
            }),
            _SEIRLabel.recovered: self._create_default(_SEIRLabel.recovered, {}),
            _SEIRLabel.deceased: self._create_default(_SEIRLabel.deceased, {}),
            _SEIRLabel.symp: {
                (a, r): {
                    _SEIRLabel.symp: 1. - (
                            symp_transition[(a, r)][_SEIRLabel.needs_hospitalization] +
                            symp_transition[(a, r)][_SEIRLabel.recovered]
                    ),
                    _SEIRLabel.needs_hospitalization: symp_transition[(a, r)][_SEIRLabel.needs_hospitalization],
                    _SEIRLabel.recovered: symp_transition[(a, r)][_SEIRLabel.recovered],
                } for a in _AgeLimit for r in Risk
            },
            _SEIRLabel.needs_hospitalization: {
                (a, r): {
                    _SEIRLabel.needs_hospitalization: 1. - (
                            needs_hosp_transition[(a, r)][_SEIRLabel.recovered] +
                            needs_hosp_transition[(a, r)][_SEIRLabel.deceased]
                    ),
                    _SEIRLabel.recovered: needs_hosp_transition[(a, r)][_SEIRLabel.recovered],
                    _SEIRLabel.deceased: needs_hosp_transition[(a, r)][_SEIRLabel.deceased]
                } for a in _AgeLimit for r in Risk
            },
            _SEIRLabel.hospitalized: {
                (a, r): {
                    _SEIRLabel.hospitalized: 1. - (
                            hosp_transition[(a, r)][_SEIRLabel.recovered] +
                            hosp_transition[(a, r)][_SEIRLabel.deceased]
                    ),
                    _SEIRLabel.recovered: hosp_transition[(a, r)][_SEIRLabel.recovered],
                    _SEIRLabel.deceased: hosp_transition[(a, r)][_SEIRLabel.deceased]
                } for a in _AgeLimit for r in Risk
            }
        }

        spp = spread_probability_params or SpreadProbabilityParams()
        self._spread_probability = truncnorm((0. - spp.mean) / spp.sigma,
                                             (1. - spp.mean) / spp.sigma,
                                             loc=spp.mean, scale=spp.sigma)
        self._pandemic_start_limit = pandemic_start_limit
        self._pandemic_started_counter = 0

    def _create_default(self, state: _SEIRLabel, probs: Dict[_SEIRLabel, float]) -> _ModelDescriptionValue:
        return defaultdict(lambda: self._model[state][None], {None: probs})

    def _get_go_to_hospital_rate(self, yhr: float, recovery_rate_symp_non_treated: float,
                                 from_symp_to_hosp_rate: float) -> float:
        yhr = 0.01 * yhr
        dividend = (recovery_rate_symp_non_treated * yhr)
        divisor = (recovery_rate_symp_non_treated - from_symp_to_hosp_rate) * yhr
        divisor = from_symp_to_hosp_rate + divisor
        return dividend / divisor

    def step(self, subject_state: Optional[IndividualInfectionState], subject_age: int,
             subject_risk: Risk, infection_probability: float) -> IndividualInfectionState:
        """
        This method implements the SEIR model for the infection.
        :param subject_state: Current SEIR state for the subject.
        :param subject_age: Age of the subject.
        :param subject_risk: Health risk for the subject.
        :param infection_probability: Probability of getting infected.

        :return: New SEIR state of the subject.
        """
        show_symptoms_states = {_SEIRLabel.symp, _SEIRLabel.hospitalized, _SEIRLabel.needs_hospitalization}
        pandemic_started = self._pandemic_started_counter >= self._pandemic_start_limit
        label = _SEIRLabel.susceptible if pandemic_started else _SEIRLabel.exposed
        self._pandemic_started_counter += 1 if not pandemic_started else 0

        subject_state = cast(SEIRInfectionState, subject_state) if subject_state \
            else SEIRInfectionState(summary=self._seir_to_summary[label],
                                    label=label,
                                    spread_probability=self._spread_probability.rvs(random_state=self._numpy_rng))
        label = subject_state.label
        exposed_rnb = -1.

        if subject_state.label == _SEIRLabel.susceptible:
            rnb = self._numpy_rng.uniform()
            if rnb < infection_probability:
                label = _SEIRLabel.exposed
                exposed_rnb = rnb
        elif subject_state.label == _SEIRLabel.needs_hospitalization and subject_state.is_hospitalized:
            label = _SEIRLabel.hospitalized
        else:
            state_probs = self._model[subject_state.label][(_get_age_limit_from_age(subject_age), subject_risk)]
            if len(state_probs) != 0:
                probs = np.array(list(state_probs.values()))
                assert abs(1. - sum(probs)) < 1e-3, f'Probabilities {probs} do not sum to one'
                label = self._numpy_rng.choice(list(state_probs.keys()), p=probs)

        return SEIRInfectionState(summary=self._seir_to_summary[label],
                                  spread_probability=subject_state.spread_probability,
                                  exposed_rnb=exposed_rnb,
                                  is_hospitalized=subject_state.is_hospitalized,
                                  shows_symptoms=label in show_symptoms_states,
                                  label=label)

    def needs_contacts(self, subject_state: Optional[IndividualInfectionState]) -> bool:
        pandemic_started = self._pandemic_started_counter >= self._pandemic_start_limit
        label = _SEIRLabel.susceptible if pandemic_started else _SEIRLabel.exposed
        self._pandemic_started_counter += 1 if not pandemic_started else 0
        subject_state = subject_state if subject_state else SEIRInfectionState(
            summary=self._seir_to_summary[label], label=label,
            spread_probability=self._spread_probability.rvs())
        return cast(SEIRInfectionState, subject_state).label == _SEIRLabel.exposed

    def reset(self) -> None:
        self._pandemic_started_counter = 0
