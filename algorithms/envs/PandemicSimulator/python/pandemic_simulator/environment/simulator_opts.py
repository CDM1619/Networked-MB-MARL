# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass

__all__ = ['PandemicSimOpts']


@dataclass(frozen=True)
class PandemicSimOpts:
    """Parameter options passed to the simulator."""

    infection_spread_rate_mean: float = 0.021
    """Mean for the bounded-gaussian infection spread rate distribution"""

    infection_spread_rate_sigma: float = 0.01
    """Std for the bounded-gaussian infection spread rate distribution"""

    spontaneous_testing_rate: float = 0.02
    """Probability of a person getting tested independent of symptoms shown"""

    symp_testing_rate: float = 0.3
    """Probability of symptomatic people getting tested"""

    critical_testing_rate: float = 1.
    """Probability of people in critical condition getting tested"""

    testing_false_positive_rate: float = 0.001  # false positives are much more rare than negatives
    """False positive rate"""

    testing_false_negative_rate: float = 0.01
    """False negative rate"""

    retest_rate: float = 0.033
    """Probability of a previously tested-positive person to get retested again"""

    sim_steps_per_regulation: int = 24
    """Number of simulation steps for each regulation"""

    use_contact_tracer: bool = False
    """Set to true to use contact tracer in the simulator"""

    contact_tracer_history_size: int = 5
    """Contact tracer history size. Only used if use_contact_tracer is True."""

    infection_threshold: int = 10
    """A threshold used by """
