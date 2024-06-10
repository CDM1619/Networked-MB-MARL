# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import dataclasses
from dataclasses import dataclass, field
from typing import Sequence, Type, Any, Dict, Optional

from .interfaces import BaseLocation, PersonRoutineAssignment
from .location import Hospital, HospitalState

__all__ = ['LocationConfig', 'PandemicSimConfig']


@dataclass
class LocationConfig:
    location_type: Type[BaseLocation]
    """Location type"""

    num: int
    """Number of locations of the given type"""

    num_assignees: int = -1
    """Number of assignees assigned to that location (used by JobCounselor)"""

    state_opts: Dict[str, Any] = field(default_factory=dict)
    """Additional options passed to the initial state."""

    extra_opts: Dict[str, Any] = field(default_factory=dict)
    """Additional options passed to the location constructor."""

    def __post_init__(self) -> None:
        for k in self.state_opts:
            assert k in [f.name for f in dataclasses.fields(self.location_type.state_type)]


@dataclass
class PandemicSimConfig:
    """Config for setting up the simulator"""

    num_persons: int = 1000
    """Number of persons in the simulator"""

    location_configs: Sequence[LocationConfig] = ()
    """Configs of all locations in the simulator"""

    regulation_compliance_prob: float = 0.99
    """The probability that a person complies to regulation every step"""

    max_hospital_capacity: int = field(init=False, default=-1)
    """Specifies maximum hospital capacity (inferred from a hospital location if there is one)"""

    person_routine_assignment: Optional[PersonRoutineAssignment] = None
    """Person routine assignment instance"""

    def __post_init__(self) -> None:
        for config in self.location_configs:
            if issubclass(config.location_type, Hospital):
                patient_capacity = config.state_opts.get('patient_capacity', HospitalState.patient_capacity)
                self.max_hospital_capacity = config.num * patient_capacity
