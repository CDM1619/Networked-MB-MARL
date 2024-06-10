# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple, Type

from .ids import LocationID, PersonID
from .infection_model import InfectionSummary
from .location import LocationSummary
from .location_states import LocationState
from .pandemic_testing import GlobalTestingState
from .person import PersonState
from .sim_time import SimTime

__all__ = ['PandemicSimState']


@dataclass
class PandemicSimState:
    """PandemicSimulator state"""

    id_to_person_state: Dict[PersonID, PersonState]
    """The state of each person in the simulator"""

    id_to_location_state: Dict[LocationID, LocationState]
    """The state of each location in the simulator"""

    location_type_infection_summary: Dict[Type, int]
    """infection summary for location type"""

    global_infection_summary: Dict[InfectionSummary, int]
    """Specifies the number of people with each infection summary"""

    global_testing_state: GlobalTestingState
    """Specifies the number of people with each infection summary after testing"""

    global_location_summary: Mapping[Tuple[str, str], LocationSummary]
    """A mapping that holds summary statistics (usually cumulative) for each location and person type tuple -
    ((Office, Worker), (School, Minor), etc.)"""

    infection_above_threshold: bool
    """A boolean that is set to True if the infection goes above a set threshold. The threshold is set in the pandemic
    sim"""

    regulation_stage: int
    """The last executed regulation stage"""

    sim_time: SimTime
    """Current simulation time"""
