# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np

from .interfaces import LocationID, globals, SimTimeTuple, BusinessBaseLocation
from .simulator_config import LocationConfig

__all__ = ['JobCounselor']


@dataclass(frozen=True)
class WorkPackage:
    work: LocationID
    work_time: SimTimeTuple


class JobCounselor:
    """Generates vacant work ids for working persons."""

    _all_work_ids_vacant_pos: List[Tuple[List[LocationID], int]]
    _numpy_rng: np.random.RandomState

    def __init__(self, location_configs: Sequence[LocationConfig]):
        """
        :param location_configs: A sequence of LocationConfigs
        """
        assert globals.registry, 'No registry found. Create the repo wide registry first by calling init_globals()'
        self._registry = globals.registry
        self._numpy_rng = globals.numpy_rng

        self._all_work_ids_vacant_pos = [(list(globals.registry.location_ids_of_type(config.location_type)),
                                          config.num * (config.num_assignees if config.num_assignees != -1 else 1000))
                                         for config in location_configs
                                         if issubclass(config.location_type, BusinessBaseLocation) and
                                         config.num_assignees != 0]

    def next_available_work(self) -> Optional[WorkPackage]:
        """Return next available work"""
        if len(self._all_work_ids_vacant_pos) == 0:
            return None

        work_type_index = self._numpy_rng.randint(0, len(self._all_work_ids_vacant_pos))
        work_ids, vacant_positions = self._all_work_ids_vacant_pos[work_type_index]

        work_id: LocationID = work_ids[self._numpy_rng.randint(0, len(work_ids))]
        vacant_positions -= 1

        if vacant_positions <= 0:
            self._all_work_ids_vacant_pos.pop(work_type_index)
        else:
            self._all_work_ids_vacant_pos[work_type_index] = (work_ids, vacant_positions)
        work_time = self._registry.get_location_work_time(work_id)
        assert work_time
        return WorkPackage(work_id, work_time)
