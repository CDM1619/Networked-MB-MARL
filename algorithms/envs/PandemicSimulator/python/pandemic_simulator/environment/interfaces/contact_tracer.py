# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np
from orderedset import OrderedSet

from .ids import PersonID

__all__ = ['ContactTracer']


class ContactTracer(ABC):
    """An interface for contact tracing apps."""

    @abstractmethod
    def new_time_slot(self) -> None:
        """
        Adds a new time slot to the contact tracing (e.g., a new day, or a new hour, depending on the granularity).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the traces.
        """
        pass

    @abstractmethod
    def add_contacts(self, contacts: OrderedSet) -> None:
        """
        Adds a trace of contacts obtained at a given time.

        :param contacts: Contacts to add.
        """
        pass

    @abstractmethod
    def get_contacts(self, person_id: PersonID) -> Mapping[PersonID, np.ndarray]:
        """
        Get contacts

        :param person_id: Person's id to trace.
        :return: Mapping from each contact's id to a numpy array of floats between 0 and 1, representing
        the contact hours scaled on a day (24 hours). The length of the sequence depends on the number of days for
        data conservation.
        """
        pass
