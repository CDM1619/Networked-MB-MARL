# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Type

from .pandemic_types import Default

__all__ = ['PandemicRegulation']

from .infection_model import Risk


@dataclass(frozen=True)
class PandemicRegulation:
    """Regulation passed to the Pandemic simulator that modifies the location rules and person behaviors."""

    location_type_to_rule_kwargs: Optional[Dict[Type, Dict[str, Any]]] = None
    """A mapping that broadcasts a generic set of rules to the locations of the same type."""

    business_type_to_rule_kwargs: Optional[Dict[Type, Dict[str, Any]]] = None
    """A mapping that broadcasts a generic set of rules to the locations of the same business
    (Essential / Non-essential) type."""

    social_distancing: Union[float, Default, None] = None
    """A value in [0, 1] that determines how people should interact amongst each other at all locations.
    1 - zero contacts, 0 - interact normally. """

    quarantine: bool = False
    """A bool to tell all persons to quarantine themselves at home."""

    quarantine_if_contact_positive: bool = False
    """A bool to tell all persons to quarantine themselves at home if a contact is positive.
    Works only if contact tracing is enabled."""

    quarantine_if_household_quarantined: bool = False
    """A bool to tell all persons to quarantine themselves at home if a household is quarantined."""

    stay_home_if_sick: bool = False
    """A bool to tell all persons to stay home if sick."""

    practice_good_hygiene: bool = False
    """A bool to tell all persons to practice good hygiene."""

    wear_facial_coverings: bool = False
    """A bool to tell all persons wear facial coverings."""

    risk_to_avoid_gathering_size: Dict[Risk, int] = field(default_factory=lambda: defaultdict(lambda: -1))
    """A mapping of the peron's risk with the gathering size (or more) that they should avoid. -1 means no limit"""

    risk_to_avoid_location_types: Optional[Dict[Risk, List[type]]] = None
    """A mapping of the peron's risk with the locations that they should avoid."""

    stage: int = 0
    """The discrete severity stage of the regulation. 0 - normal, severity increases with the value."""
