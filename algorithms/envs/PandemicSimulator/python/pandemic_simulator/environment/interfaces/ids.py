# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass

__all__ = ['LocationID', 'PersonID']


@dataclass(frozen=True)
class LocationID:
    name: str


@dataclass(frozen=True)
class PersonID:
    name: str
    age: int
