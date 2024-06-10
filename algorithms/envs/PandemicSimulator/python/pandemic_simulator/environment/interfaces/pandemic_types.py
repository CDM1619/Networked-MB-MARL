# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import NewType

__all__ = ['Default', 'NoOP', 'DEFAULT', 'NOOP']

# A type for indicating to use the default value (this is useful when None has a semantic meaning).
Default = NewType('Default', int)
DEFAULT = Default(0)

# A type for indicating no operation was performed by person's step.
NoOP = NewType('NoOP', int)
NOOP = NoOP(0)
