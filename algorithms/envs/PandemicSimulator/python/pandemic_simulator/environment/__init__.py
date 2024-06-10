# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
# flake8: noqa
from typing import Optional

import numpy as np
from structlog import BoundLogger

from .city_registry import *
from .contact_tracing import *
from .done import *
from .infection_model import *
from .interfaces import *
from .job_counselor import *
from .location import *
from .make_population import *
from .pandemic_env import *
from .pandemic_sim import *
from .pandemic_testing_strategies import *
from .person import *
from .reward import *
from .simulator_config import *
from .simulator_opts import *


def init_globals(registry: Optional[Registry] = None,
                 seed: Optional[int] = None,
                 log: Optional[BoundLogger] = None) -> None:
    """
    Initialize globals for the simulator

    :param registry: Registry instance for the environment
    :param seed: numpy random seed
    :param log: optional logger
    :return: None
    """
    globals.registry = registry or CityRegistry()
    globals.numpy_rng = np.random.RandomState(seed)
    if log:
        log.info('Initialized globals for the simulator')
