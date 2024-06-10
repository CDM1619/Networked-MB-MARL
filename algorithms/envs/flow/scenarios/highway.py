"""Pending deprecation file.

To view the actual content, go to: flow/networks/highway.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.networks.highway import HighwayNetwork
from algorithms.envs.flow.networks.highway import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.highway',
            'flow.networks.highway.HighwayNetwork')
class HighwayScenario(HighwayNetwork):
    """See parent class."""

    pass
