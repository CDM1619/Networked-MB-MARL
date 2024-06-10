"""Pending deprecation file.

To view the actual content, go to: flow/networks/minicity.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.networks.minicity import MiniCityNetwork


@deprecated('flow.scenarios.minicity',
            'flow.networks.minicity.MiniCityNetwork')
class MiniCityScenario(MiniCityNetwork):
    """See parent class."""

    pass
