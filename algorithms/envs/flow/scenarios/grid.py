"""Pending deprecation file.

To view the actual content, go to: flow/networks/traffic_light_grid.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.networks.traffic_light_grid import TrafficLightGridNetwork
from algorithms.envs.flow.networks.traffic_light_grid import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.grid',
            'flow.networks.traffic_light_grid.TrafficLightGridNetwork')
class SimpleGridScenario(TrafficLightGridNetwork):
    """See parent class."""

    pass
