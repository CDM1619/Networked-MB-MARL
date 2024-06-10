"""Pending deprecation file.

To view the actual content, go to: flow/networks/bay_bridge.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.networks.bay_bridge import BayBridgeNetwork


@deprecated('flow.scenarios.bay_bridge',
            'flow.networks.bay_bridge.BayBridgeNetwork')
class BayBridgeScenario(BayBridgeNetwork):
    """See parent class."""

    pass
