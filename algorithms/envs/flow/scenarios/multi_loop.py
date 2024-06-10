"""Pending deprecation file.

To view the actual content, go to: flow/networks/multi_ring.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.networks.multi_ring import MultiRingNetwork
from algorithms.envs.flow.networks.multi_ring import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.multi_loop',
            'flow.networks.multi_ring.MultiRingNetwork')
class MultiLoopScenario(MultiRingNetwork):
    """See parent class."""

    pass
