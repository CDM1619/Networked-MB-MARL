"""Pending deprecation file.

To view the actual content, go to: flow/scenarios/base.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.networks.base import Network


@deprecated('flow.scenarios.base_scenario',
            'flow.networks.base.Network')
class Scenario(Network):
    """See parent class."""

    pass
