"""Pending deprecation file.

To view the actual content, go to: flow/networks/merge.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.networks.merge import MergeNetwork
from algorithms.envs.flow.networks.merge import ADDITIONAL_NET_PARAMS  # noqa: F401


@deprecated('flow.scenarios.merge',
            'flow.networks.merge.MergeNetwork')
class MergeScenario(MergeNetwork):
    """See parent class."""

    pass
