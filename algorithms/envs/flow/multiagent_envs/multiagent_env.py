"""Pending deprecation file.

To view the actual content, go to: flow/envs/multiagent/base.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.envs.multiagent.base import MultiEnv as MAEnv


@deprecated('flow.multiagent_envs.multiagent_env',
            'flow.envs.multiagent.base.MultiEnv')
class MultiEnv(MAEnv):
    """See parent class."""

    pass
