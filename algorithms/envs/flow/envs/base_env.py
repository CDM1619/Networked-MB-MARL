"""Pending deprecation file.

To view the actual content, go to: flow/envs/base.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.envs.base import Env as BaseEnv


@deprecated('flow.envs.base_env', 'flow.envs.base.Env')
class Env(BaseEnv):
    """See parent class."""

    pass
