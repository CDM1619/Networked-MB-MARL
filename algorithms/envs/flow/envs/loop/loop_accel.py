"""Pending deprecation file.

To view the actual content, go to: flow/envs/ring/accel.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.envs.ring.accel import AccelEnv as AEnv


@deprecated('flow.envs.loop.accel',
            'flow.envs.ring.accel.AccelEnv')
class AccelEnv(AEnv):
    """See parent class."""

    pass
