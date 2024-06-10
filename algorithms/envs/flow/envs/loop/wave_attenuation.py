"""Pending deprecation file.

To view the actual content, go to: flow/envs/ring/wave_attenuation.py
"""
from algorithms.envs.flow.utils.flow_warnings import deprecated
from algorithms.envs.flow.envs.ring.wave_attenuation import WaveAttenuationEnv as WAEnv
from algorithms.envs.flow.envs.ring.wave_attenuation import WaveAttenuationPOEnv as WAPOEnv


@deprecated('flow.envs.loop.wave_attenuation',
            'flow.envs.ring.wave_attenuation.WaveAttenuationEnv')
class WaveAttenuationEnv(WAEnv):
    """See parent class."""

    pass


@deprecated('flow.envs.loop.wave_attenuation',
            'flow.envs.ring.wave_attenuation.WaveAttenuationPOEnv')
class WaveAttenuationPOEnv(WAPOEnv):
    """See parent class."""

    pass
