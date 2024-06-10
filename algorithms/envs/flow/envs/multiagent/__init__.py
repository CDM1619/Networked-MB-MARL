"""Empty init file to ensure documentation for multi-agent envs is created."""

from algorithms.envs.flow.envs.multiagent.base import MultiEnv
from algorithms.envs.flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from algorithms.envs.flow.envs.multiagent.ring.wave_attenuation import \
    MultiAgentWaveAttenuationPOEnv
from algorithms.envs.flow.envs.multiagent.ring.accel import AdversarialAccelEnv
from algorithms.envs.flow.envs.multiagent.ring.accel import MultiAgentAccelPOEnv
from algorithms.envs.flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from algorithms.envs.flow.envs.multiagent.highway import MultiAgentHighwayPOEnv
from algorithms.envs.flow.envs.multiagent.merge import MultiAgentMergePOEnv
from algorithms.envs.flow.envs.multiagent.i210 import I210MultiEnv

__all__ = [
    'MultiEnv',
    'AdversarialAccelEnv',
    'MultiWaveAttenuationPOEnv',
    'MultiTrafficLightGridPOEnv',
    'MultiAgentHighwayPOEnv',
    'MultiAgentAccelPOEnv',
    'MultiAgentWaveAttenuationPOEnv',
    'MultiAgentMergePOEnv',
    'I210MultiEnv'
]
