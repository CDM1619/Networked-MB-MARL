"""Empty init file to ensure documentation for multi-agent envs is created."""

from algorithms.envs.flow.multiagent_envs.multiagent_env import MultiEnv
from algorithms.envs.flow.multiagent_envs.loop.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from algorithms.envs.flow.multiagent_envs.loop.loop_accel import AdversarialAccelEnv
from algorithms.envs.flow.multiagent_envs.traffic_light_grid import MultiTrafficLightGridPOEnv
from algorithms.envs.flow.multiagent_envs.highway import MultiAgentHighwayPOEnv

__all__ = [
    'MultiEnv',
    'AdversarialAccelEnv',
    'MultiWaveAttenuationPOEnv',
    'MultiTrafficLightGridPOEnv',
    'MultiAgentHighwayPOEnv'
]
