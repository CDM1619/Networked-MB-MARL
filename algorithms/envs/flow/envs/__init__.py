"""Contains all callable environments in Flow."""
from algorithms.envs.flow.envs.base import Env
from algorithms.envs.flow.envs.bay_bridge import BayBridgeEnv
from algorithms.envs.flow.envs.bottleneck import BottleneckAccelEnv, BottleneckEnv, \
    BottleneckDesiredVelocityEnv
from algorithms.envs.flow.envs.traffic_light_grid import TrafficLightGridEnv, \
    TrafficLightGridPOEnv, TrafficLightGridTestEnv, TrafficLightGridBenchmarkEnv
from algorithms.envs.flow.envs.ring.lane_change_accel import LaneChangeAccelEnv, \
    LaneChangeAccelPOEnv
from algorithms.envs.flow.envs.ring.accel import AccelEnv
from algorithms.envs.flow.envs.ring.wave_attenuation import WaveAttenuationEnv, \
    WaveAttenuationPOEnv
from algorithms.envs.flow.envs.merge import MergePOEnv
#from test import TestEnv

# deprecated classes whose names have changed
#from bottleneck_env import BottleNeckAccelEnv
#from bottleneck_env import DesiredVelocityEnv
#from green_wave_env import PO_TrafficLightGridEnv
#from green_wave_env import GreenWaveTestEnv


__all__ = [
    'Env',
    'AccelEnv',
    'LaneChangeAccelEnv',
    'LaneChangeAccelPOEnv',
    'TrafficLightGridTestEnv',
    'MergePOEnv',
    'BottleneckEnv',
    'BottleneckAccelEnv',
    'WaveAttenuationEnv',
    'WaveAttenuationPOEnv',
    'TrafficLightGridEnv',
    'TrafficLightGridPOEnv',
    'TrafficLightGridBenchmarkEnv',
    'BottleneckDesiredVelocityEnv',
    'TestEnv',
    'BayBridgeEnv',
    # deprecated classes
    'BottleNeckAccelEnv',
    'DesiredVelocityEnv',
    'PO_TrafficLightGridEnv',
    'GreenWaveTestEnv',
]
