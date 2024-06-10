"""Empty init file to handle deprecations."""

# base scenario class
from algorithms.envs.flow.scenarios.base import Scenario

# custom scenarios
from algorithms.envs.flow.scenarios.bay_bridge import BayBridgeScenario
from algorithms.envs.flow.scenarios.bay_bridge_toll import BayBridgeTollScenario
from algorithms.envs.flow.scenarios.bottleneck import BottleneckScenario
from algorithms.envs.flow.scenarios.figure_eight import FigureEightScenario
from algorithms.envs.flow.scenarios.traffic_light_grid import TrafficLightGridScenario
from algorithms.envs.flow.scenarios.highway import HighwayScenario
from algorithms.envs.flow.scenarios.ring import RingScenario
from algorithms.envs.flow.scenarios.merge import MergeScenario
from algorithms.envs.flow.scenarios.multi_ring import MultiRingScenario
from algorithms.envs.flow.scenarios.minicity import MiniCityScenario
from algorithms.envs.flow.scenarios.highway_ramps import HighwayRampsScenario

# deprecated classes whose names have changed
from algorithms.envs.flow.scenarios.figure_eight import Figure8Scenario
from algorithms.envs.flow.scenarios.loop import LoopScenario
from algorithms.envs.flow.scenarios.grid import SimpleGridScenario
from algorithms.envs.flow.scenarios.multi_loop import MultiLoopScenario


__all__ = [
    "Scenario",
    "BayBridgeScenario",
    "BayBridgeTollScenario",
    "BottleneckScenario",
    "FigureEightScenario",
    "TrafficLightGridScenario",
    "HighwayScenario",
    "RingScenario",
    "MergeScenario",
    "MultiRingScenario",
    "MiniCityScenario",
    "HighwayRampsScenario",
    # deprecated classes
    "Figure8Scenario",
    "LoopScenario",
    "SimpleGridScenario",
    "MultiLoopScenario",
]
