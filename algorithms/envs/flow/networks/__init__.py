"""Contains all available networks in Flow."""

# base network class
from algorithms.envs.flow.networks.base import Network

# custom networks
from algorithms.envs.flow.networks.bay_bridge import BayBridgeNetwork
from algorithms.envs.flow.networks.bay_bridge_toll import BayBridgeTollNetwork
from algorithms.envs.flow.networks.bottleneck import BottleneckNetwork
from algorithms.envs.flow.networks.figure_eight import FigureEightNetwork
from algorithms.envs.flow.networks.traffic_light_grid import TrafficLightGridNetwork
from algorithms.envs.flow.networks.highway import HighwayNetwork
from algorithms.envs.flow.networks.ring import RingNetwork
from algorithms.envs.flow.networks.merge import MergeNetwork
from algorithms.envs.flow.networks.multi_ring import MultiRingNetwork
from algorithms.envs.flow.networks.minicity import MiniCityNetwork
from algorithms.envs.flow.networks.highway_ramps import HighwayRampsNetwork
from algorithms.envs.flow.networks.i210_subnetwork import I210SubNetwork

__all__ = [
    "Network", "BayBridgeNetwork", "BayBridgeTollNetwork",
    "BottleneckNetwork", "FigureEightNetwork", "TrafficLightGridNetwork",
    "HighwayNetwork", "RingNetwork", "MergeNetwork", "MultiRingNetwork",
    "MiniCityNetwork", "HighwayRampsNetwork", "I210SubNetwork"
]
