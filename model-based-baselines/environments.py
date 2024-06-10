from enum import Enum


class Env(str, Enum):
    FLATLAND = "flatland"
    STARCRAFT = "starcraft"
    CACC = "cacc"
    ATSC = "atsc"
    PowerGrid = "powergrid"


class FlatlandType(str, Enum):
    FIVE_AGENTS = "5_agents"
    TEN_AGENTS = "10_agents"
    FIFTEEN_AGENTS = "15_agents"

FLATLAND_OBS_SIZE = 205
FLATLAND_ACTION_SIZE = 3
RANDOM_SEED = 23
ENV = Env.FLATLAND
ENV_NAME = "5_agents"
