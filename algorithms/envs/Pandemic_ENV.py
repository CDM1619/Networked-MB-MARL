from tqdm import trange
import random

import sys
#sys.path.append("/home/chengdong/MARL/algorithms/envs/PandemicSimulator")
from algorithms.envs.PandemicSimulator.python import pandemic_simulator as ps


# init globals
ps.init_globals(seed=0)

# select a simulator config
sim_config = ps.sh.small_town_config    #  ['town_config':1w, 'above_medium_town_config':5000, "medium_town_config":2000,  'small_town_config':1000, 'tiny_town_config':500]


Nums_Location = []
for i in range(9):
    Nums_Location.append(sim_config.location_configs[i].num)
nums_Home = sim_config.location_configs[0].num
nums_GroceryStore = sim_config.location_configs[1].num
nums_Office = sim_config.location_configs[2].num
nums_School = sim_config.location_configs[3].num
nums_Hospital = sim_config.location_configs[4].num
nums_RetailStore = sim_config.location_configs[5].num
nums_HairSalon = sim_config.location_configs[6].num
nums_Restaurant = sim_config.location_configs[7].num
nums_Bar = sim_config.location_configs[8].num

# make env
env = ps.env.PandemicGymEnv.from_config(Nums_Location, sim_config, pandemic_regulations=ps.sh.austin_regulations)

def Pandemic():
    return env


def Pandemic_2():
    print("777777777777777777777777777777777777777777777777")
    return env