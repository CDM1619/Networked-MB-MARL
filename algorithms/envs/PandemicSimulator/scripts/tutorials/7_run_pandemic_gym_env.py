# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange
import random

import sys
sys.path.append("/home/chengdong/MARL/algorithms/envs/PandemicSimulator")
from python import pandemic_simulator as ps



def run_pandemic_gym_env() -> None:
    """Here we execute the gym envrionment wrapped simulator using austin regulations,
    a small town config and default person routines."""

    print('\nA tutorial that runs the OpenAI Gym environment wrapped simulator', flush=True)

    # init globals
    ps.init_globals(seed=0)

    # select a simulator config
    sim_config = ps.sh.town_config          #  ['town_config', 'small_town_config', 'test_config', 'tiny_town_config', 'medium_town_config', 'above_medium_town_config']
    

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

    # setup viz
    viz = ps.viz.GymViz.from_config(sim_config=sim_config)

    # run stage-0 action steps in the environment
    env.reset()
    for _ in trange(10, desc='Simulating day'):
        #action_actual = random.randint(0,4)
        
        action_actual = []
        for i in range(10):
            action_actual.append(random.randint(0,4))
            
        obs, reward, done, aux = env.step(action=action_actual)  # here the action is the discrete regulation stage identifier
        


        
        
        #print("state=",state)

                

        print("reward=",reward)
        print("done=",done)
        #print("aux=",aux)
        viz.record((obs, reward))

    # generate plots
    viz.plot()


if __name__ == '__main__':
    run_pandemic_gym_env()

