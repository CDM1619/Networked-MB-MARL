from .Real_Power_net.var_voltage_control.voltage_control_env import VoltageControl
import numpy as np
import yaml



# load env args
with open("./algorithms/envs/Real_Power_net/args/env_args/var_voltage_control.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
net_topology ='case199_3min_final'    # case141_3min_final / case322_3min_final /case199_3min_final
data_path[-1] = net_topology 
env_config_dict["data_path"] = "/".join(data_path)

# set the action range
assert net_topology in ['case141_3min_final', 'case322_3min_final', "case199_3min_final"], f'{net_topology} is not a valid scenario.'
if net_topology == 'case141_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.6
elif net_topology == 'case322_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8
elif net_topology == 'case199_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.6


# define control mode and voltage barrier function
env_config_dict["mode"] = 'decentralised'                   #decentralised    distributed


if net_topology =='case141_3min_final':
    env_config_dict["voltage_barrier_type"] = 'bowl'
if net_topology =='case322_3min_final':
    env_config_dict["voltage_barrier_type"] = 'l2'
if net_topology =='case199_3min_final':
    env_config_dict["voltage_barrier_type"] = 'l2'



env_config_dict["pv_scale"] = 0.5

env = VoltageControl(env_config_dict)


env.reset()

# define envs
def Real_Power_Env():
    return VoltageControl(env_config_dict)


