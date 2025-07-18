Official Implementation of [Efficient and Scalable Reinforcement Learning for Large-scale Network Control](https://www.nature.com/articles/s42256-024-00879-7)

# Algorithms
1. Model-based Decentralized Policy Optimization, Our method
2. DPPO (Decentralized PPO)
3. CPPO (Centralized PPO)
4. IMPO (Independent Model-based Policy Optimization)
5. IC3Net (Individualized Controlled Continuous Communication Model)
6. Model-free baselines (CommNet, NeurComm, DIAL, ConseNet ... For more details and code, please refer to: https://arxiv.org/abs/2004.01339)
7. Model-based baselines (MAG, For more details and code, please refer to: https://ojs.aaai.org/index.php/AAAI/article/view/26241)

Key parameters for our decentralized algorithms:
1. radius_v: communication radius for value function, 1,2,3....
2. radius_pi: communication radius for policy, default 1
3. radius_p: communication radius for environment model, default 1

    
# Environments
1. CACC Catchup
2. CACC Slowdown
3. Ring Attenuation
4. Figure Eight
5. ATSC Grid
6. ATSC Monaco
7. ATSC New York
8. Power-Grid
9. Real Power Net
10. Pandemic Net

# Software requirements
## OS Requirements
Linux: Ubuntu 20.04
Driver Version: 535.154.05   
CUDA Version: 12.2
## Python Dependencies
Python 3.8+

For other dependent packages, please refer to environment.yml

# Environment setup
## First, install sumo
CACC, Flow and ATSC are developed based on Sumo, you need to install the corresponding version of sumo as follows:
1. SUMO installation. Version 1.11.0

The commit number of SUMO, available at https://github.com/eclipse/sumo used to run the results is 2147d155b1.
To install SUMO, you are recommended to refer to https://sumo.dlr.de/docs/Installing/Linux_Build.html to install the specific version via repository checkout. Note that the latest version of SUMO is not compatible with Flow environments.
In brief, after you checkout to that version, run the following command to build the SUMO binaries.
```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
cd <sumo_dir> # please insert the correct directory name here
export SUMO_HOME="$PWD"
mkdir build/cmake-build && cd build/cmake-build
cmake ../..
make -j$(nproc)
```
After building, you need to manually ad the bin folder into your path:
```
export PATH=$PATH:$SUMO_HOME/bin
```

2. Setting up the environment.

It's recommended to set up the environment via Anaconda. The environment specification is in environment.yml.
After installing the required packages, run
```
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```
in terminal to include the SUMO python packages.

## Second, Install conda environment
Due to the numerous environments and algorithms involved in this project, conflicts between packages are inevitable and difficult to perfectly resolve in the same environment. Therefore, we will establish two conda environments to perform different training tasks separately.

1.To run environments and algorithms related to CACC, FLOW, ATSC, and PowerGrid, Environment 1 needs to be configured:
```
conda env create -f environment_1.yml
```

2.To run environments and algorithms related to Pandemic, Real Power-net and baselines, Environment 2 needs to be configured:
```
conda env create -f environment_2.yml
```
and then:
```
conda activate Pandemic_RealPower_baselines
```
and then:
```
cd algorithms/envs/PandemicSimulator
```
and then (Sometimes you may need to run it twice): 
```
python3 -m pip install -e .
```
You need to install both environments 1 and 2 at the same time. If an error occurs while running in one environment, please switch to the other environment and try again, thanks.

## Logging data during training
We uses WandB as logger. 
1. Setting up WandB.

Before running our code, you should log in to WandB locally. Please refer to https://docs.wandb.ai/quickstart for more detail.

## Downloading the Dataset

1. Download the data from the [link](https://drive.google.com/file/d/1-GGPBSolVjX1HseJVblNY3KoTqfblmLh/view?usp=sharing). refer to https://github.com/Future-Power-Networks/MAPDN
2. Unzip the zip file
3. Go to the directory `algorithms/envs/Real_Power_net/var_voltage_control/` and create a folder called `data`, Then create 3 folders:
    * `case141_3min_final`
    * `case322_3min_final`
    * `case199_3min_final`
5. Move data in `case141_3min_final` to folder `algorithms/envs/Real_Power_net/var_voltage_control/data/case141_3min_final`
6. Move data in `case322_3min_final` to folder `algorithms/envs/Real_Power_net/var_voltage_control/data/case322_3min_final`
7. Move data (`load_active.csv`, `load_reactive.csv`, `pv_active.csv`, except for `model.p`) in `case33_3min_final` to folder `algorithms/envs/Real_Power_net/var_voltage_control/data/case199_3min_final`

# Train and evaluate
Train the agent (DPPO, CPPO, IC3Net, Our method) by:

```python
python launcher.py --env ENV --algo ALGO --device DEVICE
```
`ENV` specifies which environment to run in, including `eight`, `ring`, `catchup`, `slowdown`, `Grid`, `Monaco`, `PowerGrid`, `Real_Power`, `Pandemic`,  `Large_city`.

`ALGO` specifies the algorithm to use, including `IC3Net`, `CPPO`, `DPPO`, `DMPO`, `IA2C`.

`DEVICE` specifies the device to use, including `cpu`, `cuda:0`, `cuda:1`, `cuda:2`...

such as:
```python
python launcher.py --env 'slowdown' --algo 'DMPO' --device 'cuda:0'
```

Train the model-free baselines (CommNet, NeurComm, DIAL, ConseNet ...) by:
```
cd commmunication-based-baselines
```
and then open main.py to set the environment and algorithm.
and then:
```python
python main.py train
```

Train the model-based baselines (MAG) by:
```
cd model-based-baselines
```
and then run the train.py, you can set the environment and algorithm, such as:
```python
python train.py --env 'powergrid' --env_name "powergrid"
```

Evaluate the agent (DPPO, CPPO, IC3Net, Our method) by:

After trainging, the actors model will be saved in checkpoints/xxx/Models/xxxbest_actor.pt,
You just need to add following code in algorithms/algo/agent/DPPO.py(DMPO.py/CPPO.py/...):
```python
self.actors.load_state_dict(torch.load(test_actors_model))
```
after initializing actors:
```python
self.collect_pi, self.actors = self._init_actors()
```
where:
```python
test_actors_model = 'checkpoints/standard _xxx/Models/xxxbest_actor.pt'
```
We also provide evaluation code. After replacing the corresponding actor model, please run the following command to evaluate in CACC, such as:
```python
python evaluate_cacc.py --env 'slowdown' --algo 'DPPO'
```
to evaluate in Flow, such as:
```python
python evaluate_flow.py --env 'ring' --algo 'DPPO'
```
to evaluate in ATSC, such as:
```python
python evaluate_atsc.py --env 'Monaco' --algo 'DPPO'
```
to evaluate in Real Power-Net, such as:
```python
python evaluate_real_power.py --env 'Real_Power' --algo 'DPPO'
```
to evaluate in Pandemic Net, such as:
```python
python evaluate_pandemic.py --env 'Pandemic' --algo 'DPPO'
```
to evaluate in Large_city (New York city), such as:
```python
python evaluate_large_city.py --env 'Large_city' --algo 'DPPO'
```
# Switch to different settings in the same environment
1. For Power Grid, two settings are used, one with 20 agents and the other with 40 agents. You need to switch between different settings by opening the file:
```python
./algorithms/envs/PowerGrid/envs/Grid_envs.py
```
and modifying the corresponding code:
```python
DER_num = 20   # 20 or 40
```

2. For Real Power Net, three settings are used, 141 bus, 322 bus and 421 bus (Corresponding to 191 bus in the code). You need to switch between different settings by opening the file:
```python
./algorithms/envs/Real_Power.py
```
and modifying the corresponding code:
```python
net_topology ='case199_3min_final'    # case141_3min_final / case322_3min_final /case199_3min_final
```

3. For Pandemic Net, five settings are used, 500 / 1000 / 2000 / 5000 / 10000 population. You need to switch between different settings by opening the file:
```python
./algorithms/envs/Pandemic_ENV.py
```
and modifying the corresponding code:
```python
sim_config = ps.sh.small_town_config    #  ['town_config':1w, 'above_medium_town_config':5000, "medium_town_config":2000,  'small_town_config':1000, 'tiny_town_config':500]
```


# Reference
Please cite the paper in the following format if you used this code during your research
```python
@article{ma2024efficient,
  title={Efficient and scalable reinforcement learning for large-scale network control},
  author={Ma, Chengdong and Li, Aming and Du, Yali and Dong, Hao and Yang, Yaodong},
  journal={Nature Machine Intelligence},
  pages={1--15},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
