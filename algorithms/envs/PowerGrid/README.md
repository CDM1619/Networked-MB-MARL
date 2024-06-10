# PowerNet: Multi-agent Deep ReinforcementLearning for Scalable Powergrid Control

An on-policy, cooperative MARL algorithm for voltage control problem in the isolated Microgrid system by incorporating  a  differentiable,  learning-based communication  protocol,  a  spatial  discount factor, and an action smoothing scheme. 

## Installation
- create an python virtual environment: `conda create -n powernet python=3.6`
- active the virtul environment: `conda activate powernet`
- install pytorch (torch>=1.2.0): `pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
- install the requirements: `pip install -r requirements.txt`

## PGSIM
A python implementation of IEEE-34 bus and a 20-DGs Microgrid system with Numba [5] which is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.  The simulation platform we developed is based on the line and load specifications detailed in [3].

<p align="center">
     <img src="docs/microgrid.png" alt="output_example" width="80%" height="80%">
     <br>Fig.1 System diagrams of the two microgrid simulation platforms: (a) microgrid-6; and (b) microgrid-20.
</p>

## Usage
To run the code, just run it via `python main.py`.  The config files contain the parameters for MARL policies.

## Cite
```
@misc{chen2020powernet,
      title={PowerNet: Multi-agent Deep Reinforcement Learning for Scalable Powergrid Control}, 
      author={Dong Chen and Zhaojian Li and Tianshu Chu and Rui Yao and Feng Qiu and Kaixiang Lin},
      year={2020},
      eprint={2011.12354},
      archivePrefix={arXiv},
      primaryClass={eess.SY}
}
```

## Reference
[1] Bidram, Ali, et al. "Distributed cooperative secondary control of microgrids using feedback linearization." IEEE Transactions on Power Systems 28.3 (2013): 3462-3470.

[2] Bidram, Ali, et al. "Secondary control of microgrids based on distributed cooperative control of multi-agent systems." IET Generation, Transmission & Distribution 7.8 (2013): 822-831.

[3] Mustafa, Aquib, et al. "Detection and Mitigation of Data Manipulation Attacks in AC Microgrids." IEEE Transactions on Smart Grid 11.3 (2019): 2588-2603.

[4] Chu, Tianshu, et al. "Multi-agent deep reinforcement learning for large-scale traffic signal control." IEEE Transactions on Intelligent Transportation Systems 21.3 (2019): 1086-1095.

[5] [Numba: an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.](https://numba.pydata.org/)
