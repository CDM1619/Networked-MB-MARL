from typing import List
from numpy.lib.arraysetops import isin
import ray
import numpy as np
import torch
from algorithms.utils import parallelEval

@ray.remote
class EnvWrapper(object):
    def __init__(self, env_fn):
        self.instance = env_fn()
    
    def get_state_(self, **data):
        return self.instance.get_state_(**data)

    def step(self, *args, **data):
        return self.instance.step(*args, **data)

    def reset(self, **data):
        return self.instance.reset(**data)
    
    def getAttr(self, name):
        return getattr(self.instance, name)

class VectorizedEnv:
    """Rescale reward is not defined by this implementation.
    If the rewards needs to be rescaled, just use non-vectorized environment."""
    def __init__(self, env_fn, env_args):
        self.envs = []
        self.n_env = env_args.n_env
        self.n_cpu = env_args.n_cpu
        self.n_gpu = env_args.n_gpu
        for i in range(self.n_env):
            env = EnvWrapper.options(num_gpus = self.n_gpu, num_cpus=self.n_cpu).remote(env_fn)
            self.envs.append(env)
        self.eval = parallelEval

        names = [
            'n_s_ls', 'n_a_ls', 'coop_gamma', 'observation_space', 'action_space', 'neighbor_mask', 'distance_mask']
        for name in names:
            result = self.envs[0].getAttr.remote(name)
            result = ray.get(result)
            setattr(self, name, result)
    
    def get_state_(self):
        data = [{} for _ in range(self.n_env)]
        results = self.eval(self.envs, 'get_state_', data)
        if isinstance(results[0], np.ndarray):
            results = np.stack(results, axis=0)
        elif isinstance(results[0], torch.Tensor):
            results = torch.stack(results, dim=0)
        return results
    
    def step(self, actions):
        n = actions.shape[0]
        results = [self.envs[i].step.remote(actions[i]) for i in range(n)]
        results = ray.get(results)
        def stackIfPossible(ls):
            if isinstance(ls, list):
                if isinstance(ls[0], np.ndarray):
                    ls = np.stack(ls, axis=0)
                elif isinstance(ls[0], torch.Tensor):
                    ls = np.stack(ls, dim=0)
            return ls
        s1s, rs, ds, infos = [], [], [], []
        for result in results:
            s1, r, d, info = result
            s1s.append(s1)
            rs.append(r)
            ds.append(d)
            infos.append(info)
        s1s, rs, ds, infos = [stackIfPossible(item) for item in [s1s, rs, ds, infos]]
        return s1s, rs, ds, infos
    
    def reset(self):
        data = [{} for _ in range(self.n_env)]
        results = self.eval(self.envs, 'reset', data)
        return self.get_state_()
