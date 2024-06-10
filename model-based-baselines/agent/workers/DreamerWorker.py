from copy import deepcopy
import numpy as np

import ray
import torch
#from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict
import time
from copy import deepcopy as dp
from environments import Env
import random
import os




@ray.remote
class DreamerWorker:

    def __init__(self,idx, env_name, env_config, controller_config):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller()
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config.ENV_TYPE






        self.logger = None
        self.env_name = env_name
        #global wandb
        #import wandb

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0
        else:
            return self.env.agents[handle].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                   and not self.env.obs_builder.deadlock_checker.is_deadlocked(handle)

    def _select_actions(self, state):
        avail_actions = []
        observations = []
        fakes = []
        cacc_state = self.env.get_state()
        #if self.env_type == Env.FLATLAND:
            #nn_mask = (1. - torch.eye(self.env.n_agents)).bool()
        #else:
            #nn_mask = None
        
        nn_mask = None

        for handle in range(self.env.n_agents):
            if self.env_type == Env.FLATLAND:
                for opp_handle in self.env.obs_builder.encountered[handle]:
                    if opp_handle != -1:
                        nn_mask[handle, opp_handle] = False
            elif self.env_type == Env.STARCRAFT:
                avail_actions.append(torch.tensor(self.env.get_avail_agent_actions(handle)))

            #if self._check_handle(handle) and handle in state:
                #fakes.append(torch.zeros(1, 1))
                #observations.append(state[handle].unsqueeze(0))
            if self.done[handle] == 1:
                fakes.append(torch.ones(1, 1))
                observations.append(self.get_absorbing_state())
            else:
                fakes.append(torch.zeros(1, 1))
                obs = torch.tensor(cacc_state[handle]).float().unsqueeze(0)
                observations.append(obs)
        if self.env_type != Env.STARCRAFT:
            for i in range(self.env.n_agents):
                avail_actions.append(torch.tensor([1]*self.env.n_actions))
        observations = torch.cat(observations).unsqueeze(0)
        av_action = torch.stack(avail_actions).unsqueeze(0) if len(avail_actions) > 0 else None
        nn_mask = nn_mask.unsqueeze(0).repeat(8, 1, 1) if nn_mask is not None else None
        actions = self.controller.step(observations, av_action, nn_mask)
        return actions, observations, torch.cat(fakes).unsqueeze(0), av_action

    def _wrap(self, d):
        for key, value in d.items():
            d[key] = torch.tensor(value).float()
        return d

    def get_absorbing_state(self):
        state = torch.zeros(1, self.in_dim)
        return state

    def augment(self, data, inverse=False):
        aug = []
        default = list(data.values())[0].reshape(1, -1)
        for handle in range(self.env.n_agents):
            if handle in data.keys():
                aug.append(data[handle].reshape(1, -1))
            else:
                aug.append(torch.ones_like(default) if inverse else torch.zeros_like(default))
        return torch.cat(aug).unsqueeze(0)

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT:
            return "episode_limit" not in info
        else:
            return steps_done < self.env.max_time_steps

    def run(self, dreamer_params):
        self.controller.receive_params(dreamer_params)
        state = self.env.reset() # list, len: n_ags, shape: (s_dim,)
        #print("state=",state)
        # self.controller.actor.init_hidden(len(state)) # batch_size=n_ags*ep_l
        # self.controller.actor.hidden = torch.zeros(1, len(state), self.controller.actor.rnn_out_dim)
        state = self._wrap(state)
        steps_done = 0
        self.done = defaultdict(lambda: False)
        rewards = []
        S = []
        while True:
        
            steps_done += 1
            actions, obs, fakes, av_actions = self._select_actions(state)
            
            #print("actions=",actions)
            #print("obs=",obs)
            #print("fakes=",fakes)
            #print("av_actions=",av_actions)

            #print("actions=",actions.dtype)
            #print("obs=",obs.dtype)
            #print("fakes=",fakes.dtype)
            #print("av_actions=",av_actions.dtype)

            next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions)])

            
            s = dp(np.array(list(next_state.values())))
            S.append(s.ravel())
            
            #print("next_state",next_state)
            print("reward",list(reward.values()))
            #print("done",done)
            #print("info",info)

            #print("next_state",next_state[0].dtype)
            #print("reward",reward[0].dtype)
            #print("done",done[0].dtype)
            #print("info",info)
            
            

            
            
            # reward一个dict，{0: rew, 1: rew, ..., N: rew}
            #rewards.append(list(reward.values())[0]) # NOTE 星际争霸共享reward
            rewards.append(sum(list(reward.values()))) # NOTE 星际争霸共享reward
            next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
            self.done = done
            self.controller.update_buffer({"action": actions,
                                           "observation": obs,
                                           "reward": self.augment(reward),
                                           "done": self.augment(done),
                                           "fake": fakes,
                                           "avail_action": av_actions})

            state = next_state
            if self.env_name == "catchup":
                if steps_done==599:
                    if self._check_termination(info, steps_done):
                        obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                        actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                        index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                        actions.scatter_(2, index.unsqueeze(-1), 1.)
                        items = {"observation": obs,
                                 "action": actions,
                                 "reward": torch.zeros(1, self.env.n_agents, 1),
                                 "fake": torch.ones(1, self.env.n_agents, 1),
                                 "done": torch.ones(1, self.env.n_agents, 1),
                                 "avail_action": torch.ones_like(actions)}
                        self.controller.update_buffer(items)
                        self.controller.update_buffer(items)
                    break
            else:
                if all([done[key] == 1 for key in range(self.env.n_agents)]):
                    if self._check_termination(info, steps_done):
                        obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                        actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                        index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                        actions.scatter_(2, index.unsqueeze(-1), 1.)
                        items = {"observation": obs,
                                 "action": actions,
                                 "reward": torch.zeros(1, self.env.n_agents, 1),
                                 "fake": torch.ones(1, self.env.n_agents, 1),
                                 "done": torch.ones(1, self.env.n_agents, 1),
                                 "avail_action": torch.ones_like(actions)}
                        self.controller.update_buffer(items)
                        self.controller.update_buffer(items)
                    break


        #self.env_name = "slowdown"
        #self.algo_name = "MAG"
        #if not os.path.exists('/home/chengdong/MARL/result/{}'.format(self.env_name)):
            #os.makedirs('/home/chengdong/MARL/result/{}'.format(self.env_name))
        #L = 10000
        #S = np.asarray(S)
        #np.savetxt('/home/chengdong/MARL/result/'+self.env_name+'/'+self.env_name+'_'+self.algo_name+str(int(time.time()*1000)%65536)+'.csv', S, delimiter=",")
        #S=S.tolist()
        #print("save successfully!")


        if self.env_type == Env.FLATLAND:
            reward = sum(
                [1 for agent in self.env.agents if agent.status == RailAgentStatus.DONE_REMOVED]) / self.env.n_agents
        else:
            reward = sum(list(reward.values()))
            #reward = 
            aver_step_reward = np.mean(rewards)



            #wandb.log({'episode_reward': aver_step_reward/8})

        return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "win_flag": reward,
                                                   "steps_done": steps_done,
                                                   "aver_step_reward": aver_step_reward}
