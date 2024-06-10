import torch
import numpy as np
from collections import defaultdict

from flatland.envs.agent_utils import RailAgentStatus

MOVEMENT_ARRAY = ((-1, 0), (0, 1), (1, 0), (0, -1))
#  @njit
def get_new_position(position, movement):
    return (position[0] + MOVEMENT_ARRAY[movement][0], position[1] + MOVEMENT_ARRAY[movement][1])

class Contradictions():
    def __init__(self):
        pass

    def reset(self, env):
        self.env = env
        self._build()


    def _build(self):
        self.rails = set()
        self.essential = defaultdict(set)

        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                transition_bit = bin(self.env.rail.get_full_transitions(*pos))
                total_transitions = transition_bit.count("1")
                if total_transitions > 0:
                    self.rails.add(pos)

        for sh, sw in self.rails:
            for sd in range(4):
                h, w, d = sh, sw, sd
                while True:
                    self.essential[(sh, sw, sd)].add((h, w))
                    cell_transitions = self.env.rail.get_transitions(h, w, d)
                    num_transitions = np.count_nonzero(cell_transitions)
                    if num_transitions != 1: break
                    for nd in range(4):
                        if cell_transitions[nd]:
                            h, w = get_new_position((h, w), nd)
                            d = nd
                            break


        self.bad = defaultdict(set)
        for h, w in self.rails:
            for d in range(4):
                for th, tw in self.essential[(h, w, d)]:
                    for td in range(4):
                        if (h, w) in self.essential[(th, tw, td)]:
                            self.bad[(h, w, d)].add((th, tw, td))


    def start_episode(self):
        self.cur_bad = set()
       
    def is_bad(self, handle, action):
        if self.env.agents[handle].status == RailAgentStatus.READY_TO_DEPART:
            pos = self.env.agents[handle].initial_position
            d = self.env.agents[handle].initial_direction
        else:
            pos = self.env.agents[handle].position
            d = self.env.agents[handle].initial_direction
        approx_pos, approx_dir = self.env.get_env_actions_new_pos(pos, self.env.agents[handle].direction, action)
        if (approx_pos[0], approx_pos[1], approx_dir) in self.cur_bad:
            return True
        return False

    def add_elem(self, handle, action):
        if self.env.agents[handle].status == RailAgentStatus.READY_TO_DEPART:
            pos = self.env.agents[handle].initial_position
        else:
            pos = self.env.agents[handle].position
        approx_pos, approx_dir = self.env.get_env_actions_new_pos(pos, self.env.agents[handle].direction, action)
        for e in self.bad[(approx_pos[0], approx_pos[1], approx_dir)]:
            self.cur_bad.add(e)
    
