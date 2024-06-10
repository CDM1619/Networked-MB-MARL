import numpy as np
import itertools
from flatland.envs.agent_utils import RailAgentStatus

from env.flatland.Flatland import get_new_position

class NetworkLoadAgentLauncher():
    def __init__(self, window_size_generator):
        self.window_size_generator = window_size_generator

    def reset(self, env):
        self.env = env
        self.window_size = self.window_size_generator(env)

        self.ready_to_depart = [0] * len(self.env.agents)
        self.send_more = self.window_size
        self.load = np.zeros(len(env.agents), dtype=np.int)

        self.pairwise_load = _calc_pairwise_load(self.env)


    def update(self):
        for handle in range(len(self.env.agents)):
            if (self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED \
                    or self.env.obs_builder.deadlock_checker.is_deadlocked(handle))\
                and self.ready_to_depart[handle] == 1:

                self._finish_agent(handle)

        while self.send_more > 0:
            best_handle = -1
            for handle, agent in enumerate(self.env.agents):
                if self.ready_to_depart[handle] == 0 and \
                        (best_handle == -1 or self.load[best_handle] > self.load[handle]):
                    best_handle = handle

            if best_handle == -1:
                self.send_more -= 1000000
                break

            self._start_agent(best_handle)

    def is_ready(self, handle):
        return self.ready_to_depart[handle] != 0

    def _finish_agent(self, handle):
        self.send_more += 1
        self.ready_to_depart[handle] = 2
        self.load -= self.pairwise_load[handle]

    def _start_agent(self, handle):
        self.send_more -= 1
        self.ready_to_depart[handle] = 1
        self.load += self.pairwise_load[handle]


# can be optimized, i guess
def _calc_pairwise_load(env):
    n_agents = len(env.agents)
    pairwise_load = np.zeros((n_agents, n_agents), dtype=np.int)
    paths = [_build_shortest_path(env, handle) for handle in range(n_agents)]

    for i, j in itertools.combinations(range(n_agents), 2):
        pairwise_load[i, j] = pairwise_load[j, i] = len(paths[i] & paths[j])

    return pairwise_load



def _build_shortest_path(env, handle):
    agent = env.agents[handle]
    pos = agent.initial_position
    dir = agent.initial_direction

    dist_min_to_target = env.obs_builder.rail_graph.dist_to_target(handle, pos[0], pos[1], dir)

    path = set()
    while dist_min_to_target:
        path.add(pos)
        possible_transitions = env.rail.get_transitions(*pos, dir)
        for new_dir in range(4):
            if possible_transitions[new_dir]:
                new_pos = get_new_position(pos, new_dir)
                new_min_dist = env.obs_builder.rail_graph.dist_to_target(handle, new_pos[0], new_pos[1], new_dir)
                if new_min_dist + 1 == dist_min_to_target:
                    dist_min_to_target = new_min_dist
                    pos, dir = new_pos, new_dir
                    break

    return path 
