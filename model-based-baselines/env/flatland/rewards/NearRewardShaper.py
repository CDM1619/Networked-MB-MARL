from collections import defaultdict

from flatland.envs.agent_utils import RailAgentStatus

_NEARINF = 1e3

class NearRewardShaper():
    def __init__(self, coeff):
        self.coeff = coeff

    def reset(self, env):
        self.prev_dist = [0] * env.n_agents
        for handle in range(env.n_agents):
            self.prev_dist[handle] = self._get_new_dist(env, handle)

    def _get_new_dist(self, env, handle):
        agent = env.agents[handle]
        if agent.status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED):
            return 0
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
            direction = agent.initial_direction
        else:
            position = agent.position
            direction = agent.direction
        dist = env.distance_map.get()[handle, position[0], position[1], direction]
        if dist == float('inf'):
            dist = _NEARINF
        return dist

    def __call__(self, env, observations, action_dict, rewards, dones):
        for handle in rewards.keys():
            prev = self.prev_dist[handle]
            nw = self._get_new_dist(env, handle)
            rewards[handle] += (prev - nw) * self.coeff
            self.prev_dist[handle] = nw

        return rewards
