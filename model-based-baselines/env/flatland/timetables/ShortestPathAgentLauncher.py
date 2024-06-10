from flatland.envs.agent_utils import RailAgentStatus

class ShortestPathAgentLauncher():
    def __init__(self, window_size_generator):
        self.window_size_generator = window_size_generator

    def _get_dist(self, handle):
        agent = self.env.agents[handle]
        assert(agent.status == RailAgentStatus.READY_TO_DEPART)
        position = agent.initial_position
        direction = agent.initial_direction
        dist = self.env.distance_map.get()[handle, position[0], position[1], direction]
        return dist

    def reset(self, env):
        self.env = env
        self.window_size = self.window_size_generator(env)
        self.order = [i for i in range(len(self.env.agents))]
        self.order.sort(key=lambda handle: self._get_dist(handle))

        self.ready_to_depart = [0] * len(self.env.agents)
        self.send_more = self.window_size
        self.cur_pos = 0

    def update(self):
        for handle in range(len(self.env.agents)):
            if (self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED \
                    or self.env.obs_builder.deadlock_checker.is_deadlocked(handle))\
                and self.ready_to_depart[handle] == 1:

                self.ready_to_depart[handle] = 2
                self.send_more += 1

        while self.send_more and self.cur_pos < len(self.order):
            self.send_more -= 1
            self.ready_to_depart[self.order[self.cur_pos]] = 1
            self.cur_pos += 1

    def is_ready(self, handle):
        return self.ready_to_depart[handle] != 0


class ConstWindowSizeGenerator():
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, env):
        return self.window_size

class LinearOnAgentNumberSizeGenerator():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, env):
        return int(self.alpha * len(env.agents) + self.beta)


