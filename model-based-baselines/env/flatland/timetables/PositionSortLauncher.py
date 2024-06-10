from flatland.envs.agent_utils import RailAgentStatus


class PositionSortLauncher():
    def __init__(self):
        pass

    def _no_opposite(self, handle):
        return not _has_opposite(self.env, handle)

    def _by_cities(self):
        self._city_n = [-1] * len(self.env.agents)
        timer = 0
        for handle, agent in enumerate(self.env.agents):
            cur_x, cur_y = agent.initial_position
            for prev in range(handle):
                x, y = self.env.agents[prev].initial_position
                if abs(x - cur_x) < 10 and abs(y - cur_y) < 10:
                    self._city_n[handle] = self._city_n[prev]
            if self._city_n[handle] == -1:
                self._city_n[handle] = timer
                timer += 1

    def reset(self, env):
        self.env = env
        self.order = [i for i in range(len(self.env.agents))]
        self._by_cities()
        self.order.sort(key=lambda handle: self._city_n[handle])

        self.ready_to_depart = [0] * len(self.env.agents)
        self.cur_pos = 0
        self.send_more = 0
        self.max_city = -1

    def update(self):
        for handle in range(len(self.env.agents)):
            if (self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED \
                    or self.env.obs_builder.deadlock_checker.is_deadlocked(handle))\
                and self.ready_to_depart[handle] == 1:

                self.ready_to_depart[handle] = 2
                self.send_more += 1


        if self.send_more >= -self.max_city * 2 - 2:
            self.max_city += 1

        for pos in range(len(self.env.agents)):
            handle = self.order[pos]

            if self.ready_to_depart[handle] == 0 and self._city_n[handle] <= self.max_city and self._no_opposite(handle):
                self.send_more -= 1
                self.ready_to_depart[handle] = 1


    def is_ready(self, handle):
        return self.ready_to_depart[handle] != 0


# TODO
def _has_opposite(env, handle, depth=1):
    agent = env.agents[handle]
    return False


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

class LinearOnMapSizeGenerator():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, env):
        return int(self.alpha * (env.width + env.height) + self.beta)

