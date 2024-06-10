import numpy as np
from collections import defaultdict
from flatland.envs.agent_utils import RailAgentStatus

from env.flatland.Flatland import get_new_position
from env.flatland.observations.SimpleObservation import ObservationDecoder

class DeadlockChecker():
    def __init__(self):
        pass

    def reset(self, env):
        self._is_deadlocked = np.zeros(len(env.agents))
        self._is_far_deadlocked = np.zeros(len(env.agents))
        self._old_deadlock = np.zeros(len(env.agents))
        self.env = env

        self.agent_positions = defaultdict(lambda: -1)
        self.far_dep = defaultdict(list)
        self.simple_dep = dict()

    def is_deadlocked(self, handle):
        return self._is_deadlocked[handle]

    def is_far_deadlocked(self, handle):
        return self._is_far_deadlocked[handle]

    def old_deadlock(self, handle):
        return self._old_deadlock[handle]

    def _check_blocked(self, handle):
        agent = self.env.agents[handle]
        transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        self.checked[handle] = 1

        for direction, transition in enumerate(transitions):
            if transition == 0:
                continue # no road
            
            new_position = get_new_position(agent.position, direction)
            handle_opp_agent = self.agent_positions[new_position]
            if handle_opp_agent == -1:
                self.checked[handle] = 2
                return False # road is free

            if self._is_deadlocked[handle_opp_agent]:
                continue # road is blocked

            if self.checked[handle_opp_agent] == 0:
                self._check_blocked(handle_opp_agent)

            if self.checked[handle_opp_agent] == 2 and not self._is_deadlocked[handle_opp_agent]:
                self.checked[handle] = 2
                return False # road may become free

            self.dep[handle].append(handle_opp_agent)

            continue # road is blocked. cycle

        if not self.dep[handle]:
            self.checked[handle] = 2
            if len(list(filter(lambda t: t != 0, transitions))) == 0:
                return False # dead-end is not deadlock
            self._is_deadlocked[handle] = 1
            self.env.obs_builder.rail_graph.deadlock_agent(handle)
            return True
        return None # shrug


    def fix_deps(self):
        any_changes = True
        # might be slow, but in practice won't
        while any_changes:
            any_changes = False
            for handle, agent in enumerate(self.env.agents):
                if self.checked[handle] == 1:
                    cnt = 0
                    for opp_handle in self.dep[handle]:
                        if self.checked[opp_handle] == 2:
                            if self._is_deadlocked[opp_handle]:
                                cnt += 1
                            else:
                                self.checked[handle] = 2
                                any_changes = True
                    if cnt == len(self.dep[handle]):
                        self.checked[handle] = 2
                        self._is_deadlocked[handle] = True
                        self.env.obs_builder.rail_graph.deadlock_agent(handle)
                        any_changes = True
        for handle, agent in enumerate(self.env.agents):
            if self.checked[handle] == 1:
                self._is_deadlocked[handle] = True
                self.env.obs_builder.rail_graph.deadlock_agent(handle)
                self.checked[handle] = 2


    def update_deadlocks(self):
        self.agent_positions.clear()
        self.dep = [list() for _ in range(len(self.env.agents))]
        for handle, agent in enumerate(self.env.agents):
            self._old_deadlock[handle] = self._is_deadlocked[handle]
            if agent.status == RailAgentStatus.ACTIVE:
                self.agent_positions[agent.position] = handle
        self.checked = [0] * len(self.env.agents)
        for handle, agent in enumerate(self.env.agents):
            if agent.status == RailAgentStatus.ACTIVE and not self._is_deadlocked[handle] \
                    and not self.checked[handle]:
                self._check_blocked(handle)

        self.fix_deps()

    # rather slow TODO
    def _far_deadlock(self, handle, observation):
        return
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.DONE_REMOVED or self._is_deadlocked[handle] or self._is_far_deadlocked[handle]:
            return

        depth = self.env.obs_builder.max_depth
        far_deadlock = np.zeros(2 ** (depth + 1) - 2, dtype=np.bool)
        for edge_id in range(len(far_deadlock)):
            far_deadlock[edge_id] = (ObservationDecoder.has_deadlock(observation, edge_id) \
                        or not ObservationDecoder.is_real(observation, edge_id)) \
                    and not ObservationDecoder.is_after_target(observation, edge_id)

        for d in range(depth - 1, 0, -1):
            l, r = 2 ** d - 2, 2 ** (d + 1) - 2
            lc, rc = 2 ** (d + 1) - 2, 2 ** (d + 2) - 2

            cfar_deadlock = far_deadlock[lc:rc:2] * far_deadlock[lc+1:rc:2]
            far_deadlock[l:r] = far_deadlock[l:r] + cfar_deadlock

        if far_deadlock[0] and far_deadlock[1]:
            self._is_far_deadlocked[handle] = True
            self._is_deadlocked[handle] = True

        self._simplest_deadlock(handle, observation)

    # deadlock if there are two agents, seeing each other and both not able to choose another way
    def _simplest_deadlock(self, handle, observation):
        if self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED or self._is_deadlocked[handle] or self._is_far_deadlocked[handle]:
            return

        if ObservationDecoder.is_real(observation, 1): return

        opp = self.env.obs_builder.encountered[handle][0]
        if opp != -1:
            self.simple_dep[handle] = opp

    def _fix_simplest_deps(self):
        for handle, opp_handle in self.simple_dep.items():
            if opp_handle in self.simple_dep.keys() \
                    and self.simple_dep[opp_handle] == handle:
                self._is_far_deadlocked[handle] = True
                self._is_deadlocked[handle] = True

        self.simple_dep.clear()
