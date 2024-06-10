from collections import defaultdict
import numpy as np

from flatland.envs.agent_utils import RailAgentStatus

from env.flatland.Flatland import get_new_position
from env.flatland.observations.SimpleObservation import ObservationDecoder

class GreedyChecker():
    def __init__(self):
        pass

    def reset(self, env, deadlock_checker):
        self.env = env
        self.deadlock_checker = deadlock_checker
        self.reinit_greedy()

    def on_decision_cell(self, h, w, d):
        return self.decision_cells[h, w, d]

    def on_switch(self, handle):
        agent = self.env.agents[handle]
        return agent.status == RailAgentStatus.ACTIVE and (*agent.position, agent.direction) in self.switches

    def greedy_position(self, handle):
        agent = self.env.agents[handle]
        return (not agent.status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED, RailAgentStatus.READY_TO_DEPART)) \
                and \
                    (not self.on_decision_cell(*agent.position, agent.direction) or agent.malfunction_data["malfunction"] != 0 \
                            or self.deadlock_checker.is_far_deadlocked(handle)) \
                and \
                    not self.deadlock_checker.is_deadlocked(handle)

    # greedy if
    # prev action was "STOP"
    # agent is near decision
    # there is an opposite agent on the edge
    def greedy_obs(self, handle, observation):
        return False
        return ObservationDecoder.is_near_next_decision(observation) \
                and self.env.obs_builder.last_action[handle] == 2 \
                and ObservationDecoder.is_more_than_one_opposite_direction(observation, 0)

    def reinit_greedy(self):
        self.greedy_way = defaultdict(int)
        rail_env = self.env
        self.location_has_target = set(agent.target for agent in self.env.agents)
        self.switches = set()

        for h in range(rail_env.height):
            for w in range(rail_env.width):
                pos = (h, w)
                transition_bit = bin(self.env.rail.get_full_transitions(*pos))
                total_transitions = transition_bit.count("1")
                if total_transitions > 2:
                    self.switches.add(pos)

        self.target_neighbors = set()
        self.switches_neighbors = set()

        for h in range(rail_env.height):
            for w in range(rail_env.width):
                pos = (h, w)
                for orientation in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, orientation)
                    for ndir in range(4):
                        if possible_transitions[ndir]:
                            nxt = get_new_position(pos, ndir)
                            if nxt in self.location_has_target:
                                self.target_neighbors.add((h, w, orientation))
                            if nxt in self.switches:
                                self.switches_neighbors.add((h, w, orientation))

        self.decision_cells = np.zeros((self.env.height, self.env.width, 4), dtype=np.bool)
        for posdir in self.switches_neighbors.union(self.target_neighbors):
            self.decision_cells[posdir] = 1
        for pos in self.switches.union(self.location_has_target):
            self.decision_cells[pos[0], pos[1], :] = 1

        self.location_has_target_array = np.zeros((self.env.height, self.env.width), dtype=np.bool)
        for pos in self.location_has_target:
            self.location_has_target_array[pos] = 1
        self.location_has_target = self.location_has_target_array
