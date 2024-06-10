from collections import defaultdict, deque
import numpy as np

from env.flatland.observations.TreeObsForRailEnv import TreeObsForRailEnv
from flatland.envs.agent_utils import RailAgentStatus

from env.flatland.RailGraph import RailGraph


# TreeObservation to vector of floats
# TODO do not calc obs for ready_to_depart unless they will start moving next turn
class SimpleObservation(TreeObsForRailEnv):
    def __init__(self, max_depth, neighbours_depth, timetable, deadlock_checker,
                 greedy_checker, parallel=False, malfunction_obs=True, eval=False):
        super(SimpleObservation, self).__init__(max_depth=max_depth)
        self.max_depth = max_depth
        self.neighbours_depth = neighbours_depth
        self.n_neighbours = (2 ** (self.neighbours_depth + 1) - 2)
        self.state_sz = (2 ** (max_depth + 1) - 2) * 14 + 9
        self.parallel = parallel
        self.malfunction_obs = malfunction_obs
        self.eval = eval

        self.timetable = timetable
        self.deadlock_checker = deadlock_checker
        self.greedy_checker = greedy_checker
        self.rail_graph = RailGraph()

    def reset(self):
        super().reset()
        self.rail_graph.reset(self.env)
        self.timetable.reset(self.env)
        self.deadlock_checker.reset(self.env)
        self.greedy_checker.reset(self.env, self.deadlock_checker)
        self.encountered = defaultdict(list)

        self.random_handles = np.random.uniform(0, 1, len(self.env.agents))
        self.last_action = -np.ones(len(self.env.agents), dtype=np.int)

    def get_many(self, handles=None, ignore_parallel=False):
        if self.parallel and not ignore_parallel:
            return []
        self.timetable.update()
        self.deadlock_checker.update_deadlocks()
        #  if self.eval: log().check_time()
        #  self.rail_graph.update()
        #  if self.eval: log().check_time("rail_graph_recalc")
        #  self.deadlock_checker._fix_simplest_deps()
        observations = super().get_many(handles)
        return observations

    def get(self, handle):
        if not self.timetable.is_ready(handle):
            return None
        if self.greedy_checker.greedy_position(handle):
            return None
        if self.deadlock_checker.old_deadlock(handle):
            return None
        if not self.malfunction_obs and self.env.agents[handle].malfunction_data["malfunction"] != 0:
            return None

        observation = self._get_internal(handle)

        if self.greedy_checker.greedy_obs(handle, observation):
            return None
        self.deadlock_checker._far_deadlock(handle, observation)

        return observation

    def _get_checks(self, handle):
        if not self.timetable.is_ready(handle):
            return False
        if self.greedy_checker.greedy_position(handle):
            return False
        if self.deadlock_checker.old_deadlock(handle):
            return False
        if not self.malfunction_obs and self.env.agents[handle].malfunction_data["malfunction"] != 0:
            return False

        return True

    def _get_node(self, handle):
        return super().get(handle)

    def _get_internal(self, handle, node=None):
        if node is None:
            node = super().get(handle)
        observation = node

        self.cur_handle = handle
        self.encountered[handle].clear()
        if observation is None:
            self.cur_dist = float('inf')
        else:
            self.cur_dist = observation.dist_min_to_target

        observation_vector = list()
        position = self._get_agent_position(handle)
        # root node is special
        # # add root features here
        observation_vector.extend([
            position[0] / 100.0,
            position[1] / 100.0,
            self.random_handles[handle],
            self.norm_bool(self.deadlock_checker.is_deadlocked(handle)),
            self.norm_bool(self.env.agents[handle].malfunction_data["malfunction"] == 0),
            self.norm_dist(self.cur_dist),
            self.norm_bool(self.greedy_checker.on_switch(handle)),
            self.norm_bool(_is_near_next_decision(observation)),
            self.norm_bool(self.env.agents[handle].status == RailAgentStatus.READY_TO_DEPART)
        ])

        self.traverse(observation, self.max_depth, observation_vector)
        return observation_vector

    def _get_agent_position(self, handle):
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED):
            agent_virtual_position = agent.target
        else:
            assert False

        return agent_virtual_position

    def norm_bool(self, val):
        return 2 * int(val) - 1

    def norm_dist(self, dist):
        if dist == float('inf'):
            return 0.
        return dist / 100. - 10  # less then zero (for inf)

    def get_features(self, node, parent, lvl):
        if self.max_depth - lvl < self.neighbours_depth:
            self.encountered[self.cur_handle].append(node.first_agent_handle)
        return [
            1 if node.dist_min_to_target >= 0 else -1,  # real node  (not after target or after target) 0
            self.norm_dist(node.dist_other_agent_encountered),  # 1
            self.norm_dist(node.dist_to_next_branch),  # 2
            self.norm_dist(node.dist_to_unusable_switch),  # 3
            self.norm_dist(node.dist_min_to_target) - self.norm_dist(parent.dist_min_to_target),
            # delta is better isn't it? # 4
            self.norm_dist(node.dist_min_to_target + node.dist_to_next_branch) - self.norm_dist(
                parent.dist_min_to_target),  # 5
            self.norm_bool(node.num_agents_same_direction >= 1),  # 6
            self.norm_bool(node.num_agents_same_direction >= 2),  # 7
            self.norm_bool(node.num_agents_opposite_direction >= 1),  # 8
            self.norm_bool(node.num_agents_opposite_direction >= 2),  # 9
            self.norm_bool(node.max_index_oppposite_direction > self.cur_handle),  # are we the best? # 10
            self.norm_bool(node.first_agent_not_opposite),  # 11
            self.norm_bool(node.max_handle_agent_not_opposite),  # 12
            self.norm_bool(node.has_deadlocked_agent),  # 13
        ]

    def get_padding_features(self, lvl):
        if self.max_depth - lvl < self.neighbours_depth:
            self.encountered[self.cur_handle].append(-1)
        return [0] * 14

    # in two directions with a lot of trash?
    def traverse(self, node, lvl, observation):

        q = deque()
        q.append((node, lvl))

        while q:
            node, lvl = q.popleft()
            assert lvl > 0

            cnt = 0
            if node is not None and node.childs:
                for value in node.childs:
                    if value:
                        cnt += 1
                        observation.extend(self.get_features(value, node, lvl))
                        if lvl - 1 != 0:
                            q.append((value, lvl - 1))

            assert (cnt <= 2)
            # cnt == 0 in case of target nearby?
            # TODO probably rewriting TreeObs can help
            for i in range(cnt, 2):
                observation.extend(self.get_padding_features(lvl))
                if lvl - 1 != 0:
                    q.append((None, lvl - 1))


# ugly, yes
class ObservationDecoder():
    @staticmethod
    def is_near_next_decision(obs):
        return obs[7] > 0

    @staticmethod
    def is_real(obs, edge_id):
        return obs[9 + 14 * edge_id + 0] != 0

    @staticmethod
    def is_after_target(obs, edge_id):
        return obs[9 + 14 * edge_id + 0] == -1

    @staticmethod
    def dist_to_other_agent(obs, edge_id):
        return (obs[9 + 14 * edge_id + 1] + 10) * 100

    @staticmethod
    def dist_to_next_branch(obs, edge_id):
        return (obs[9 + 14 * edge_id + 2] + 10) * 100

    @staticmethod
    def dist_to_unusable_switch(obs, edge_id):
        return (obs[9 + 14 * edge_id + 3] + 10) * 100

    @staticmethod
    def is_more_than_one_opposite_direction(obs, edge_id):
        return obs[9 + 14 * edge_id + 8] > 0

    @staticmethod
    def is_more_than_two_opposite_direction(obs, edge_id):
        return obs[9 + 14 * edge_id + 9] > 0

    @staticmethod
    def has_deadlock(obs, edge_id):
        return obs[9 + 14 * edge_id + 13] > 0


def _is_near_next_decision(node):
    next_node = None
    if node is None or not node.childs:
        return False

    for value in node.childs:
        if value:
            if next_node is not None:
                return False  # on switch
            next_node = value

    return next_node.dist_to_next_branch == 1 or next_node.dist_to_unusable_switch == 1
