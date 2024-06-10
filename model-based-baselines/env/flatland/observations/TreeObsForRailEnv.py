# copied from flatland for optimizations
# from numba import jit, njit, int64, int32, jit_module
import collections
from typing import Optional, List, Dict

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus

from env.flatland.Flatland import get_new_position

def fast_argmax(x):
    if x[0]:
        return 0
    if x[1]:
        return 1
    if x[2]:
        return 2
    return 3


class TreeObsForRailEnv(ObservationBuilder):
    Node = collections.namedtuple('Node', ('dist_other_agent_encountered',
                                          'dist_to_unusable_switch',
                                          'dist_to_next_branch',
                                          'dist_min_to_target',
                                          'num_agents_same_direction',
                                          'num_agents_opposite_direction',
                                          'num_agents_malfunctioning',
                                          'max_index_oppposite_direction',
                                          'max_handle_agent_not_opposite',
                                          'has_deadlocked_agent',
                                          'first_agent_handle',
                                          'first_agent_not_opposite',
                                          'childs'))

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, max_depth: int):
        super().__init__()
        self.max_depth = max_depth
        self.graph = Graph()
        self.cached_nodes = dict()

    def reset(self):
        self.location_with_agent = -np.ones((self.env.height, self.env.width), dtype=np.int)
        self.graph.reset(self.env)
        self.prev_step_positions = list()

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        if handles is None:
            handles = []

        for pos in self.prev_step_positions:
            self.location_with_agent[pos] = -1
        self.prev_step_positions.clear()
        for handle, _agent in enumerate(self.env.agents):
            if _agent.status == RailAgentStatus.ACTIVE:
                self.location_with_agent[_agent.position] = handle
                self.prev_step_positions.append(_agent.position)

        self.cached_nodes.clear()
        observation = super().get_many(handles)

        return observation

    def get(self, handle: int = 0) -> Node:
        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Root node
        # Here information about the agent itself is stored
        root_node_observation = TreeObsForRailEnv.Node(dist_other_agent_encountered=0,
                                                       dist_to_unusable_switch=0,
                                                       dist_to_next_branch=0,
                                                       dist_min_to_target=self.rail_graph.dist_to_target(handle, *agent_virtual_position, agent.direction),
                                                       num_agents_same_direction=0, num_agents_opposite_direction=0,
                                                       num_agents_malfunctioning=agent.malfunction_data['malfunction'],
                                                       max_index_oppposite_direction=0, max_handle_agent_not_opposite=0,
                                                       has_deadlocked_agent=0,
                                                       first_agent_handle=0,
                                                       first_agent_not_opposite=0,
                                                       childs=[None]*4)

        # Child nodes
        orientation = agent.direction

        if num_transitions == 1:  # Whaat? TODO
            orientation = np.argmax(possible_transitions)

        for i, branch_direction in enumerate([(orientation + 4 + i) % 4 for i in range(-1, 3)]):
            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_virtual_position, branch_direction)

                branch_observation = self._explore_branch(handle, new_cell, branch_direction, 1)
                root_node_observation.childs[i] = branch_observation

        return root_node_observation

    def _explore_branch(self, handle, position, direction, depth, was_target=False):
        start_position = (position[0], position[1], direction)
        if start_position in self.cached_nodes:
            node, position, direction = self.cached_nodes[start_position]
        else:
            node, position, direction = self._explore_line(position, direction)
            self.cached_nodes[start_position] = node, position, direction

        last_is_target = (position == self.env.agents[handle].target)

        if was_target:
            dist_min_to_target = -1
        elif last_is_target:
            dist_min_to_target = 0
        else:
            dist_min_to_target = self.rail_graph.dist_to_target(handle, position[0], position[1], direction)

        node = node._replace(dist_min_to_target=dist_min_to_target, childs=[None]*4) # copy returned here!

        if depth == self.max_depth:
            return node
        if node.has_deadlocked_agent: # thee end
            return node

        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions

        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation = self._explore_branch(handle, new_cell, branch_direction,
                                                                    depth + 1, was_target=last_is_target or was_target)
                node.childs[i] = branch_observation

        return node


    def _explore_line(self, position, direction):
        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops # TODO Whaat? Treat dead-ends not as nodes
        tot_dist = 1
        other_agent_encountered = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        num_agents_malfunctioning = 0
        max_index_oppposite_direction = 0
        has_deadlocked_agent = 0
        first_agent_handle = -1
        first_agent_not_opposite = True
        max_handle_agent_not_opposite = True
        max_handle = -1
        while True:
            if self.location_with_agent[position] != -1:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                other_handle = self.location_with_agent[position]

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.env.agents[other_handle].malfunction_data['malfunction'] > 0:
                    num_agents_malfunctioning += 1

                if self.deadlock_checker.is_deadlocked(other_handle): # hack SimpleObservation has deadlock_checker
                    has_deadlocked_agent = 1
                elif first_agent_handle == -1:
                    first_agent_handle = other_handle
                    first_agent_not_opposite = (self.env.agents[first_agent_handle].direction == direction)

                if self.env.agents[other_handle].direction == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1
                    if other_handle > max_handle:
                        max_handle = other_handle
                        max_handle_agent_not_opposite = True
                else:
                    # If no agent in the same direction was found all agents in that position are other direction
                    other_agent_opposite_direction += 1
                    max_index_oppposite_direction = max(max_index_oppposite_direction, other_handle)
                    if other_handle > max_handle:
                        max_handle = other_handle
                        max_handle_agent_not_opposite = False

            if self.greedy_checker.location_has_target[position]:
                break

            only_direction = self.graph.get_if_one_transition(position, direction)
            if only_direction != -1:
                position = get_new_position(position, only_direction)
                direction = only_direction
                tot_dist += 1
                continue

            # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            num_transitions = np.count_nonzero(cell_transitions)
            total_transitions = self.graph.get_total_transitions(position)

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                if total_transitions == 1:
                    # Dead-end!
                    break

                assert False
            elif num_transitions > 0:
                # Switch detected
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                break


        node = TreeObsForRailEnv.Node(dist_other_agent_encountered=other_agent_encountered,
                                      dist_to_unusable_switch=unusable_switch,
                                      dist_to_next_branch=tot_dist,
                                      dist_min_to_target=None,
                                      num_agents_same_direction=other_agent_same_direction,
                                      num_agents_opposite_direction=other_agent_opposite_direction,
                                      num_agents_malfunctioning=num_agents_malfunctioning,
                                      max_index_oppposite_direction=max_index_oppposite_direction,
                                      max_handle_agent_not_opposite=max_handle_agent_not_opposite,
                                      has_deadlocked_agent=has_deadlocked_agent,
                                      first_agent_handle=first_agent_handle,
                                      first_agent_not_opposite=first_agent_not_opposite,
                                      childs=[None]*4)

        return node, position, direction

class Graph():
    def __init__(self):
        self.env = None

    def reset(self, env):
        self.env = env
        self.one_transition = -np.ones((self.env.height, self.env.width, 4), dtype=np.int)
        self.total_transitions = np.empty((self.env.height, self.env.width))

        for h in range(env.height):
            for w in range(env.width):
                for d in range(4):
                    cell_transitions = self.env.rail.get_transitions(h, w, d)
                    num_transitions = np.count_nonzero(cell_transitions)
                    transition_bit = bin(self.env.rail.get_full_transitions(h, w))
                    total_transitions = transition_bit.count("1")
                    if int(transition_bit, 2) == int('1000010000100001', 2):
                        # Treat the crossing as a straight rail cell
                        total_transitions = 2

                    self.total_transitions[(h, w)] = total_transitions
                    if num_transitions == 1 and total_transitions != 1:
                        direction = fast_argmax(cell_transitions)
                        self.one_transition[(h, w, d)] = direction


    def get_if_one_transition(self, pos, d):
        return self.one_transition[pos[0], pos[1], d]

    def get_total_transitions(self, pos):
        return self.total_transitions[pos]
