import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4_utils import get_new_position

class ShortPathObs(ObservationBuilder):
    def __init__(self):
        super(ShortPathObs, self).__init__()
        self.state_sz = 3

    def reset(self):
        pass

    def get(self, handle: int = 0):
        agent = self.env.agents[handle]


        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
        elif agent.status == RailAgentStatus.DONE_REMOVED:
            return [0, 0, 0]
        else:
            position = agent.position

        possible_transitions = self.env.rail.get_transitions(*position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        if num_transitions == 1:
            observation = [1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])

            observation = [0, 0]
            observation[np.argmin(min_distances)] = 1

        dist = self.env.distance_map.get()[handle, position[0], position[1], agent.direction]
        if dist == float('inf'):
            dist = self.env._max_episode_steps
        dist /= self.env._max_episode_steps
        return [dist] + observation

