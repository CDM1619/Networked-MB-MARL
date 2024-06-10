#  from numba import jit, njit, float64, int32

from enum import IntEnum

from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import RailAgentStatus

from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from env.flatland.Contradictions import Contradictions

MOVEMENT_ARRAY = ((-1, 0), (0, 1), (1, 0), (0, -1))
#  @njit
def get_new_position(position, movement):
    return (position[0] + MOVEMENT_ARRAY[movement][0], position[1] + MOVEMENT_ARRAY[movement][1])



class TrainAction(IntEnum):
    NOTHING = 0
    LEFT = 1
    FORWARD = 2
    RIGHT = 3
    STOP = 4

class FlatlandWrapper():
    def __init__(self, env, reward_shaper):
        self.env = env
        self.reward_shaper = reward_shaper
        self.width = self.env.width
        self.height = self.env.height
        self._max_episode_steps = env._max_episode_steps

        #  self.n_cities = self.env.max_num_cities
        self.n_cities = 3 # TODO fix?
        self.n_agents = env.number_of_agents # chekc if works with remote
        self.action_sz = 3
        self.state_sz = self.env.obs_builder.state_sz
        self.steps = 0
        self.total_reward = 0
        self.max_time_steps = int(8 * (self.env.width + self.env.height + self.n_agents / self.n_cities))

        self.agents = DelegatedAttribute(self.env, "agents")
        self.rail = DelegatedAttribute(self.env, "rail")
        self.obs_builder = DelegatedAttribute(self.env, "obs_builder")
        self.distance_map = DelegatedAttribute(self.env, "distance_map")

        self.cur_env = 0
        self.contr = Contradictions()

    def step(self, action_dict):
        transformed_action_dict = dict()
        for handle, value in action_dict.items():
            action = self.transform_action(handle, value)
            if action != -1:
                transformed_action_dict[handle] = action

        action_dict = transformed_action_dict

        obs, reward, done, info = self.env.step(action_dict)
        for key in set(obs.keys()):
            if obs[key] is None:
                del obs[key]

        for handle in set(obs.keys()):
            if self.obs_builder.deadlock_checker.is_deadlocked(handle):
                done[handle] = 1 # the end

        real_reward = sum(reward.values())
        for key, value in reward.items():
            reward[key] = 0 # just lol
        reward = self.reward_shaper(self, obs, action_dict, reward, done)

        self.total_reward += real_reward
        self.steps += 1

        return obs, reward, done, info, real_reward

    def reset(self):
        self.steps, self.total_reward = 0, 0
        obs, info = self.env.reset()
        self.n_agents = len(self.env.agents) # check if works with remote
        for key in set(obs.keys()):
            if obs[key] is None:
                del obs[key]
        self.reward_shaper.reset(self)
        self.contr.reset(self)
        return obs

    def greedy_position(self, handle):
        return False

    def greedy_action(self, handle):
        return None

    def reinit_greedy(self):
        pass

    def render(self):
        self.env.render()

    def get_steps(self):
        return self.steps

    def get_total_reward(self):
        return self.total_reward

    def get_available_actions(self, handle):
        agent = self.env.agents[handle]
        position = agent.position
        direction = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
            direction = agent.initial_direction
        
        transitions = self.env.rail.get_transitions(*position, direction)
        available_actions = []
        for i in range(-1, 2): # 'L', 'F', 'R'
            new_dir = (direction + i + 4) % 4
            if transitions[new_dir]:
                available_actions.append(i + 2);
        return available_actions

    # agent chooses on of three actions:
    #   0 -- move in the leftmost direction
    #   1 -- move in the rightmost direction
    #   2 -- stop moving
    def transform_action(self, handle, action):
        self.env.obs_builder.last_action[handle] = action
        if action == 2:
            if self.env.agents[handle].status == RailAgentStatus.READY_TO_DEPART:
                return -1
            return TrainAction.STOP

        if self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED:
            return -1

        available_actions = self.get_available_actions(handle)

        if len(available_actions) == 0: # DEAD-END, LET'S TURN
            return TrainAction.FORWARD
        if len(available_actions) == 1 and action == 1: # INVALID CHOICE
            return available_actions[0]
        return available_actions[action]


    def get_env_actions_new_pos(self, pos, dir, action):
        if action == 0 or action == 4: return pos, dir

        transitions = self.env.rail.get_transitions(*pos, dir)
        new_dir = (dir + action - 2 + 4) % 4
        if transitions[new_dir]:
            return get_new_position(pos, new_dir), new_dir
        return pos, dir


# Just creates env and env_renderer
class Flatland():
    def __init__(
            self,
            width,
            height,
            n_agents,
            n_cities,
            grid_distribution_of_cities,
            max_rails_between_cities,
            max_rail_in_cities,
            observation_builder,
            random_seed,
            malfunction_rate
        ):

        self.width = width
        self.height = height
        self.number_of_agents = n_agents
        self.max_num_cities = n_cities
        self.obs_builder = observation_builder

        rail_generator = sparse_rail_generator(
            max_num_cities=n_cities,
            seed=random_seed,
            grid_mode=grid_distribution_of_cities,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rail_in_cities,
        )

        speed_ration_map = {1. : 1.}
        schedule_generator = sparse_schedule_generator(speed_ration_map)
        stochastic_data = MalfunctionParameters(malfunction_rate, min_duration=20, max_duration=50)

        self.env = RailEnv(
            width=width, 
            height=height,
            rail_generator=rail_generator,
            schedule_generator=schedule_generator,
            number_of_agents=n_agents,
            obs_builder_object=observation_builder,
            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
            remove_agents_at_target=True,
            random_seed=random_seed
        )
        self._max_episode_steps = self.env._max_episode_steps

        self.env_renderer = RenderTool(
            self.env,
            agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            show_debug=True,
            screen_height=600,
            screen_width=800
        )

        self.agents = DelegatedAttribute(self.env, "agents")
        self.rail = DelegatedAttribute(self.env, "rail")
        self.obs_builder = DelegatedAttribute(self.env, "obs_builder")
        self.distance_map = DelegatedAttribute(self.env, "distance_map")


    def step(self, action_dict):
        return self.env.step(action_dict)

    def reset(self):
        self.env_renderer.reset()
        obs, info = self.env.reset()
        return obs, info

    def render(self):
        self.env_renderer.render_env(show=True, show_observations=False, show_predictions=False)


class DelegatedAttribute:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def __getattr__(self, name):
        return getattr(getattr(self.owner, self.name), name)

    def __get__(self, instance, owner):
        return getattr(self.owner, self.name)

    def __getitem__(self, handle):
        return getattr(self.owner, self.name)[handle]

    def __len__(self):
        return len(getattr(self.owner, self.name))
