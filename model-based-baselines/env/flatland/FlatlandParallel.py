import ray
import copy

from env.flatland.Flatland import DelegatedAttribute, TrainAction
from flatland.envs.agent_utils import RailAgentStatus
from logger import log, init_logger

NUM_CPUS = 4
ray.init(num_cpus=NUM_CPUS)

@ray.remote
class FlatlandCopy():
    def __init__(self, env, identificator):
        self.env = copy.deepcopy(env)
        self.identificator = identificator
        init_logger("logdir", "tmp{}".format(self.identificator))

    def reset(self, env):
        pass

    def step(self, action_dict):
        self.env.step(action_dict)
        self.env.obs_builder.get_many(None, ignore_parallel=True)
        return True

    def get_nodes(self, handles):
        return [(handle, self.get_node(handle)) for handle in handles]

    def get_node(self, handle):
        return (handle, self.env.obs_builder._get_node(handle), self.identificator)


@ray.remote
def get_obs(obs_builder, handle):
    return (handle, obs_builder._get_node(handle))

class FlatlandParallel():
    def __init__(self, env):
        self.env = env
        self.n_agents = len(self.env.agents)

        self.agents = DelegatedAttribute(self.env, "agents")
        self.rail = DelegatedAttribute(self.env, "rail")
        self.obs_builder = DelegatedAttribute(self.env, "obs_builder")

        self.copies = [FlatlandCopy.remote(self.env, i) for i in range(NUM_CPUS - 1)]
        self.copies_pool = ray.util.ActorPool(self.copies)
        self.step_ids = None

    def reset(self):
        observation = self._get_observations()
        return observation

    def step(self, action_dict):
        self.step_ids = [copy.step.remote(action_dict) for copy in self.copies]

    def _get_observations(self):
        self.env.obs_builder.get_many(None, ignore_parallel=True)
        if self.step_ids:
            ray.get(self.step_ids) # finish step in env copies

        start_time = log().check_time()

        observation_dict = {handle: None for handle in range(self.n_agents)}

        valid_handles = [handle for handle in range(self.n_agents) if self.env.obs_builder._get_checks(handle)]
        obs = self.copies_pool.map_unordered(lambda copy, handle: copy.get_node.remote(handle) ,valid_handles)
        for handle, node, worker_id in obs:
            observation_dict[handle] = self.env.obs_builder._get_internal(handle, node)


        #  queries, cur_handle = list(), 0
        #  for copy in self.copies:
            #  while cur_handle < self.n_agents and not self.env.obs_builder._get_checks(cur_handle):
                #  cur_handle += 1
            #  if cur_handle < self.n_agents:
                #  queries.append(copy.get_node.remote(cur_handle))
                #  cur_handle += 1

        #  while queries:
            #  done_id, queries = ray.wait(queries)
            #  handle, node, worker_handle = ray.get(done_id)[0]
            #  observation_dict[handle] = self.env.obs_builder._get_internal(handle, node)

            #  while cur_handle < self.n_agents and not self.env.obs_builder._get_checks(cur_handle):
                #  cur_handle += 1
            #  if cur_handle < self.n_agents:
                #  queries.append(self.copies[worker_handle].get_node.remote(cur_handle))
                #  cur_handle += 1

        log().prev_time = start_time
        log().check_time("full_observation")
        return observation_dict

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
        if action == 2:
            return TrainAction.STOP

        available_actions = self.get_available_actions(handle)

        if len(available_actions) == 0: # DEAD-END, LET'S TURN
            return TrainAction.FORWARD
        if len(available_actions) == 1 and action == 1: # INVALID CHOICE
            return available_actions[0]
        return available_actions[action]
