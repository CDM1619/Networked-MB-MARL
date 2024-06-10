from flatland.envs.agent_utils import RailAgentStatus

class NoStopShaper():
    def __init__(self, on_switch_value, other_value):
        self.on_switch_value = on_switch_value
        self.other_value = other_value

    def reset(self, env):
        self.switches = set()
        for h in range(env.height):
            for w in range(env.width):
                pos = (h, w)
                transition_bit = bin(env.rail.get_full_transitions(*pos))
                total_transitions = transition_bit.count("1")
                if total_transitions > 2:
                    self.switches.add(pos)

    def __call__(self, env, observations, action_dict, rewards, dones):
        for handle in rewards.keys():
            if env.agents[handle].status == RailAgentStatus.ACTIVE \
                    and handle in action_dict and action_dict[handle] == 2: # action is stop
                if env.agents[handle].position in self.switches:
                    rewards[handle] += self.on_switch_value
                else:
                    rewards[handle] += self.other_value

        return rewards

