from flatland.envs.agent_utils import RailAgentStatus

class FinishRewardShaper():
    def __init__(self, finish_value):
        self.finish_value = finish_value

    def reset(self, env):
        self.already_rewarded = [False] * len(env.agents)

    def __call__(self, env, observations, action_dict, rewards, dones):
        for handle in rewards.keys():
            if env.agents[handle].status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED) \
                    and not self.already_rewarded[handle]:
                self.already_rewarded[handle] = True
                rewards[handle] += self.finish_value
        return rewards


