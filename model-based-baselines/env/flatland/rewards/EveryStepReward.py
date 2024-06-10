class EveryStepReward():
    def __init__(self, value):
        self.value = value

    def reset(self, env):
        pass

    def __call__(self, env, observations, action_dict, rewards, dones):
        for handle in rewards.keys():
            if not dones[handle]:
                rewards[handle] += self.value
        return rewards
