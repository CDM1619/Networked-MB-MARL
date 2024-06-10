class SparseRewardShaper():
    def __init__(self):
        pass

    def reset(self, env):
        pass

    def __call__(self, env, observations, action_dict, rewards, dones):
        for handle in rewards.keys():
            if rewards[handle] == -1:
                rewards[handle] = 0
        return rewards
