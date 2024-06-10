class FakeRewardShaper():
    def __init__(self):
        pass

    def reset(self, env):
        pass

    def __call__(self, env, observations, action_dict, rewards, dones):
        return rewards

