class RewardsComposer():
    def __init__(self, reward_shapers):
        self.reward_shapers = reward_shapers

    def reset(self, env):
        for shaper in self.reward_shapers:
            shaper.reset(env)

    def __call__(self, env, observations, action_dict, rewards, dones):
        for shaper in self.reward_shapers:
            rewards = shaper(env, observations, action_dict, rewards, dones) # TODO shape inplace in reward shapers
        return rewards
