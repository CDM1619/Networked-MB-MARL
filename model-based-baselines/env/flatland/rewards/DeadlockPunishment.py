class DeadlockPunishment():
    def __init__(self, value):
        self.value = value

    def reset(self, env):
        self.deadlock_checker = env.obs_builder.deadlock_checker
        self.already_punished = [False] * len(env.agents)

    def __call__(self, env, observations, action_dict, rewards, dones):
        for handle in rewards.keys():
            if self.deadlock_checker.is_deadlocked(handle):
                if not self.already_punished[handle]:
                    self.already_punished[handle] = True
                    rewards[handle] += self.value
        return rewards

