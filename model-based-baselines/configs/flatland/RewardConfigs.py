from configs.Config import Config
from env.flatland.rewards import SimpleRewardShaper, SparseRewardShaper, NearRewardShaper, DeadlockPunishment, \
    RewardsComposer, NoStopShaper, FinishRewardShaper


class RewardConfig(Config):
    def __init__(self):
        pass

    def create_reward_shaper(self):
        pass


class SimpleRewardConfig(RewardConfig):
    def __init__(self):
        pass

    def create_reward_shaper(self):
        return SimpleRewardShaper()


class SparseRewardConfig(RewardConfig):
    def __init__(self):
        pass

    def create_reward_shaper(self):
        return SparseRewardShaper()


class NearRewardConfig(RewardConfig):
    def __init__(self, coeff):
        self.coeff = coeff

    def create_reward_shaper(self):
        return NearRewardShaper(self.coeff)


class DeadlockPunishmentConfig(RewardConfig):
    def __init__(self, value):
        self.value = value

    def create_reward_shaper(self):
        return DeadlockPunishment(self.value)


class RewardsComposerConfig(RewardConfig):
    def __init__(self, reward_shaper_configs):
        self.reward_shapers_configs = reward_shaper_configs

    def create_reward_shaper(self):
        reward_shapers = [config.create_reward_shaper() for config in self.reward_shapers_configs]
        return RewardsComposer(reward_shapers)


class NotStopShaperConfig(RewardConfig):
    def __init__(self, on_switch_value, other_value):
        self.on_switch_value = on_switch_value
        self.other_value = other_value

    def create_reward_shaper(self):
        return NoStopShaper(self.on_switch_value, self.other_value)


class FinishRewardConfig(RewardConfig):
    def __init__(self, finish_value):
        self.finish_value = finish_value

    def create_reward_shaper(self):
        return FinishRewardShaper(self.finish_value)
