import numpy as np
import copy

# Attention, in Multiagent every worker goes throug curriculum
class EnvCurriculum():
    def __init__(self, env_configs, env_episodes):
        self.env_configs = env_configs
        self.env_episodes = env_episodes
        self._pos_env = -1
        self.env = self._start_next()


    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        if name == "update_scores":
            return self.update_scores
        return getattr(self.env, name)


    def _start_next(self):
        if self._pos_env + 1 < len(self.env_configs): # keep launching last env_episode
            self._pos_env += 1
        self._cur_env_episode = 0
        return self.env_configs[self._pos_env].create_env()

    def update_scores(self, gae):
        pass

    def _reset(self):
        if self._cur_env_episode == self.env_episodes[self._pos_env]:
            self.env = self._start_next()
        self._cur_env_episode += 1
        return self.env.reset()


class EnvCurriculumSample():
    def __init__(self, env_configs, env_probs):
        self.envs = [config.create_env() for config in env_configs]
        self.env_probs = np.array(env_probs, dtype=np.float)
        self.env_probs /= np.sum(self.env_probs)

        self.env = self._start_next()

    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        if name == "update_scores":
            return self.update_scores
        return getattr(self.env, name)

    def _start_next(self):
        self._pos_env = np.random.choice(len(self.envs), p=self.env_probs)
        return self.envs[self._pos_env]

    def update_scores(self, gae):
        pass

    def _reset(self):
        self.env = self._start_next()
        return self.env.reset()


class EnvCurriculumPrioritizedSample():
    def __init__(self, env_configs, repeat_random_seed):
        self.repeat_random_seed = repeat_random_seed
        if not repeat_random_seed:
            self.env_configs = env_configs
        else:
            self.env_configs = list()
            for i in range(10):
                self.env_configs += copy.deepcopy(env_configs)
                for config in env_configs:
                    config.update_random_seed

        self.scores = 1e9*np.ones(len(self.env_configs))
        self.cnt = np.zeros(len(self.env_configs), dtype=np.long) + 1e-9

        self.ro = 0.1
        self.beta = 0.1
        self.env = self._start_next()

    def __getattr__(self, name):
        if name == "cur_env":
            return self._pos_env
        if name == "reset":
            return self._reset
        if name == "update_scores":
            return self.update_scores
        return getattr(self.env, name)

    def update_scores(self, gae):
        self.scores[self._pos_env] = gae.mean()

    def _start_next(self):
        i_ranks = np.argsort(-self.scores) # descending_order
        ranks = np.empty_like(i_ranks)
        ranks[i_ranks] = np.arange(len(self.scores)) + 1
        p_score = (1. / ranks) ** (1. / self.beta)
        p_score /= p_score.sum()

        p_cnt = self.cnt.sum() - self.cnt + 1e-15
        p_cnt /= p_cnt.sum()

        probs = (1 - self.ro) * p_score + self.ro * p_cnt

        self._pos_env = np.random.choice(len(self.env_configs), p=probs)
        self.cnt[self._pos_env] += 1

        if not self.repeat_random_seed:
            self.env_configs[self._pos_env].update_random_seed()
        return self.env_configs[self._pos_env].create_env()

    def _reset(self):
        self.env = self._start_next()
        return self.env.reset()
