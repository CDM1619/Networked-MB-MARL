import ray
import wandb

from agent.workers.DreamerWorker import DreamerWorker


class DreamerServer:
    def __init__(self, logger, env_name, n_workers, env_config, controller_config, model):
        ray.init()

        self.workers = [DreamerWorker.remote(i, env_name, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]

    def append(self, idx, update):
        self.tasks.append(self.workers[idx].run.remote(update))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs


class DreamerRunner:

    def __init__(self, logger, env_name, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.server = DreamerServer(logger, env_name, n_workers, env_config, controller_config, self.learner.params())
        self.config = learner_config
        self.env_name = env_name

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10):
        cur_steps, cur_episode = 0, 0
        win_count = 0
        incre_win_rates = []
        log_interval = 20

        while True:
            rollout, info = self.server.run()
            self.learner.step(rollout)
            cur_steps += info["steps_done"]
            cur_episode += 1
            win_count += info["win_flag"] # win: 1, lose: 0
            if cur_episode % log_interval == 0: # log
                win_rate = win_count / log_interval
                incre_win_rates.append(win_rate)
                if self.config.use_wandb:
                    wandb.log({'incre_win_rate': win_rate, 'total_step': cur_steps})
                print('map: {}, cur_step: {}, incre_win_rate: {}'.format(self.env_name, cur_steps, win_rate))
                win_count = 0
                if self.config.use_wandb:
                    wandb.log({'aver_step_reward': info["aver_step_reward"], 'total_step': cur_steps})
                
            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            self.server.append(info['idx'], self.learner.params())

