import sys
from copy import deepcopy
from pathlib import Path
from more_itertools import sample

import numpy as np
import torch
import socket
import setproctitle
import itertools

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout, m_r_perdictor_loss, get_model_loss_for_m_r_training
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import MADDPGCritic
from networks.dreamer.dense import DenseModel


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:

    def __init__(self, config):
        self.config = config
        self.model = [DreamerModel(config).to(config.DEVICE).eval() for _ in range(config.n_nets)]
        if config.obs_as_pol_in:
            self.actor = Actor(config, config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
        else:
            self.actor = Actor(config, config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
        
        self.critic = MADDPGCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        for model in self.model:
            initialize_weights(model, mode='xavier')
        initialize_weights(self.actor)
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)

        self.replay_buffer = DreamerMemory(config, config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        self.elite_idxs = np.arange(config.n_elites)

        if config.use_MPCmodel:
            # self.o_a_feat = DenseModel(config.FEAT + config.ACTION_SIZE, config.HIDDEN, 2, config.HIDDEN) # NOTE 暂时多智能体共享网络
            self.m_r_predictor = DenseModel(config.FEAT, 1, 4, config.HIDDEN).to(config.DEVICE)
            # params = [self.m_r_predictor.parameters()]
            # params.append(self.o_a_feat.parameters())
            self.m_r_optim = torch.optim.Adam(self.m_r_predictor.parameters(), lr=5e-4)
        else:
            self.m_r_predictor = None

        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)
        global wandb
        import wandb
        # wandb.init(dir=config.LOG_FOLDER)
        # config.seed = torch.randint(0, 10000, (1,)).item()
        # wandb.init(config=config,
        #             project='Multi-Agent Ensemble',
        #             entity='zarzard',
        #             notes=socket.gethostname(),
        #             name=str('mamba') + "_" +
        #                 str(config.seed),
        #             group=config.env_name,
        #             dir=config.LOG_FOLDER,
        #             job_type="training",
        #             reinit=True)
        # setproctitle.setproctitle(
        # str(config.env_name) + "@" + str(config.seed))

    def init_optimizers(self):
        self.model_optimizer = [torch.optim.Adam(model.parameters(), lr=self.config.MODEL_LR) for model in self.model]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

    def params(self):
        return {'elite_idxs': self.elite_idxs,
                'model': [{k: v.cpu() for k, v in model.state_dict().items()} for model in self.model],
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        
        
        #print("1111111111=",len(rollout['observation']))
        #print("222222222222222=",len(rollout['action']))
        #print("33333333333333=",len(rollout['reward']))
        #print("444444444444444444=",len(rollout['done']))
        #print("555555555555555555=",len(rollout['fake']))
        #print("66666666666666666666=",len(rollout['last']))
        #print("7777777777777777777=",len(rollout['avail_action']))

        
        
        
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        # ------------------------------ train model ------------------------------
        # if len(self.replay_buffer)/self.config.SEQ_LENGTH < self.config.MODEL_BATCH_SIZE:
        #     self.B = self.config.MODEL_BATCH_SIZE
        losses = []
        self.replay_buffer.init_sampled_idx()
        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            loss = self.train_model(samples)
            losses.append(loss)
        losses = np.stack(losses).mean(0) # (n_nets,)
        self.elite_idxs = np.argsort(losses)[:self.config.n_elites]
        # ------------------------train m_r_predictor--------------------------------
        if self.config.use_MPCmodel:
            # samples = self.replay_buffer.sample_all() # (T, B, n_ags, _dim)
            assert len(self.model) == 1
            # losses_per_step = []
            for epoch in range(self.config.m_r_predictor_epochs):
                samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
                with torch.no_grad():
                    _, loss_per_step = get_model_loss_for_m_r_training(self.config, self.model[0], samples['observation'], samples['action'], samples['av_action'],
                    samples['reward'], samples['done'], samples['fake'], samples['last']) # (T-1,)
                # losses_per_step.append(loss_per_step)
            # losses_per_step = torch.cat(losses_per_step, dim=1) # (T, epoch*epoch_batch_size (40*60), n_ags, _dim)
                self.train_m_r_predictor(samples, loss_per_step)

        # else:
        #     if (len(self.replay_buffer)/self.config.SEQ_LENGTH) % self.config.MODEL_BATCH_SIZE == 0 and \
        #         len(self.replay_buffer) / (self.config.SEQ_LENGTH) <= self.config.max_MODEL_BATCH_SIZE:
        #         # 增大样本规模到buffer数据量，最多翻6倍，即B: MODEL_BATCH_SIZE -> max_MODEL_BATCH_SIZE
        #         self.B = int(len(self.replay_buffer) / self.config.SEQ_LENGTH)
        #     self.prep_model_training()
        #     samples = self.replay_buffer.sample(self.B, repeat=False) # repeat的含义：整条轨迹完全相同才算repeat，一部分重叠不算在内
        #     # -----------分出训练集和测试集-----------
        #     holdout_samples = {}
        #     num_holdouts = int(self.config.holdout_ratio * self.B)
        #     for key in samples.keys():
        #         holdout_samples[key] = samples[key][:, :num_holdouts] # 由于sample时就已经随机打乱了，所以这里直接截断
        #         samples[key] = samples[key][:, num_holdouts:]
        #     # -----------外循环，到holdout_loss下降到一定程度才结束-----------
        #     for epoch in itertools.count():
        #         # 打乱训练集轨迹之间顺序
        #         for key in samples.keys():
        #             samples[key] = samples[key][:, np.random.permutation(samples[key].shape[1])]
        #         # -----------内循环，把训练集的轨迹分成多个mini_batch，每次对一个mini_batch求梯度-----------
        #         for start_idx in range(0, self.B-num_holdouts, self.config.mini_model_batch_size):
        #             mini_sample = {}
        #             for key in samples.keys():
        #                 mini_sample[key] = samples[key][:, start_idx: start_idx + self.config.mini_model_batch_size]
        #             self.train_model(mini_sample)
        #         # -----------验证集loss-----------
        #         with torch.no_grad():
        #             holdout_loss = self.train_model(holdout_samples, validation=True)
        #             break_train = self._save_best(epoch, holdout_loss)
        #             if break_train:
        #                 print('number of model epochs: {}, holdout_loss: {}'.format(epoch, holdout_loss))
        #                 break

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE) # 初始
            self.train_agent(samples)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        if self.config.n_nets == 1:
            holdout_losses = [holdout_losses]
        for i in range(int(self.config.n_nets)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True
        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def prep_model_training(self):
        self._max_epochs_since_update = 4
        self._epochs_since_update = 0
        self._snapshots = {i: (None, 1e10) for i in range(self.config.n_nets)}

    def train_m_r_predictor(self, mini_sample, mini_loss):
        self.m_r_predictor.train()
        assert len(self.model) == 1
        mr_losses = []
        # for epoch in range(self.config.m_r_predictor_epochs):
        #     idx = np.random.randint(0, loss.shape[1], self.config.m_r_predictor_batch_size)
        #     mini_loss = loss[:, idx]
        #     mini_sample = {}
        #     for key in samples.keys():
        #         mini_sample[key] = samples[key][:, idx]
        m_r_loss = m_r_perdictor_loss(self.config, self.model[0], self.m_r_predictor, mini_sample['observation'], 
            mini_sample['action'], mini_sample['av_action'], mini_sample['reward'], mini_sample['done'], mini_sample['fake'], mini_sample['last'], mini_loss)
        self.m_r_optim.zero_grad()
        m_r_loss.backward()
        mr_losses.append(m_r_loss.item())
        # print('epoch: {}, m_r_loss: {}'.format(epoch,  m_r_loss.item()))
        self.m_r_optim.step()
        self.m_r_predictor.eval()
        if self.config.use_wandb:
            wandb.log({'m_r_loss': np.mean(mr_losses)})

    def train_model(self, samples, validation=False):
        losses = []
        for i, model in enumerate(self.model):
            model.train()
            self.model_optimizer[i].zero_grad()
        # samples['observation'].shape: (ep_l, B, n_ags, _dim)
            loss, _ = model_loss(self.config, model, samples['observation'], samples['action'], samples['av_action'],
                          samples['reward'], samples['done'], samples['fake'], samples['last'])
            losses.append(loss.detach().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRAD_CLIP)
            self.model_optimizer[i].step()
            model.eval()
        # self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)            
        return np.array(losses)

    def train_agent(self, samples):
        actions_ens, av_actions_ens, old_policy_ens, imag_feat_ens, returns_ens, imag_obs_ens = [list() for i in range(6)]
        for i in self.elite_idxs:
            # print(samples['action'].shape) # (ep_l-1, roll_B, n_ags, _dim)
            actions, av_actions, old_policy, imag_feat, returns, imag_obs = actor_rollout(samples['observation'],
                                                                            samples['action'],
                                                                            samples['last'], 
                                                                            self.model[i],
                                                                            self.actor,
                                                                            self.critic if self.config.ENV_TYPE != Env.STARCRAFT
                                                                            else self.old_critic,
                                                                            self.config, 
                                                                            self.total_samples,
                                                                            self.m_r_predictor) # old_pol是logits
            # print(actions.shape, imag_feat.shape, imag_obs.shape) # ((roll_len-1)*(ep_l-2)*roll_B, n_ags, _dim)
            actions_ens.append(actions)
            av_actions_ens.append(av_actions)
            old_policy_ens.append(old_policy)
            imag_feat_ens.append(imag_feat)
            returns_ens.append(returns)
            if self.config.obs_as_pol_in:
                imag_obs_ens.append(imag_obs)
        actions, av_actions, old_policy, imag_feat, returns = torch.concat(actions_ens, dim=0), torch.concat(av_actions_ens, dim=0), \
            torch.concat(old_policy_ens, dim=0), torch.concat(imag_feat_ens, dim=0), torch.concat(returns_ens, dim=0)
        # (n_elites*(roll_len-1)*roll_B, n_ags, _dim) where roll_B = (ep_l-2)*roll_episode_B
        if self.config.obs_as_pol_in:
            imag_obs = torch.concat(imag_obs_ens, dim=0)
        adv = returns.detach() - self.critic(imag_feat, actions).detach()
        if self.config.ENV_TYPE != Env.STARCRAFT:
            adv = advantage(adv)
        # if self.config.use_wandb:
        #     wandb.log({'Agent/Returns': returns.mean()})
        for _ in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            # inds = np.arange(actions.shape[0])
            step = 2000
            # self.actor.hidden = torch.zeros(1, int(actions.shape[0]*actions.shape[1]), self.actor.rnn_out_dim, device=actions.device)
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]
                # NOTE
                if self.config.obs_as_pol_in:
                    self.actor.hidden = torch.zeros(1, int(actions.shape[1]), self.actor.rnn_out_dim, device=actions.device)
                    loss = actor_loss(imag_obs[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy, self.config)
                else:
                    loss = actor_loss(imag_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                    old_policy[idx], adv[idx], self.actor, self.entropy, self.config)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING
                val_loss = value_loss(self.critic, actions[idx], imag_feat[idx], returns[idx])
                if self.config.use_wandb and np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
