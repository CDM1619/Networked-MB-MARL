import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss
from agent.utils.params import FreezeParameters
from networks.dreamer.rnns import rollout_representation, rollout_policy

# 需要找到每一步(o, a)->loss的样本

def m_r_perdictor_loss(config, model, m_r_predictor, obs, action, av_action, reward, done, fake, last, loss):
    # shape: (T, B, n_ags, _dim); loss.shape: (T-1, B, n_ags, 1)
    time_steps, batch_size, n_agents = obs.shape[:3]
    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)[0]
    # feat = torch.cat([post.stoch, deters], -1) # (T-1, B, n_ags, _dim)
    feat = prior.get_features() # (T-1, B, n_ags, _dim)

    inputs = feat.reshape(-1, n_agents, feat.shape[-1]).detach()
    label = loss.reshape(-1, n_agents, loss.shape[-1])
    # feats = o_a_feat(inputs) # (-1, n_ags, _dim) # .reshape(time_steps - 1, batch_size, n_agents, feats.shape[-1]).sum(-2)
    output = m_r_predictor(inputs)
    return F.smooth_l1_loss(output, label)

def get_model_loss_for_m_r_training(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps, batch_size, n_agents = obs.shape[:3] # T=19

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1) # (T-1, B, n_ags, _dim)
    feat_dec = post.get_features()

    _, i_feat, rec_loss_per_step = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    rec_loss_per_step = rec_loss_per_step.reshape(time_steps - 1, batch_size, n_agents, -1).mean(dim=-1, keepdim=True)

    # print(model.reward_model(feat).shape, reward[1:].shape) # (T-1, B, n_ags, 1)
    _, div_per_step = state_divergence_loss(prior, post, config)
    div_per_step = div_per_step.unsqueeze(-1)

    model_loss_per_step = div_per_step  # TODO 试验一下！！
    if config.rec:
        model_loss_per_step += rec_loss_per_step
    if config.rew:
        rew_loss_per_step = F.smooth_l1_loss(model.reward_model(feat), reward[1:], reduction='none') # (T-1, B, n_ags, 1)
        model_loss_per_step += rew_loss_per_step
    if config.avl:
        _, av_loss_per_step = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
        av_loss_per_step = av_loss_per_step.unsqueeze(-1)
        model_loss_per_step += av_loss_per_step
    if config.pcont:
        _, pcont_loss_per_step = log_prob_loss(model.pcont, feat, (1. - done[1:])) # (T-1, B, n_ags)
        pcont_loss_per_step = pcont_loss_per_step.unsqueeze(-1)
        model_loss_per_step += pcont_loss_per_step
    if config.dis:
        i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)
        dis_loss_per_step = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))[1]
        model_loss_per_step[1:] += dis_loss_per_step.reshape(time_steps-2, batch_size, n_agents, 1)

    return None, model_loss_per_step

def model_loss(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps, batch_size, n_agents = obs.shape[:3] # T=19

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1) # (T-1, B, n_ags, _dim)
    feat_dec = post.get_features()

    reconstruction_loss, i_feat, rec_loss_per_step = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    rec_loss_per_step = rec_loss_per_step.reshape(time_steps - 1, batch_size, n_agents, -1).mean(dim=-1, keepdim=True)

    # print(model.reward_model(feat).shape, reward[1:].shape) # (T-1, B, n_ags, 1)
    rew_loss_per_step = F.smooth_l1_loss(model.reward_model(feat), reward[1:], reduction='none') # (T-1, B, n_ags, 1)
    reward_loss = rew_loss_per_step.mean()

    pcont_loss, pcont_loss_per_step = log_prob_loss(model.pcont, feat, (1. - done[1:])) # (T-1, B, n_ags)


    #print("av_action=",av_action)
    av_action_loss, av_loss_per_step = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    
    
    pcont_loss_per_step, av_loss_per_step = pcont_loss_per_step.unsqueeze(-1), av_loss_per_step.unsqueeze(-1)

    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)
    dis_loss, _ = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))

    div, div_per_step = state_divergence_loss(prior, post, config)
    div_per_step = div_per_step.unsqueeze(-1)

    model_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss
    model_loss_per_step = rec_loss_per_step + rew_loss_per_step + pcont_loss_per_step + div_per_step + av_loss_per_step

    return model_loss, model_loss_per_step.detach()

def get_max_rollout_length(args, env_step):
    rollout_length = (min(max(args.rollout_min_length + (env_step - args.rollout_min_step)
                                / (args.rollout_max_step - args.rollout_min_step) * (args.rollout_max_length - args.rollout_min_length),
                                args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)

def actor_rollout(obs, action, last, model, actor, critic, config, env_step, m_r_predictor):
    n_agents = obs.shape[2]
    max_rollout_length = get_max_rollout_length(config, env_step)
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_policy(m_r_predictor, model.observation_decoder, model.transition, model.av_action, max_rollout_length, actor, post, config)
    
    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), returns.detach(), 
              items["imag_obs"][:-1].detach() if items["imag_obs"] is not None else None]
    return [batch_multi_agent(v, n_agents) for v in output]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(states, actions)
        discount_arr = model.pcont(rew_states).mean
        # if config.use_wandb:
            # wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                    # 'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                    # 'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA)
    return returns


def calculate_reward(model, states, mask=None):
    imag_reward = model.reward_model(states)
    if mask is not None:
        imag_reward *= mask
    return imag_reward


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


def actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight, config):
    if config.obs_as_pol_in:
        imag_states = imag_states.reshape(config.n_elites, config.HORIZON-1, config.BATCH_SIZE, actions.shape[1], imag_states.shape[-1])
        imag_states = imag_states.transpose(1, 0).reshape(config.n_elites*(config.HORIZON-1), -1, imag_states.shape[-1])
    _, new_policy = actor(imag_states)
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -
           F.log_softmax(old_policy, dim=-1).gather(2, actions)).exp()
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage)
    # if np.random.randint(10) == 9:
    #     wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})
    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()


def value_loss(critic, actions, imag_feat, targets):
    value_pred = critic(imag_feat, actions)
    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)
