"""
Main function for training and evaluating MARL algorithms in Powernet
"""
from __future__ import print_function, division
import argparse
import configparser
import logging
from torch.utils.tensorboard.writer import SummaryWriter
from envs.Grid_envs import GridEnv
from agents.models import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL
from trainer import (Counter, Trainer, Tester, Evaluator,
                     check_dir, copy_file, find_file,
                     init_dir, init_log, init_test_flag)


def parse_args():
    default_base_dir = './ma2c_cnet_der6'
    default_config_dir = 'configs/config_ma2c_cnet_DER6.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using A2C'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(2000, 2500, 100)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args


def init_agent(env, config, total_step, seed):
    if env.agent == 'ia2c':
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    elif env.agent == 'ia2c_fp':
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_nc':
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_cnet':
        # CommNet, it calculates the mean of all messages instead of encoding them
        return MA2C_CNET(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    elif env.agent == 'ma2c_cu':
        """
        ConseNet: the critic is fully decentralized
        but each takes global observations and performs consensus updates
        """
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_dial':
        """
        the message is generated together with action-value estimation 
        by each DQN agent, then it is encoded and summed with other 
        input signals at the receiver side.
        """
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    else:
        return None


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    seed = config.getint('ENV_CONFIG', 'seed')
    env = GridEnv(config['ENV_CONFIG'], random_seed=seed)
    logging.info('Training: a dim %r, agent dim: %d' % (env.n_a_ls, env.n_agent))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
    model = init_agent(env, config['MODEL_CONFIG'], total_step, torch_seed)
    #model.load(dirs['model'], train_mode=True)

    # disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
    trainer = Trainer(env, model, global_counter, summary_writer, output_path=dirs['data'], model_path=dirs['model'])
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    model.save(dirs['model'], final_step)
    summary_writer.close()


def evaluate_fn(agent_dir, output_dir, seeds):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file 
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = GridEnv(config['ENV_CONFIG'], train_mode=False)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    if model is None:
        return
    model_dir = agent_dir + '/model/'
    if not model.load(model_dir):
        return
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log', 'eva_data/voltage'])
    init_log(dirs['eva_log'])
    output_dir = dirs['eva_data']
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
