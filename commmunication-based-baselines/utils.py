# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 01:12:00 2023

@author: 86153
"""


import itertools
import logging
import numpy as np
import time
import os
import pandas as pd
import subprocess
from copy import deepcopy as dp


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'sudo cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, env_name,  algo_name, model, global_counter, summary_writer, output_path=None, logger=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.env_name = env_name
        self.algo_name =algo_name
        if self.env_name == "Pandemic" or self.env_name == "Large_city":
            if self.algo_name == "ConseNet":
                self.agent = "ma2c_cu"
            elif self.algo_name == "FPrint":
                self.agent = "ia2c_fp"
            else:
                self.agent = "ma2c_cu"
        else:
            self.agent = self.env.agent



        self.model = model
        if self.env_name == "slowdown" or self.env_name == "catchup":
            self.n_step = 600
        elif self.env_name == "Grid" or self.env_name == "Monaco":
            self.n_step = 720
        elif self.env_name == "PowerGrid":
            self.n_step = self.env.T
        elif self.env_name == "Pandemic":
            self.n_step = self.env.T
        elif self.env_name == "Large_city":
            self.n_step = self.env.T
        else:       
            self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        #assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        self.env.train_mode = True
        self.logger = logger

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
        else:
            self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)

    def _get_policy(self, ob, done, mode='train'):
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
        else:
            #print("ob=",ob.shape)
            #print("done=",done)
            policy = self.model.forward(ob, done)
        action = []
        for pi in policy:
            if mode == 'train':
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action.append(np.argmax(pi))
        return policy, np.array(action)

    def _get_value(self, ob, done, action):
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
        else:
            self.naction = self.env.get_neighbor_action(action)
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
        return value

    def _log_episode(self, global_step, mean_reward, std_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        self.data.append(log)
        self._add_summary(mean_reward, global_step)
        self.summary_writer.flush()

    def explore(self, prev_ob, prev_done):
        ob = prev_ob

        if self.env_name == "Large_city":
            self.env.clear()
            self.env.reset()
            ob = self.env.get_state_()

        done = prev_done
        self.ep_reward = 0
        L = 0
        for _ in range(self.n_step):
            # pre-decision
            # print("ob=",ob)
            # print("done=",done)
            L+=1
            policy, action = self._get_policy(ob, done)


            # post-decision
            value = self._get_value(ob, done, action)
            # transition
            self.env.update_fingerprint(policy)
            
            #print("action=",action)
            

            
            if self.env_name == "slowdown" or self.env_name == "catchup":
                #print("action=",action)

                next_ob, reward, done, _ = self.env.step(action)
                
                s = dp(np.array(next_ob))
                self.S.append(s.ravel())
                
                done = done[0]
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)

                #print("next_ob=",next_ob)
                #print("self.env.coop_gamma=",self.env.coop_gamma)
                #print("reward=",reward)
                #print("done=",done)
                #print("global_reward=",global_reward)

            elif self.env_name == "Grid" or self.env_name == "Monaco":

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)

                #print("next_ob=",next_ob)
                #print("reward=",reward)
                #print("done=",done)
                #print("global_reward=",global_reward)
 
            elif self.env_name == "Pandemic":
                #print("action=",action)

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)

                #print("next_ob=",next_ob)
                #print("reward=",reward)
                #print("done=",done)
                #print("global_reward=",global_reward)   

            elif self.env_name == "Large_city":
                #print("action=",action)

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)            




            else:

                next_ob, reward, done, global_reward = self.env.step(action)
                #print("reward=",reward)

                #print("global_reward=",global_reward)


            self.logger.log(interaction=None)


            episode_r = reward
            if episode_r.ndim > 1:
                episode_r = episode_r.mean(axis=0)
            
            #print("episode_r=",episode_r)
            self.ep_reward += episode_r


            
            
            
            
            self.episode_rewards.append(global_reward/self.env.n_agents)
            
            global_step = self.global_counter.next()
            self.cur_step += 1
            # collect experience
            if self.agent.startswith('ma2c'):
                self.model.add_transition(ob, self.ps, action, reward, value, done)
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))
            # terminal check must be inside batch loop for CACC env
            if done:
                if self.env_name != "catchup":
                    break
                else:
                    continue
            ob = next_ob
        if done:
            R = np.zeros(self.model.n_agent)
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)
        
        


        if self.env_name == "slowdown" or self.env_name == "catchup":
            self.logger.log(episode_reward=np.array(self.ep_reward).sum()/6, episode=None)
        elif self.env_name == "Grid" or self.env_name == "Monaco":
            self.logger.log(episode_reward=np.array(self.ep_reward).sum(), episode=None)
        elif self.env_name == "PowerGrid":
            self.logger.log(episode_reward=np.array(self.ep_reward).sum()/self.env.T, episode=None)
        elif self.env_name == "Pandemic":
            self.logger.log(episode_reward=np.array(self.ep_reward).sum(), episode=None)
        elif self.env_name == "Large_city":
            self.logger.log(episode_reward=np.array(self.ep_reward).sum(), episode=None)
            
      





        return ob, done, R

    def perform(self, test_ind, gui=False):
        if self.env_name == "Pandemic":
            self.env.reset()
            ob = self.env.get_state_()
        elif self.env_name == "Large_city":
            self.env.clear()
            self.env.reset()
            ob = self.env.get_state_()
        else:
            ob = self.env.reset(gui=gui, test_ind=test_ind)


        rewards = []
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                # in on-policy learning, test policy has to be stochastic
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    # for mission-critic tasks like CACC, we need deterministic policy
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)

            if self.env_name == "slowdown" or self.env_name == "catchup":

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Grid" or self.env_name == "Monaco":

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                global_reward = np.sum(reward)
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Pandemic":

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                done = done.astype(float64)
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)
            elif self.env_name == "Large_city":

                next_ob, reward, done, _ = self.env.step(action)
                done = done[0]
                done = done.astype(float64)
                global_reward = np.sum(reward)
                
                if self.env.coop_gamma <= 0:
                    reward = np.sum(reward)

    
            else:

                next_ob, reward, done, global_reward = self.env.step(action)

            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        self.S = []
        ii = 0
        while not self.global_counter.should_stop():
            # np.random.seed(self.env.seed)
            if self.env_name == "Pandemic":
                self.env.reset()
                ob = self.env.get_state_()
            elif self.env_name == "Large_city":
                if ii != 0:
                    self.env.clear()
                self.env.reset()
                ob = self.env.get_state_()
            else:
                ob = self.env.reset()
            ii+=1
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            self.episode_rewards = []
            
            while True:
                #ob = [np.array(row) for row in ob]

                ob, done, R = self.explore(ob, done)
                
                
                #print("Length=",len(self.S))
                #if len(self.S)>=6000 and len(self.S)<=6300:
                    #if not os.path.exists('/home/chengdong/MARL/result/{}'.format(self.env_name)):
                        #os.makedirs('/home/chengdong/MARL/result/{}'.format(self.env_name))
                    #L = 10000
                    #S = np.asarray(self.S)
                    #np.savetxt('/home/chengdong/MARL/result/'+self.env_name+'/'+self.env_name+'_'+self.algo_name+'.csv', S, delimiter=",")
                    #S=S.tolist()
                    #print("save successfully!")
                    #break

                dt = self.env.T - self.cur_step
                global_step = self.global_counter.cur_step
                
                if self.env_name == "Pandemic":
                    R = [float(x) for x in R]
                
                #print("RRRRRR=",R)
                #print("dtdtdtdtdt=",dt)
                #print("global_step=",global_step)
                
                self.model.backward(R, dt, self.summary_writer, global_step)
                # termination
                if done:
                    self.env.terminate()
                    # pytorch implementation is faster, wait SUMO for 1s
                    time.sleep(1)
                    break
            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not self.env.name.startswith('atsc'):
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
            self._log_episode(global_step, mean_reward, std_reward)

        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, gui=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.gui = gui

    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, gui=self.gui)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()