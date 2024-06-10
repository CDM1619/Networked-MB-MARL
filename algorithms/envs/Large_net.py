#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2021 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
import sys
import optparse
import random
import sumolib
import numpy as np
import gym
from gym.spaces import Box, Discrete
import configparser
import os
import pdb
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
from copy import deepcopy as dp





class Large_city_net(gym.Wrapper):
    def __init__(self, net_path,sim_path):
        self.obj = "queue"     # queue   wait  arrived
        
        self.net_path = net_path
        self.sim_path = sim_path
        self.net = sumolib.net.readNet(self.net_path)
        self.AdjacencyList = self.generateTopology()    #获取邻接矩阵
        #print("self.AdjacencyList=",self.AdjacencyList)
        self.Edges = self.net.getEdges()
        self.VEH_LEN_M = 200
        self.coop_gamma = -1
        self.T = 500
        
        
        
        self.gui = False
        if self.gui:
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.traci = traci
        self.traci.start([self.sumoBinary, "-c", sim_path,"--no-warnings"])
        



        self.n_agent = self.traci.trafficlight.getIDCount()
        self.n_agents = self.n_agent
        if self.n_agent < 500:
            self.single_step_second = 20
        else:
            self.single_step_second = 40
        print("n_agent= ", self.n_agent)     #交通灯的数量
        self.id_list = list(self.traci.trafficlight.getIDList())
        #print("id_list= ", self.id_list)       #交通灯 的id
        #phase = traci.trafficlight.getAllProgramLogics(id_list[119])
        # print("phase=",len(phase[0].phases))    #action 的数量
        
        self.A  = []
        for i in range(self.n_agent):
            phase = self.traci.trafficlight.getAllProgramLogics(self.id_list[i])
            a = len(phase[0].phases)
            self.A.append(a)
            
        self.n_action = max(self.A)
        self.action_space = Discrete(self.n_action)
        
        # print("A=",max(A))       ##action最多的交通灯 = 动作空间的维度
        self.neighbor_mask = self.get_neighbor_matrix()
        self.distance_mask = dp(self.neighbor_mask)
        self.E_id, self.E_from, self.E_to, self.E_to_state, self.E_from_state = self.get_edge_matrix()

        self.node_names = self.id_list
        self.ILDS_in = []
        self.CAP=[]
        ss = []
        for node_name in self.node_names:
            lanes_in = self.traci.trafficlight.getControlledLanes(node_name)
            ilds_in = []
            lanes_cap = []
            for lane_name in lanes_in:
                cur_ilds_in = [lane_name]
                ilds_in.append(cur_ilds_in)
                cur_cap = 0
                for ild_name in cur_ilds_in:
                    cur_cap += self.traci.lane.getLength(ild_name)
                lanes_cap.append(cur_cap/float(self.VEH_LEN_M))
            ss.append(len(lanes_cap))
            self.ILDS_in.append(ilds_in)
            self.CAP.append(lanes_cap)
        self.n_s = max(ss)
        self.n_s_ls = [self.n_s]*self.n_agent
        self.n_a_ls = [self.n_action]*self.n_agent
        print("state_space=",self.n_s)
        self.fp = np.ones((self.n_agent, self.n_action)) / self.n_action

        self.state_heterogeneous_space = ss
        for i in range(self.n_agent):
            self.state_heterogeneous_space[i] = ss[i]
            indices = np.where(self.neighbor_mask[i] == 1)[0]
            for j in range(len(indices)):
                self.state_heterogeneous_space[i]+=ss[j]
        np.savetxt('state_heterogeneous_space.csv', np.array(self.state_heterogeneous_space), delimiter=',')




        self.traci.close()
        sys.stdout.flush()



    def cal_n_order_matrix(self,n_nodes,max_order,adj):
        def calculate_high_order_adj(max_order):
            result_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
            for i in range(n_nodes):
                for j in range(n_nodes):                  
                    if abs(j-i) <= max_order:
                        result_matrix[i][j] = 1 

            return result_matrix

        adjacency_matrix = np.eye(n_nodes)
        result = calculate_high_order_adj(max_order)
        for q in range(n_nodes):
            for k in range(n_nodes):
                if adj[q][k]==1:
                    result[q][k]=1
        return result - np.eye(n_nodes)



    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def get_neighbor_matrix(self):
        self.neighbor_mask = np.eye(self.n_agent)
        for i in range(self.n_agent):
            if self.id_list[i] in self.AdjacencyList:
                L = list(self.AdjacencyList[self.id_list[i]].keys())
                l1 = len(L)
                for j in range(l1):
                    if L[j] in self.id_list:
                        index = self.id_list.index(L[j])
                        self.neighbor_mask[i][index] = 1
        return self.neighbor_mask

    def get_edge_matrix(self):
        E_id = []
        E_from = []
        E_to = []
        
        for i in range(len(self.Edges)):
            E_id.append(self.Edges[i].getID())
            E_from.append(self.Edges[i].getFromNode().getID())
            E_to.append(self.Edges[i].getToNode().getID())
        
        E_to_state = [] 
        E_from_state = [] 
        for p in range(self.n_agent):
            E_index_to = [index for index, value in enumerate(E_to) if value == self.id_list[p]]
            E_index_from = [index for index, value in enumerate(E_from) if value == self.id_list[p]]
            e_to = []
            e_from = []
            for q in E_index_to:
                e_to.append(E_id[q])
            for w in E_index_from:
                e_from.append(E_id[w])
            E_to_state.append(e_to)
            E_from_state.append(e_from)
        return E_id, E_from, E_to, E_to_state,E_from_state
    

    def generateTopology(self):	
        AdjacencyList = {}
        for e in self.net.getEdges():
            if AdjacencyList.__contains__(str(e.getFromNode().getID()))==False:
                AdjacencyList[str(e.getFromNode().getID())]={}
            AdjacencyList[str(e.getFromNode().getID())][str(e.getToNode().getID())] = e.getLanes()[0].getLength()
        return AdjacencyList
    
    # def get_options(self):
    #     optParser = optparse.OptionParser()
    #     optParser.add_option("--nogui", action="store_true",
    #                          default=True, help="run the commandline version of sumo")
    #     options, args = optParser.parse_args()
    #     return options

    # def get_state(self,old_state):
    #     cur_state_delta = []
    #     #print("old_state=",state) 
    #     for k in range(self.n_agent): 
    #         cur_wave_to = 0    
    #         cur_wave_from = 0       
    #         for o in range(len(self.E_to_state[k])):
    #             cur_wave_to += self.traci.edge.getLastStepVehicleNumber(self.E_to_state[k][o]) 
    #         #print("cur_wave=",cur_wave)
    #         cur_state_delta.append(cur_wave_to)
    #     cur_state_delta = np.array(cur_state_delta)
    #     # print("cur_state=",cur_state)             #状态信息，获取每个路口上个时刻驶入的车辆数量
    #     new_state = old_state + cur_state_delta
    #     # print("new_state=",new_state) 

    #     return new_state



    def get_state(self):
        cur_state = []
        for k, ild in enumerate(self.ILDS_in):

            cur_wave = []
            for j, ild_seg in enumerate(ild):
                cur_wave.append(self.traci.lane.getLastStepVehicleNumber(ild_seg[0])/self.CAP[k][j])
            cur_state.append(cur_wave)
        cur_state = np.array(cur_state)



        cur_state_padded = np.array([np.pad(sublist, (0, self.n_s - len(sublist)), constant_values=0.0) for sublist in cur_state])




        return cur_state_padded



    def get_reward(self):

        if self.obj == "queue":

            queues = []
            for k, ild in enumerate(self.ILDS_in):
                cur_queue = 0
                for j, ild_seg in enumerate(ild):
                    cur_queue += self.traci.lane.getLastStepHaltingNumber(ild_seg[0])
                queues.append(cur_queue)
            reward = -np.array(queues)/10000


        elif self.obj == "wait":
            waits = []
            for k, ild in enumerate(self.ILDS_in):
                for j, ild_seg in enumerate(ild):
                    max_pos = 0
                    cur_cars = self.traci.lane.getLastStepVehicleIDs(ild_seg[0])
                    for vid in cur_cars:
                        car_pos = self.traci.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.traci.vehicle.getWaitingTime(vid)   
                            waits.append(car_wait)
            reward = -np.array(waits)/10000



        return reward


        
    def reset(self):

        # if self.gui:
        #     self.sumoBinary = checkBinary('sumo-gui')
        # else:
        #     self.sumoBinary = checkBinary('sumo')
        self.traci.start([self.sumoBinary, "-c", self.sim_path,"--no-warnings"])


        self.state_heterogeneous_space


        #state = np.zeros((self.n_agent, self.n_s))

        state = []
        for i in range(len(self.state_heterogeneous_space)):
            state.append(np.zeros((1,self.state_heterogeneous_space[i]))[0])

        state = np.zeros((self.n_agent, self.n_s))
        self.state = state


        return self.state 

    def clear(self):
        self.traci.close()
        sys.stdout.flush()
        return
       
    
    # def rescaleReward(self, reward, _):
    #     return reward*200/720*self.n_agent
        
    def step(self, action):

        #self.traci.trafficlight.setRedYellowGreenState(node_name, phase)
        #self.traci.trafficlight.setPhaseDuration(node_name, phase_duration)

        for i in range(self.n_agent):
            if action[i] <= self.A[i]-1:
                self.traci.trafficlight.setPhase(self.id_list[i], action[i])   #设置具体的action
                #self.traci.trafficlight.setPhaseDuration(self.id_list[i], self.single_step_second)
            else:
                a = action[i]
                while a > self.A[i]-1:
                    a = a - self.A[i]
                self.traci.trafficlight.setPhase(self.id_list[i], a)   #设置具体的action
                #self.traci.trafficlight.setPhaseDuration(self.id_list[i], self.single_step_second)

        self.arrived = []
        for _ in range(self.single_step_second):   
            self.traci.simulationStep()


            if self.obj == "arrived":
                arrived_vehicles = self.traci.simulation.getArrivedIDList()
                num_arrived_vehicles = len(arrived_vehicles)
                if num_arrived_vehicles>0:
                    self.arrived.append(num_arrived_vehicles)
        
        
        state_old = self.state
        state = self.get_state()
        
        
        if self.obj == "arrived":
            reward = np.repeat(sum(self.arrived), self.n_agent)
        else:
            reward = self.get_reward()

        

        reward = np.array(reward, dtype=np.float32)
        done = np.array([False]*self.n_agent, dtype=np.float32)
        # print('dddd=',done)
        # done = np.array(done, dtype=np.float32)
        self.state = state


        return state, reward, done, None

    def get_state_(self):
        state = self.state
        return state


def Large_city_Env():
    net_path = parent_dir + "/algorithms/envs/Large_city_data/newyork_map.net.xml"    # 436 agents
    sim_path = parent_dir + "/algorithms/envs/Large_city_data/newyork_map.sumocfg"

    
    return Large_city_net(net_path,sim_path)





class Large_city_net_2(gym.Wrapper):
    def __init__(self, net_path,sim_path):
        self.obj = "queue"     # queue   wait
        
        self.net_path = net_path
        self.sim_path = sim_path
        self.net = sumolib.net.readNet(self.net_path)
        self.AdjacencyList = self.generateTopology()    #获取邻接矩阵
        #print("self.AdjacencyList=",self.AdjacencyList)
        self.Edges = self.net.getEdges()
        self.VEH_LEN_M = 200
        self.coop_gamma = -1
        self.T = 500
        
        

        
        self.gui = True
        if self.gui:
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.traci = traci
        self.traci.start([self.sumoBinary, "-c", sim_path, "--no-warnings"])
        
        



        self.n_agent = self.traci.trafficlight.getIDCount()
        self.n_agents = self.n_agent
        if self.n_agent < 500:
            self.single_step_second = 20
        else:
            self.single_step_second = 40
        print("n_agent= ", self.n_agent)     #交通灯的数量
        self.id_list = list(self.traci.trafficlight.getIDList())
        #print("id_list= ", self.id_list)       #交通灯 的id
        #phase = traci.trafficlight.getAllProgramLogics(id_list[119])
        # print("phase=",len(phase[0].phases))    #action 的数量
        
        self.A  = []
        for i in range(self.n_agent):
            phase = self.traci.trafficlight.getAllProgramLogics(self.id_list[i])
            #print("phase=",phase)
            a = len(phase[0].phases)
            self.A.append(a)
            
        self.n_action = max(self.A)
        self.action_space = Discrete(self.n_action)
        
        # print("A=",max(A))       ##action最多的交通灯 = 动作空间的维度
        self.neighbor_mask = self.get_neighbor_matrix()
        self.distance_mask = dp(self.neighbor_mask)
        self.E_id, self.E_from, self.E_to, self.E_to_state, self.E_from_state = self.get_edge_matrix()

        self.node_names = self.id_list
        self.ILDS_in = []
        self.CAP=[]
        ss = []
        for node_name in self.node_names:
            lanes_in = self.traci.trafficlight.getControlledLanes(node_name)
            ilds_in = []
            lanes_cap = []
            for lane_name in lanes_in:
                cur_ilds_in = [lane_name]
                ilds_in.append(cur_ilds_in)
                cur_cap = 0
                for ild_name in cur_ilds_in:
                    cur_cap += self.traci.lane.getLength(ild_name)
                lanes_cap.append(cur_cap/float(self.VEH_LEN_M))
            ss.append(len(lanes_cap))
            self.ILDS_in.append(ilds_in)
            self.CAP.append(lanes_cap)
        self.n_s = max(ss)
        self.n_s_ls = [self.n_s]*self.n_agent
        self.n_a_ls = [self.n_action]*self.n_agent
        print("state_space=",self.n_s)
        self.fp = np.ones((self.n_agent, self.n_action)) / self.n_action

        self.state_heterogeneous_space = ss
        for i in range(self.n_agent):
            self.state_heterogeneous_space[i] = ss[i]
            indices = np.where(self.neighbor_mask[i] == 1)[0]
            for j in range(len(indices)):
                self.state_heterogeneous_space[i]+=ss[j]
        np.savetxt('state_heterogeneous_space.csv', np.array(self.state_heterogeneous_space), delimiter=',')




        #self.traci.close()
        #sys.stdout.flush()



    def cal_n_order_matrix(self,n_nodes,max_order,adj):
        def calculate_high_order_adj(max_order):
            result_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
            for i in range(n_nodes):
                for j in range(n_nodes):                  
                    if abs(j-i) <= max_order:
                        result_matrix[i][j] = 1 

            return result_matrix

        adjacency_matrix = np.eye(n_nodes)
        result = calculate_high_order_adj(max_order)
        for q in range(n_nodes):
            for k in range(n_nodes):
                if adj[q][k]==1:
                    result[q][k]=1
        return result - np.eye(n_nodes)


    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def get_neighbor_matrix(self):
        self.neighbor_mask = np.eye(self.n_agent)
        for i in range(self.n_agent):
            if self.id_list[i] in self.AdjacencyList:
                L = list(self.AdjacencyList[self.id_list[i]].keys())
                l1 = len(L)
                for j in range(l1):
                    if L[j] in self.id_list:
                        index = self.id_list.index(L[j])
                        self.neighbor_mask[i][index] = 1
        return self.neighbor_mask

    def get_edge_matrix(self):
        E_id = []
        E_from = []
        E_to = []
        
        for i in range(len(self.Edges)):
            E_id.append(self.Edges[i].getID())
            E_from.append(self.Edges[i].getFromNode().getID())
            E_to.append(self.Edges[i].getToNode().getID())
        
        E_to_state = [] 
        E_from_state = [] 
        for p in range(self.n_agent):
            E_index_to = [index for index, value in enumerate(E_to) if value == self.id_list[p]]
            E_index_from = [index for index, value in enumerate(E_from) if value == self.id_list[p]]
            e_to = []
            e_from = []
            for q in E_index_to:
                e_to.append(E_id[q])
            for w in E_index_from:
                e_from.append(E_id[w])
            E_to_state.append(e_to)
            E_from_state.append(e_from)
        return E_id, E_from, E_to, E_to_state,E_from_state
    

    def generateTopology(self):	
        AdjacencyList = {}
        for e in self.net.getEdges():
            if AdjacencyList.__contains__(str(e.getFromNode().getID()))==False:
                AdjacencyList[str(e.getFromNode().getID())]={}
            AdjacencyList[str(e.getFromNode().getID())][str(e.getToNode().getID())] = e.getLanes()[0].getLength()
        return AdjacencyList
    
    # def get_options(self):
    #     optParser = optparse.OptionParser()
    #     optParser.add_option("--nogui", action="store_true",
    #                          default=True, help="run the commandline version of sumo")
    #     options, args = optParser.parse_args()
    #     return options

    # def get_state(self,old_state):
    #     cur_state_delta = []
    #     #print("old_state=",state) 
    #     for k in range(self.n_agent): 
    #         cur_wave_to = 0    
    #         cur_wave_from = 0       
    #         for o in range(len(self.E_to_state[k])):
    #             cur_wave_to += self.traci.edge.getLastStepVehicleNumber(self.E_to_state[k][o]) 
    #         #print("cur_wave=",cur_wave)
    #         cur_state_delta.append(cur_wave_to)
    #     cur_state_delta = np.array(cur_state_delta)
    #     # print("cur_state=",cur_state)             #状态信息，获取每个路口上个时刻驶入的车辆数量
    #     new_state = old_state + cur_state_delta
    #     # print("new_state=",new_state) 

    #     return new_state



    def get_state(self):
        cur_state = []
        for k, ild in enumerate(self.ILDS_in):

            cur_wave = []
            for j, ild_seg in enumerate(ild):
                cur_wave.append(self.traci.lane.getLastStepVehicleNumber(ild_seg[0])/self.CAP[k][j])
            cur_state.append(cur_wave)
        cur_state = np.array(cur_state)



        cur_state_padded = np.array([np.pad(sublist, (0, self.n_s - len(sublist)), constant_values=0.0) for sublist in cur_state])




        return cur_state_padded



    def get_reward(self):

        if self.obj == "queue":

            queues = []
            for k, ild in enumerate(self.ILDS_in):
                cur_queue = 0
                for j, ild_seg in enumerate(ild):
                    cur_queue += self.traci.lane.getLastStepHaltingNumber(ild_seg[0])
                queues.append(cur_queue)
            reward_queue = -np.array(queues)


            waits = []
            for k, ild in enumerate(self.ILDS_in):
                for j, ild_seg in enumerate(ild):
                    max_pos = 0
                    cur_cars = self.traci.lane.getLastStepVehicleIDs(ild_seg[0])
                    for vid in cur_cars:
                        car_pos = self.traci.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.traci.vehicle.getWaitingTime(vid)   
                            waits.append(car_wait)
            reward_wait = -np.array(waits)


        elif self.obj == "wait":
            waits = []
            for k, ild in enumerate(self.ILDS_in):
                for j, ild_seg in enumerate(ild):
                    max_pos = 0
                    cur_cars = self.traci.lane.getLastStepVehicleIDs(ild_seg[0])
                    for vid in cur_cars:
                        car_pos = self.traci.vehicle.getLanePosition(vid)
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.traci.vehicle.getWaitingTime(vid)   
                            waits.append(car_wait)
            reward_wait = -np.array(waits)



        return reward_queue, reward_wait


        
    def reset(self):

        # if self.gui:
        #     self.sumoBinary = checkBinary('sumo-gui')
        # else:
        #     self.sumoBinary = checkBinary('sumo')
        #self.traci.start([self.sumoBinary, "-c", self.sim_path])


        self.state_heterogeneous_space


        #state = np.zeros((self.n_agent, self.n_s))

        state = []
        for i in range(len(self.state_heterogeneous_space)):
            state.append(np.zeros((1,self.state_heterogeneous_space[i]))[0])

        state = np.zeros((self.n_agent, self.n_s))
        self.state = state
        self.arrived = []
        self.halting = []
        self.speed = []


        return self.state 

    def clear(self):
        self.traci.close()
        sys.stdout.flush()
        return
       
    
    # def rescaleReward(self, reward, _):
    #     return reward*200/720*self.n_agent
        
    def step(self, action):
        

        for i in range(self.n_agent):
            if action[i] <= self.A[i]-1:
                self.traci.trafficlight.setPhase(self.id_list[i], action[i])   #设置具体的action
                #self.traci.trafficlight.setPhaseDuration(self.id_list[i], self.single_step_second)
            else:
                a = action[i]
                while a > self.A[i]-1:
                    a = a - self.A[i]
                self.traci.trafficlight.setPhase(self.id_list[i], a)   #设置具体的action
                #self.traci.trafficlight.setPhaseDuration(self.id_list[i], self.single_step_second)

        for _ in range(self.single_step_second):   
            self.traci.simulationStep()


            arrived_vehicles = self.traci.simulation.getArrivedIDList()
            num_arrived_vehicles = len(arrived_vehicles)
            if num_arrived_vehicles>0:
                #print("num_arrived_vehicles=",num_arrived_vehicles)
                self.arrived.append(num_arrived_vehicles)
            else:
                self.arrived.append(0)

    

            # halting vehicle
            # halting_vehicles = self.traci.simulation.getCollidingVehiclesIDList()
            # halting_vehicle_count = len(halting_vehicles)
            # self.halting.append(halting_vehicle_count)
            # #print("halting_vehicle_count=",halting_vehicle_count)

            threshold_speed = 0.1
            halting_vehicles = [vehicle for vehicle in self.traci.vehicle.getIDList() if self.traci.vehicle.getSpeed(vehicle) < threshold_speed]
            halting_vehicle_count = len(halting_vehicles)
            self.halting.append(halting_vehicle_count)
            #print("halting_vehicle_count=",halting_vehicle_count)


            
            # vehicle speed
            all_vehicles = self.traci.vehicle.getIDList()
            total_speed = 0
            for vehicle_id in all_vehicles:
                speed = self.traci.vehicle.getSpeed(vehicle_id)
                print("speed=",speed)
                total_speed += speed
            if len(all_vehicles)>0:
                average_speed = total_speed / len(all_vehicles)
                #print("average_speed=",average_speed)
                self.speed.append(average_speed)
            else:
                self.speed.append(0)



        


        # # 设置绿灯通过一会，不然会让某些车辆等待太久
        # for i in range(self.n_agent):
        #     self.traci.trafficlight.setPhase(self.id_list[i], 0)   
        # for _ in range(self.single_step_second):   
        #     self.traci.simulationStep()


        
        state_old = self.state
        state = self.get_state()
        
        
        
        reward_queue, reward_wait = self.get_reward()

        reward = reward_queue


        

        reward = np.array(reward, dtype=np.float32)
        done = np.array([False]*self.n_agent, dtype=np.float32)
        # print('dddd=',done)
        # done = np.array(done, dtype=np.float32)
        self.state = state


        return state, reward, done, reward_queue, reward_wait

    def get_state_(self):
        state = self.state
        return state


def Large_city_Env_2():

    
    net_path = parent_dir + "/algorithms/envs/Large_city_data/newyork_map.net.xml"    # 436 agents
    sim_path = parent_dir + "/algorithms/envs/Large_city_data/newyork_map.sumocfg"
    return Large_city_net_2(net_path,sim_path)


# this is the main entry point of this script
if __name__ == "__main__":

        
    env = Large_city_Env()
    env.reset()


    for i in range(3000):
        action = [5]*436
        state, reward, done, info = env.step(action)
        print("state=",state)
        print("reward=",reward)

