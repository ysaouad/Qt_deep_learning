import torch as T
import os
from DQN import LinearDeepQNetwork
import random, time
import numpy as np

class Agent():
    def __init__(self, input_dims, n_actions, lr, mem_size, gamma=0.99,
                 epsilon=1.0, eps_dec=5e-7, eps_min=0.01, batch_size=32):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.exp_dict = {"state" : [], "action" : [], "reward" : [], "state_" : [], "done" : []}
        self.batch_size = batch_size
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.Q_ = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.step = 0
        self.mem_size = mem_size

    def choose_action(self, observation):
        if random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def exp_replay(self, state, action, reward, state_,done):
            
            #t0= time.perf_counter() 
            exp = {"state" : state, "action" : action, "reward" : reward, "state_" : state_, "done" : done}
            if len(self.exp_dict) < self.mem_size :
                for key in self.exp_dict.keys():
                    self.exp_dict[key].append(exp[key])
            else :
                i = self.step % self.mem_size
                for key in self.exp_dict.keys():
                    self.exp_dict[key][i] = exp[key]
            batch = {}
            if self.batch_size < len(self.exp_dict["state"]):
                for key in self.exp_dict.keys():
                    batch[key] = random.sample(self.exp_dict[key], k=self.batch_size)
            self.learn(batch)
            
            self.step +=1
            
            #t1 = time.perf_counter() - t0
            #print("Time elapsed exp_replay: ", t1 ) 
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, batch):
        
        if self.batch_size > self.step:
            return False
        
        self.Q.optimizer.zero_grad()
        states = T.tensor(np.array(batch['state']), dtype=T.float).to(self.Q.device)
        actions = T.tensor(batch['action']).to(self.Q.device)
        rewards = T.tensor(batch['reward']).to(self.Q.device)
        states_ = T.tensor(np.array(batch['state_']), dtype=T.float).to(self.Q.device)
        dones = T.tensor(batch['done']).to(self.Q.device)
        indices = np.arange(self.batch_size)
        
        q_pred = self.Q.forward(states)[indices,actions]
        q_next = self.Q_.forward(states_).max()

        q_target = rewards + self.gamma*q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()
        
    def save_models(self):
        self.Q.save_checkpoint()
        self.Q_.save_checkpoint()

    def load_models(self):
        self.Q_.load_checkpoint()
        self.Q.load_checkpoint()