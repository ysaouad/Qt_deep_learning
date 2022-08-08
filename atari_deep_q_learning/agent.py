import torch as T
import os
from DQN import LinearDeepQNetwork
import random
import numpy as np

class Agent():
    def __init__(self, input_dims, n_actions, lr, mem_size, batch_size, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.1):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.lr = lr
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.Q_ = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.step = 0
        self.mem_size = mem_size
        self.index_mem = 0
        
        self.states = np.zeros((self.mem_size, *input_dims),dtype=np.float32)
        self.states_ = np.zeros((self.mem_size, *input_dims),dtype=np.float32)
        self.actions= np.zeros(self.mem_size, dtype=np.int64)
        self.rewards = np.zeros(self.mem_size, dtype=np.float32)
        self.dones = np.zeros(self.mem_size, dtype=np.bool)
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def exp_replay(self, state, action, reward, state_, done):
            self.index_mem = self.step % self.mem_size
            
            self.step +=1
            self.states[self.index_mem] = state
            self.states_[self.index_mem] = state_
            self.actions[self.index_mem] = action
            self.rewards[self.index_mem] = reward
            self.dones[self.index_mem] = done
            
            if self.batch_size > self.step:
             return 
            
            curr_mem = min(self.mem_size, self.step)
            
            index_batch = np.random.choice(curr_mem, self.batch_size, replace=False)
            states = self.states[index_batch]
            states_ = self.states_[index_batch]
            actions = self.actions[index_batch]
            rewards = self.rewards[index_batch]
            dones = self.dones[index_batch]
            

            self.learn(states, states_, actions, rewards, dones)
            
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, states, states_, actions, rewards, dones):
        
        self.Q.optimizer.zero_grad()
        
        states = T.tensor(states).to(self.Q.device)
        actions = T.tensor(actions).to(self.Q.device)
        rewards = T.tensor(rewards).to(self.Q.device)
        states_ = T.tensor(states_).to(self.Q.device)
        dones = T.tensor(dones).to(self.Q.device)
        indices = np.arange(self.batch_size)
        
        q_pred = self.Q.forward(states)[indices,actions]
        q_next = self.Q_.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
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