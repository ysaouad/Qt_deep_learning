import torch as T
import os
import numpy as np
from DQN import LinearDeepQNetwork

class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.exp_list = []
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.Q_ = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def exp_replay(self, state, action, reward, state_,done):
        
            self.exp_list.append([state,action,reward,state_,done])
            rand = np.random.randint(0,len(self.exp_list))
            self.learn(self.exp_list[rand][0],self.exp_list[rand][1],self.exp_list[rand][2],self.exp_list[rand][3],self.exp_list[rand][4])
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_, done):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)
        dones = T.tensor(done).to(self.Q.device)
        
        q_pred = self.Q.forward(states)[actions]
        q_next = self.Q_.forward(states_).max(dim=0)[0] 
        
        q_next[dones] = 0.0   

        q_target = rewards + self.gamma*q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()