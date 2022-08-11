import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import os


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_size):
        super(DuelingDeepQNetwork, self).__init__()
        self.cv1 = nn.Conv2d(input_size[0], 32, 8, stride=4)
        self.cv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.cv3 = nn.Conv2d(64, 64, 3, stride=1)   
        input_size_fc1 = self.output_size_cv3(input_size)

        self.fc1 = nn.Linear(input_size_fc1, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512,n_actions)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device) 
        
    def forward(self, state):
        cv1 = F.relu(self.cv1(state))
        cv2 = F.relu(self.cv2(cv1))
        cv3 = F.relu(self.cv3(cv2))
        cv3 = cv3.view(cv3.size()[0],-1)
        fc1 = F.relu(self.fc1(cv3))
        V = self.V(fc1)
        A = self.A(fc1)

        return V, A
    
        return actions
    
    def output_size_cv3(self, input_size):
        shape = T.zeros(1, *input_size)
        shape = self.cv1(shape)
        shape = self.cv2(shape)
        shape = int(np.prod(self.cv3(shape).size()))

        return shape
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), r'C:\Users\YAA5\Documents\DQN')
        

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(r'C:\Users\YAA5\Documents\DQN'))