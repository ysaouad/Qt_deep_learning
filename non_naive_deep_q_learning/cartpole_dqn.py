import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import os
import matplotlib.pyplot as plt
import numpy as np
from util import plot_learning_curve
from agent import Agent


os.chdir() #SET OS DIR FOR THE PLOT

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []
    step = 0
    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)
    
    print('Using device:', agent.Q.device)
    
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        
        while not done:
            
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.exp_replay(obs,action, reward, obs_,done)
            obs = obs_
            step += 1
            if step % 1000 == 0:
                agent.Q_.load_state_dict(agent.Q.state_dict())
                
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episodenp ', i, 'score %.1f avg score %.1f epsilon %.2f' %
                  (score, avg_score, agent.epsilon))

filename = 'cartpole_naive_dqn.png'
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, eps_history, filename)