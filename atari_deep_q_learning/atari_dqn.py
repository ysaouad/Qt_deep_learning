import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import os
import matplotlib.pyplot as plt
from util import build_env,plot_learning_curve
from agent import Agent


os.chdir(r'C:\Users\YAA5\Desktop\atari_deep_q_learning') #SET OS DIR FOR THE PLOT

if __name__ == '__main__':
    env = build_env('PongNoFrameskip-v4')
    n_games = 250
    best_score = -np.inf
    scores = []
    eps_history = []
    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n, mem_size=50000)
    load = False
    
    if load:
        agent.load_models()
    
    print('Using device:', agent.Q.device)
    
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            
            
            obs = obs_
            if agent.step % 10 == 0:
                agent.Q_.load_state_dict(agent.Q.state_dict())
            
            if not load:
                agent.exp_replay(obs,action, reward, obs_,done)
        
        avg_score = np.mean(scores[-5:])  
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if avg_score > best_score:
            agent.save_models()
            best_score = avg_score
        
        print('episode ', i, 'score %.1f epsilon %.2f' %
                  (score, agent.epsilon))

filename = 'atari_dqn.png'
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, eps_history, filename)