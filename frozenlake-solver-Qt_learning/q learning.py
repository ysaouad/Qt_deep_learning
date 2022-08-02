import gym
import matplotlib.pyplot as plt
import numpy as np
from Qt_agent_epsfactor import Qt_Agent

if __name__=='__main__':
    env = gym.make('FrozenLake-v1')
    #Initializing our environment
    agent = Qt_Agent(alpha=0.001, gamma = 0.9, eps_start=1, eps_min=0.01, eps_fact=0.9999995, env = env)
    #alpha is the learning rate. Use a value close to 0.001 ideally. gamma is the discount factor 
    #(keep it high enough if you want your agent to focus on future reward)
    #eps_start is where your epsilon will start. The higher, the more your agent will exploit.
    #the algorithm will stop decrementing epsilon at epsilon = eps_min
    #the factor eps_fact will decrement epsilon. The smaller it is, the quicker it'll turn epsilon in "exploit" mode
    
    
    scores = []
    win_pct_list = []
    n_games = 500000
    #determine the number of games
       
    for i in range(n_games):
        
        done = False
        agent.state = env.reset()
        score = 0
        # at the beginnning of each game, we reinitialize the environment
        while not done:
            action = int(agent.action())
            obs, reward,done,info = env.step(action)
            agent.update(action,reward,obs)
            agent.state = obs
            score += reward
            #this while loop will choose an action according to the agent.action() function, then execute the action with env.step(action)
            #and update the Q table + decrement epsilon, according to the results of this step.
            #The state of the agent is then updated.
            
        scores.append(score)
        #This is simply a list of scores that are averaged after every 100 games, and the end value is used as % of win, appended to a list
        
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)       
            #this is how we're appending 
            if i % 1000 == 0:
                print('episode', i, 'win pct%.2f' % win_pct, 'epsilon %.2f' % agent.eps)

               
    plt.plot(win_pct_list)
    plt.show()
