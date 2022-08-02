import numpy as np

#Qtie Agent class : a cute 'n short class for Q learning

class Qt_Agent:

    def __init__(self, eps_start, eps_min, eps_fact, env, alpha, gamma):
        self.env = env
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_fact = eps_fact
        self.alpha = alpha
        self.gamma =  gamma
        self.Q = np.zeros([env.observation_space.n,env.action_space.n])
        self.state = 0
    
    #  Lots of input parameters ! The agent is to be initialized once, at the very beginning of your code
    #  env simply allows us to adapt the Q-table to the given environment
    #  eps_start, min, and fact allows us to solve the explore-exploit dillema. The lower the eps_start, the more quickly your agent will focus on exploit
    #  The higher eps_fact, the slower the EPS will be decremented. 0.9999995 is a good value for it.
    #  alpha will determine how quickly your Q table will be adjusted : it's your learning rate 
    #  gamma will allow you to ponderate for time preference : it's your discount factor (for future estimated rewards)
    #  Q will be a numpy matrix
    #  state simply allows us to keep in memory the previous state, from the environment we're using.
    def update_state(self, obs):
        self.state = obs        
    
    def action(self):
        choice = np.random.randint(4) if np.random.binomial(1,self.eps) == 1 else np.argmax(self.Q[self.state])
        # if eps is high, it'll probably make a random choice from 0 to 3, else, it'll more probably take the highest Q in the array corresponding to the current state of the agent
        return choice
    
    def update(self,action,reward,stateprime):
        self.Q[self.state][action] += self.alpha * (reward + self.gamma*(np.amax(self.Q[stateprime]) - self.Q[self.state][action])) 
        # Q equation, derived from Bellman's
        if self.eps > self.eps_min :
            self.eps = self.eps_fact*self.eps
        # if we notice that eps is low enough, we stop decrementing it. Else, we decrement it by multiplying it by our decrement factor.
        self.state = stateprime
        # we update our previous state value with the current one
