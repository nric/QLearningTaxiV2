
#%%
import gym
import numpy as np 

class QLearnAgent():
    def __init__(self,env):
        #self.Q = {i:{j:0 for j in range(env.action_space)} for i in range(env.state_space)}
        self.observation_space = env.observation_space.n # size of our state   
        self.action_space = env.action_space.n # number of actions
        self.actions = env.action_space
        self.Q = np.zeros((self.observation_space, self.action_space))
        self.update_counts = np.zeros((self.observation_space, self.action_space), dtype=np.dtype(int)) #Required for dynamic Learning Rate
        self.episode_N = 0
        # Hyper Parameter
        self.GAMMA = 0.95  # Discount factor from Bellman Equation
        self.START_ALPHA = 0.1  # Learning rate, how much we update our Q table each step
        self.ALPHA_TAPER = 0.01 # How much our adaptive learning rate is adjusted each S,A pair update
        self.AlPHA = self.START_ALPHA
        self.START_EPSILON = 1.0  # Start Probability of random action
        self.EPSILON_TAPER = 0.01 # How much epsilon is adjusted each episode step
        self.EPSILON = self.START_EPSILON
        
    
    def _epsilon_acion(self,state):
        if np.random.random() > (1-self.EPSILON):
            return self.actions.sample()
        else:
            return np.argmax(self.Q[env.env.s])
        
    
    def learn(self,state,action,reward,next_state):
        #do alpha math here 
        self.alpha = self.START_ALPHA / (1.0 + self.update_counts[state][action] * self.ALPHA_TAPER)
        #update counts
        self.update_counts[state][action] += 1
        #Calc new Q for current (aka previous, depends on definition) state
        TD_Target_value = reward + self.GAMMA*max(self.Q[next_state])
        self.Q[state][action] += self.AlPHA * (TD_Target_value - self.Q[state][action] )
        
    
    def get_action(self,state):
        #do epsiolon math
        self.EPSILON = self.START_EPSILON / (1.0 + self.episode_N * self.EPSILON_TAPER)
        #return epsilon greedy
        return self._epsilon_acion(state)
    
    
    
N_Epoch = 15000
N_MaxSteps = 200
total_reward = 0 #Just a metric for observing learn progress
steps = 0 #Just another metric for observing learn progress
#Create enviroment and Agent:
env = gym.make('Taxi-v2')
agent = QLearnAgent(env)

#Loop over Epochs:
for epoch in range(N_Epoch):
    state = env.reset()
    agent.episode_N = epoch+1
    if epoch % 1000 == 0:
        print(f"epoch:{epoch} Avg reward:{total_reward/1000} Avg steps:{steps/1000}")
        total_reward = 0 
        steps = 0
    #Loop over steps. Break early if done:
    for step in range(N_MaxSteps):
        #Collect variables for Q-Learn:
        state = env.env.s
        action = agent.get_action(state)
        next_state,reward,done,info = env.step(action)
        #Here, the agent.learn fkt is called which does the Q-Learning:
        agent.learn(state,action,reward,next_state)
        #Record metrics, test if done:
        total_reward += reward
        if done:
            steps += step
            break
    