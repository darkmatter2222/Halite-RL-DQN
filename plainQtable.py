import gym
import numpy as np
from SusmanGame import SusmanGameEnv
# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n,env.action_space.n])
# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-leanring
eta = .628
gamma = .9
episodes = 5000
rev_list = [] # rewards per episode calculate
# 3. Q-learning Algorithm
for i in range(episodes):
    # Reset environment
    state = env.reset()
    rAll = 0
    done = False
    turn = 0
    #The Q-Table learning algorithm
    while turn < 99:
        env.render()
        turn += 1
        # Choose action from Q table
        multiplier = (1. / (i + 1))
        random_result = np.random.randn(1, env.action_space.n )
        random_result *= multiplier
        action = np.argmax(Q[state, :] + random_result)
        #Get new state & reward from environment
        state1, reward, done, info = env.step(action)
        #Update Q-Table with new knowledge
        Q[state, action] = Q[state, action] + eta * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])
        rAll += reward
        state = state1
        if done == True:
            break
    rev_list.append(rAll)
    env.render()
print ("Reward Sum on all episodes " + str(sum(rev_list) / episodes))
print ("Final Values Q-Table")
print (Q)