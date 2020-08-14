import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from SusmanGame import SusmanGameEnv
import time
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import cv2

def render_image(env, state, directive):

    npimage=np.array(state)

    new_image = np.zeros([env.board_height, env.board_width, 3])


    for height in range(env.board_height):
        for width in range(env.board_width):
            new_image[height][width] = (0, npimage[2][height][width], 0)
            if npimage[0][height][width] == 1:
                new_image[height][width] = (0, 254, 0)
            if npimage[1][height][width] == 1:
                if directive == 'Exploite':
                    new_image[height][width] = (254, 0, 0)
                elif directive == 'Explore':
                    new_image[height][width] = (0, 0, 254)
                else:
                    new_image[height][width] = (254, 0, 254)


    n = 100
    new_image = new_image.repeat(n,axis=0).repeat(n,axis=1)
    cv2.imshow('image',new_image)
    cv2.waitKey(1)

def one_hot_state(state, state_space):
    state_m = np.zeros(state_space)
    state_m[0][state] = 1
    return state_m


def experience_replay():
    # Sample minibatch from the memory
    minibatch = random.sample(memory, batch_size)
    # Extract informations from each memory
    for state, action, reward, next_state, done in minibatch:
        # if done, make our target reward
        target = reward
        if not done:
            # predict the future discounted reward
            target = reward + gamma * \
                     np.max(model.predict(next_state))
        # make the agent to approximately map
        # the current state to future discounted reward
        # We'll call that target_f
        target_f = model.predict(state)
        target_f[0][action] = target
        # Train the Neural Net with the state and target_f
        history = model.fit(state, target_f, epochs=1, verbose=0)
        #print(f"loss:{history.history['loss']} accuracy{history.history['accuracy']}")

base_dir = 'N:\\Halite'
model_dir = 'Models'
tensorboard_dir = 'Logs'
model_name = 'SusmanGameDQNv1'

history = {'Loose Fall Off Map': 0, 'Win Got Target': 0}

# 1. Parameters of Q-leanring
gamma = .9
learning_rate = 0.8
episode = 10001
capacity = 64 * 1
batch_size = 32 * 1

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.001  # Exponential decay rate for exploration prob

# 2. Load Environment
env = SusmanGameEnv()
#env = gym.make("FrozenLake-v0", is_slippery=False)

# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
state_space = env.observation_space.shape
action_space = env.action_space.n

# Neural network model for DQN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=state_space),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24 * 10, activation='relu'),
    tf.keras.layers.Dense(action_space, activation='softmax')
])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

model.summary()

reward_array = []
r_rate_array = []
memory = deque([], maxlen=capacity)
last_r_rate = 0
last_r_rate_30 = 0

for i in range(episode):
    state = env.reset()
    total_reward = 0
    done = False
    runnin_history = []
    while not done:
        #epsilon -= decay_rate
        #epsilon = max(epsilon, min_epsilon)
        state1 = np.array([env.get_current_state()])

        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = np.random.uniform()

        directive = ''
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            prediction_results1 = model.predict(state1)
            action = np.argmax(prediction_results1)
            directive = 'Exploite'
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            directive = 'Explore'




        state2, reward, done, info = env.step(action)
        render_image(env, state2, directive)
        state2 = np.array([state2])
        prediction_results = model.predict([state2])
        prediction = np.max(prediction_results)

        target = (reward + gamma * prediction)

        target_f = model.predict(state1)
        target_f[0][action] = target
        history = model.fit(state1, target_f, epochs=1, verbose=0)
        total_reward += reward
        runnin_history.append(history.history["loss"])
        state = state2

        #env.render()

        # Training with experience replay
        # appending to memory
        memory.append((state1, action, reward, state2, done))
        # experience replay
    #if i > batch_size:
        #experience_replay()

    reward_array.append(total_reward)
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)

    #if i % 10 == 0 and i != 0:
        #print('Episode {} Total Reward: {} Reward Rate {}'.format(i, total_reward, str(sum(reward_array) / i)))
    try:
        r_rate = sum(reward_array) / i
        r_rate_30 = sum(reward_array[-30:]) / 30
        r_rate_color = ''
        r_rate_30_color = ''
        round_to = 2

        if r_rate < last_r_rate:
            r_rate_color = f'\x1b[31m{round(r_rate, round_to)}\x1b[0m'
        else:
            r_rate_color = f'\x1b[32m{round(r_rate, round_to)}\x1b[0m'

        if r_rate_30 < last_r_rate_30:
            r_rate_30_color = f'\x1b[31m{round(r_rate_30, round_to)}\x1b[0m'
        else:
            r_rate_30_color = f'\x1b[32m{round(r_rate_30, round_to)}\x1b[0m'


        last_r_rate = r_rate
        r_rate_array.append(r_rate)
        last_r_rate_30 = r_rate_30

        print(f'Epi:{i}\t Total R:{round(total_reward, 2)}\t R Rate {r_rate_color}\t '
              f'R Rate L30 {r_rate_30_color}\t Epsilon {round(epsilon, 4)}\t '
              f'A Loss:{sum(runnin_history[0]) / len(runnin_history[0]) }')
    except:
        lol = 1


model.save(f'{base_dir}\\{model_dir}\\{model_name}_Complete.h5')