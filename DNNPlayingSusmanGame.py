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



# 1. Parameters of Q-leanring
gamma = .9
learning_rate = 0.002
episode = 10001
capacity = 64 * 3
batch_size = 32 * 3

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
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dense(action_space, activation='softmax')
])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

tensor_board = tf.keras.callbacks.TensorBoard(log_dir=f"{base_dir}\\{tensorboard_dir}\\{time.time()}")
model_save = tf.keras.callbacks.ModelCheckpoint(
        f'{base_dir}\\{model_dir}\\{model_name}_Checkpoint.h5',
        monitor='accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

model.summary()

reward_array = []
memory = deque([], maxlen=capacity)
for i in range(episode):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        #epsilon -= decay_rate
        #epsilon = max(epsilon, min_epsilon)
        state1 = np.array([env.get_state()])

        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = np.random.uniform()

        directive = ''
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(model.predict(state1))
            directive = 'Exploite'
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            directive = 'Explore'

        # Training without experience replay
        #print('Before')
        #env.render()
        state2, reward, done, info = env.step(action)
        #print(f'Directive:{directive}\tDirection:{env.direction_by_int[action]}\tResult:{info}')
        #env.render()
        #print('\n\n')
        state2 = np.array([state2])
        target = (reward + gamma *
                  np.max(model.predict([state2])))

        target_f = model.predict(state1)
        target_f[0][action] = target
        history = model.fit(state1, target_f, epochs=1, verbose=0)
        total_reward += reward

        state = state2

        # Training with experience replay
        # appending to memory
        memory.append((state1, action, reward, state2, done))
        # experience replay
    if i > batch_size:
        experience_replay()

    reward_array.append(total_reward)
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)

    #if i % 10 == 0 and i != 0:
        #print('Episode {} Total Reward: {} Reward Rate {}'.format(i, total_reward, str(sum(reward_array) / i)))
    try:
        print('Episode {} Total Reward: {} Reward Rate {} Epsilon {}'.format(i, total_reward, str(sum(reward_array) / i), epsilon))
    except:
        lol = 1

model.save(f'{base_dir}\\{model_dir}\\{model_name}_Complete.h5')