import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from SusmanGameResearch.SusmanGame import SusmanGameEnv

learning_rate = .628
gamma = .9
episodes = 50000

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.001  # Exponential decay rate for exploration prob

reward_history = []

env = SusmanGameEnv()
state_space = env.observation_space.shape
action_space = env.action_space.n

# Neural network model for DQN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=state_space),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_space, activation='softmax')
])
model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

for episide_count in range(episodes):
    # Reset environment
    state = env.reset()
    #env.render()
    total_reward = 0
    done = False
    current_turn = 0
    while current_turn < 99:
        state1 = np.array([env.get_historical_state()])
        current_turn += 1

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

        multiplier = (1. / (episide_count + 1))
        random_result = np.random.randn(1, env.action_space.n )
        random_result *= multiplier

        prediction_results1 = model.predict(state1)
        action1 = np.argmax(prediction_results1 + random_result)

        state2, reward, done, info = env.step(action1)
        state2 = np.array([state2])
        prediction_results2 = model.predict(state2)

        target_predictions = learning_rate * (reward + gamma * prediction_results1)

        history = model.fit(state1, target_predictions, epochs=10, verbose=0)
        total_reward += reward
        #env.render()
        reward_history.append(total_reward)
        if done == True:
            break

    print(f'Total R:{sum(reward_history)} \tAverage:{(sum(reward_history) / episodes)}\tEpsilon:{epsilon}')
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episide_count)
