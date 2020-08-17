from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
import tensorflow_addons as tfa
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from HaliteWrapperV0 import HaliteWrapperV0
from tqdm import tqdm

tf.compat.v1.enable_v2_behavior()

num_iterations = 20000000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 10  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64 * 10 # @param {type:"integer"}
learning_rate = 0.0001  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 100  # @param {type:"integer"}
eval_interval = 400  # @param {type:"integer"}

train_py_env = HaliteWrapperV0()
eval_py_env = HaliteWrapperV0()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
train_env._env._envs[0].uuid = 'Training...'
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
eval_env._env._envs[0].uuid = 'Testing...'

fc_layer_params = (512,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in tqdm(range(num_episodes)):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


compute_avg_return(eval_env, random_policy, num_eval_episodes)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in tqdm(range(steps)):
        collect_step(env, policy, buffer)


collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)

avg_return, score = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]
iterator = iter(dataset)

for _ in tqdm(range(num_iterations)):
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        eval_env._env._envs[0].save_image = False
        avg_return, score = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        eval_env._env._envs[0].save_image = False
        print('step = {0}: Average Return = {1:.2f}, score {2}'.format(step, avg_return, score))
        returns.append(avg_return)


