from os.path import dirname, abspath, join
import sys
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', '..', '..'))
THIS_DIR = abspath(join(THIS_DIR))
sys.path.append(THIS_DIR)
sys.path.append(CODE_DIR)
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from halite_rl.environments.halite_v4.env import halite_ship_navigation
from tqdm import tqdm
import os
import cv2
import json
from tf_agents.policies import policy_saver
import numpy as np
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow_addons as tfa

# loading configuration...
print('loading configuration...')
_config = {}
with open('config.json') as f:
    _config = json.load(f)

# set tensorflow compatibility
tf.compat.v1.enable_v2_behavior()

# setting hyperparameters
 #_num_iterations = 20000000  # @param {type:"integer"}
#_initial_collect_steps = 10  # @param {type:"integer"}
#_collect_steps_per_iteration = 10  # @param {type:"integer"}
_replay_buffer_max_length = 4000   # @param {type:"integer"}
_batch_size = 64  # @param {type:"integer"}
_learning_rate = 0.001  # @param {type:"number"}
_num_train_episodes = 100 # @param {type:"integer"}
_num_eval_episodes = 10  # @param {type:"integer"}
_num_save_episodes = 20  # @param {type:"integer"}
#_render_on_episode = 10  # @param {type:"integer"}


reward_history = []
loss_history = []

# build policy directories
host_name = socket.gethostname()
base_directory_key = 'base_dir'
target = f'{host_name}-base_dir'
if target in _config['files']['policy']:
    base_directory_key = target


_save_policy_dir = os.path.join(_config['files']['policy'][base_directory_key],
                                _config['files']['policy']['save_policy']['dir'],
                                _config['files']['policy']['save_policy']['name'])

_checkpoint_policy_dir = os.path.join(_config['files']['policy'][base_directory_key],
                                      _config['files']['policy']['checkpoint_policy']['dir'],
                                      _config['files']['policy']['checkpoint_policy']['name'])

# instantiate two environments. I personally don't feel this is necessary,
# however google did it in their tutorial...
_train_py_env = halite_ship_navigation(env_name='Training', render_me=True)
_eval_py_env = halite_ship_navigation(env_name='Testing', render_me=True)

# wrap the pure python game in a tensorflow wrapper
_train_env = tf_py_environment.TFPyEnvironment(_train_py_env)
_eval_env = tf_py_environment.TFPyEnvironment(_eval_py_env)


print('Building Network...')
_fc_layer_params = (64,)

_q_net = q_network.QNetwork(
    _train_env.observation_spec(),
    _train_env.action_spec(),
    fc_layer_params=_fc_layer_params)

clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=0.1, maximal_learning_rate=0.0001, step_size=200,
                                          scale_fn=lambda x:1.)

_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=clr)

_train_step_counter = tf.Variable(0)

_agent = dqn_agent.DqnAgent(
    _train_env.time_step_spec(),
    _train_env.action_spec(),
    q_network=_q_net,
    optimizer=_optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=_train_step_counter)

_agent.initialize()

_eval_policy = _agent.policy
_collect_policy = _agent.collect_policy

_random_policy = random_tf_policy.RandomTFPolicy(_train_env.time_step_spec(),
                                                 _train_env.action_spec())


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


_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=_agent.collect_data_spec,
    batch_size=_train_env.batch_size,
    max_length=_replay_buffer_max_length)


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    _replay_buffer.add_batch(traj)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def render_history():
    figure, axes = plt.subplots(2)
    canvas = FigureCanvas(figure)

    axes[0].plot(reward_history, 'red')
    axes[0].plot(smooth(reward_history, 4), 'orange')
    axes[0].plot(smooth(reward_history, 8), 'yellow')
    axes[0].plot(smooth(reward_history, 16), 'green')
    axes[0].plot(smooth(reward_history, 32), 'blue')
    axes[0].plot(smooth(reward_history, 64), 'purple')
    axes[1].plot(loss_history, 'red')
    axes[1].plot(smooth(loss_history, 4), 'orange')
    axes[1].plot(smooth(loss_history, 8), 'yellow')
    axes[1].plot(smooth(loss_history, 16), 'green')
    axes[1].plot(smooth(loss_history, 32), 'blue')
    axes[1].plot(smooth(loss_history, 64), 'purple')
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    img = image.reshape(canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like
    cv2.imshow("plot", img)
    plt.close('all')
# collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = _replay_buffer.as_dataset(
    num_parallel_calls=30,
    sample_batch_size=_batch_size,
    num_steps=2).prefetch(30)

_agent.train = common.function(_agent.train)

_agent.train_step_counter.assign(0)
#print('initial collect...')
#avg_return = compute_avg_return(_eval_env, _agent.policy, _num_eval_episodes)
#returns = [avg_return]
iterator = iter(dataset)

train_checkpointer = common.Checkpointer(
    ckpt_dir=_checkpoint_policy_dir,
    max_to_keep=1,
    agent=_agent,
    policy=_agent.policy,
    replay_buffer=_replay_buffer,
    global_step=_train_step_counter
)

tf_policy_saver = policy_saver.PolicySaver(_agent.policy)

restore_network = True

if restore_network:
    train_checkpointer.initialize_or_restore()

#_train_env.pyenv._envs[0].set_rendering(enabled=False)

returns = []

while True:
    print('Collecting...')
    for _ in tqdm(range(_num_train_episodes)):
        time_step = _train_env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            collect_step(_train_env, _agent.collect_policy)
            time_step = _train_env.current_time_step()
    print('Training...')
    experience, unused_info = next(iterator)
    train_loss = _agent.train(experience).loss
    step = _agent.train_step_counter.numpy()
    print('step = {0}: loss = {1}'.format(step, train_loss))
    print('Evaulating...')
    avg_return = compute_avg_return(_eval_env, _agent.policy, _num_eval_episodes)
    returns.append(avg_return)
    if step % _num_save_episodes == 0:
        train_checkpointer.save(_train_step_counter)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    reward_history.append(avg_return)
    loss_history.append(train_loss)
    render_history()
    if step % _num_save_episodes == 0:
        tf_policy_saver.save(_save_policy_dir)

policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(_agent.policy)
tf_policy_saver.save(policy_dir)