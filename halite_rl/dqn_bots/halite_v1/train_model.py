import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from halite_rl.environments.halite_v1.env import halite
from tqdm import tqdm
import os
import json
from tf_agents.policies import policy_saver
import socket

# loading configuration...
print('loading configuration...')
_config = {}
with open('config.json') as f:
    _config = json.load(f)

# set tensorflow compatibility
tf.compat.v1.enable_v2_behavior()

# setting hyperparameters
_num_iterations = 20000000  # @param {type:"integer"}
_initial_collect_steps = 100  # @param {type:"integer"}
_collect_steps_per_iteration = 10  # @param {type:"integer"}
_replay_buffer_max_length = 100000  # @param {type:"integer"}
_batch_size = 64 * 10  # @param {type:"integer"}
_learning_rate = 0.0001  # @param {type:"number"}
_train_steps = 4000  # @param {type:"integer"}
_num_eval_episodes = 10  # @param {type:"integer"}

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
_train_py_env = halite(window_name='Training', render_me=False)
_eval_py_env = halite(window_name='Testing', render_me=True)

# wrap the pure python game in a tensorflow wrapper
_train_env = tf_py_environment.TFPyEnvironment(_train_py_env)
_eval_env = tf_py_environment.TFPyEnvironment(_eval_py_env)


print('Building Network...')
_fc_layer_params = (512,)

_q_net = q_network.QNetwork(
    _train_env.observation_spec(),
    _train_env.action_spec(),
    fc_layer_params=_fc_layer_params)

_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=_learning_rate)

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


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


# collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = _replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=_batch_size,
    num_steps=2).prefetch(3)

_agent.train = common.function(_agent.train)

_agent.train_step_counter.assign(0)
print('initial collect...')
avg_return = compute_avg_return(_eval_env, _agent.policy, _num_eval_episodes)
returns = [avg_return]
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

while True:
    print('Training...')
    for _ in tqdm(range(_train_steps)):
        for _ in range(_collect_steps_per_iteration):
            collect_step(_train_env, _agent.collect_policy, _replay_buffer)

        experience, unused_info = next(iterator)
        train_loss = _agent.train(experience).loss

        step = _agent.train_step_counter.numpy()

    print('step = {0}: loss = {1}'.format(step, train_loss))
    print('Eval Started...')
    avg_return = compute_avg_return(_eval_env, _agent.policy, _num_eval_episodes)
    returns.append(avg_return)
    train_checkpointer.save(_train_step_counter)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    tf_policy_saver.save(_save_policy_dir)

policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(_agent.policy)
tf_policy_saver.save(policy_dir)
