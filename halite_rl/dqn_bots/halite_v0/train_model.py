import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments.wrappers import ActionDiscretizeWrapper
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from halite_rl.environments.halite_v0.env import halite
from tqdm import tqdm
import os
from tf_agents.policies import policy_saver

print('Initialization...')
tf.compat.v1.enable_v2_behavior()

num_iterations = 20000000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 10  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64 * 10 # @param {type:"integer"}
learning_rate = 0.0001  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

train_steps = 4000 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 400  # @param {type:"integer"}

base_dir = 'N:\\Halite'
policy_dir = 'Policy'
tensorboard_dir = 'Logs'
model_name = 'v4'

tempdir = f'{base_dir}\\{policy_dir}'


train_py_env = halite()
eval_py_env = halite()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
train_env._env._envs[0].uuid = 'Training...'
print('Action Spec:', train_env.action_spec())
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
eval_env._env._envs[0].uuid = 'Testing...'

print('Building Network...')

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


#compute_avg_return(eval_env, random_policy, num_eval_episodes)

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
    for _ in range(steps):
        collect_step(env, policy, buffer)


#collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

agent.train = common.function(agent.train)

agent.train_step_counter.assign(0)
print('initial collect...')
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]
iterator = iter(dataset)

checkpoint_dir = os.path.join(tempdir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)
policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

restore_network = True

if restore_network:
    train_checkpointer.initialize_or_restore()

while True:
    print('Training...')
    for _ in tqdm(range(train_steps)):
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

    print('step = {0}: loss = {1}'.format(step, train_loss))
    print('Eval Started...')
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns.append(avg_return)
    train_checkpointer.save(train_step_counter)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    tf_policy_saver.save(policy_dir)


policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)

