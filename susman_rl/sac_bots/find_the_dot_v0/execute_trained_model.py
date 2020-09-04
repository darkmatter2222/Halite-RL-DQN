import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from susman_rl.environments.find_the_dot_v0.env import find_the_dot
from tqdm import tqdm
import os
import json
from tf_agents.policies import policy_saver

print('loading configuration...')
_config = {}
with open('config.json') as f:
    _config = json.load(f)

# build policy directories
_save_policy_dir = os.path.join(_config['files']['policy']['base_dir'],
                                _config['files']['policy']['save_policy']['dir'],
                                _config['files']['policy']['save_policy']['name'])


def compute_avg_return(environment, policy, num_episodes=1000):
    score = {'win': 0, 'loss': 0, 'timeout': 0}
    total_return = 0.0
    for _ in tqdm(range(num_episodes)):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        history = environment._env.envs[0].get_game_history()
        final_step = history[len(history) - 1]
        if final_step == 'Max Tries':
            score['timeout'] += 1
        elif final_step == 'Loose Fall Off Map':
            score['loss'] += 1
        elif final_step == 'Won Got the Goal':
            score['win'] += 1

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0], score


_eval_py_env = find_the_dot(window_name='Testing')
_eval_env = tf_py_environment.TFPyEnvironment(_eval_py_env)

saved_policy = tf.compat.v2.saved_model.load(_save_policy_dir)

avg_return, score = compute_avg_return(_eval_env, saved_policy)
print('Average Return = {0:.2f}, score {1}'.format(avg_return, score))
