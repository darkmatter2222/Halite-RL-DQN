import gym
from gym import spaces
import numpy as np
import random
import scipy as sp

class SusmanGameEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    direction_by_int = {0: 'NORTH', 1: 'EAST', 2: 'SOUTH', 3: 'WEST'}
    def __init__(self):
        super(SusmanGameEnv, self).__init__()
        self.board_width = 3
        self.board_height = 3
        self.max_turns = self.board_width + self.board_height * 30
        self.board = np.zeros([self.board_height, self.board_width])
        self.reward_heatmap = np.zeros([self.board_height, self.board_width])
        self.player_location = {'x':0, 'y': 0}
        self.this_turn = 0
        self.running_reward = 0
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4) # East, West
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=
                    (3, self.board_height, self.board_width), dtype=np.uint8)
        # Dimension 0 = Board (One Hot)
        # Dimension 1 = Player (One Hot)
        # Dimension 2 = Reward Heat Map (One Hot)

        self.history = np.zeros([self.max_turns, 3, self.board_height, self.board_width])
        self.reset_board()
        self.set_goal()
        self.append_to_state()

    def get_observations(self, reward = 0, done = False, info = ''):
        #state, reward, done, info
        return self.get_current_state().tolist(), reward, done, info

    def get_current_state(self):
        state = np.zeros([3, self.board_height, self.board_width])

        for y in range(self.board_height):
            for x in range(self.board_width):
                state[0, y, x] = self.board[y, x]
                state[2, y, x] = self.reward_heatmap[y, x]
                if self.player_location['x'] == x and self.player_location['y'] == y:
                    state[1, y, x] = 1

        return state

    def get_historical_state(self):
        return self.history

    def append_to_state(self):
        self.history[self.this_turn] = self.get_current_state()

    def set_goal(self):
        self.set_player()
        rand_y = 0
        rand_x = 0

        while True:
            if self.board_width > 1:
                rand_x = random.randrange(0, self.board_width - 1)
            if self.board_height > 1:
                rand_y = random.randrange(0, self.board_height - 1)
            if self.player_location['x'] != rand_x or self.player_location['y'] != rand_y:
                break

        self.board[rand_y, rand_x] = 1
        #self.reward_heatmap[rand_y, rand_x] = 100 * (self.board_height * self.board_width)
        self.reward_heatmap[rand_y, rand_x] = 1


        sigma_y = 1
        sigma_x = 1
        # Apply gaussian filter
        sigma = [sigma_y, sigma_x]
        self.reward_heatmap = sp.ndimage.filters.gaussian_filter(self.reward_heatmap, sigma, mode='constant')
        lol = 1

    def set_player(self):
        rand_y = 0
        rand_x = 0

        if self.board_width > 1:
            rand_x = random.randrange(0, self.board_width - 1)
        if self.board_height > 1:
            rand_y = random.randrange(0, self.board_height - 1)
        self.player_location['x'] = rand_x
        self.player_location['y'] = rand_y

    def step(self, action):
        reward = 0
        info = ''
        done = False
        continue_reward = -1
        win_reward = 100
        loose_reward = 0
        # 0=N 1=E 2=S 3=W
        if action == 0:  # Move North
            self.player_location['y'] = self.player_location['y'] - 1
        elif action == 1:  # Move East
            self.player_location['x'] = self.player_location['x'] + 1
        elif action == 2:  # Move South
            self.player_location['y'] = self.player_location['y'] + 1
        elif action == 3:  # Move West
            self.player_location['x'] = self.player_location['x'] - 1
        else:
            raise ValueError

        # Max Tries?
        if self.this_turn == self.max_turns - 1:
            info = 'Max Tries'
            done = True
            reward = 0
        else:
            # Loose Fall Off Map?
            if self.player_location['y'] < 0 or self.player_location['x'] < 0 or\
                    self.player_location['x'] >= self.board_width or self.player_location['y'] >= self.board_height:
                info = 'Loose Fall Off Map'
                done = True
                reward = loose_reward
            elif self.board[self.player_location['y'], self.player_location['x']] == 1:
                info = 'Won Got the Goal'
                done = True
                reward = win_reward
            #elif self.reward_heatmap[self.player_location['y'], self.player_location['x']] != 0:
                #info = 'Continue w/ reward'
                #done = False
                #reward = self.reward_heatmap[self.player_location['y'], self.player_location['x']]
            else:
                info = 'Continue'
                done = False
                reward = continue_reward

        self.append_to_state()

        self.this_turn += 1

        return self.get_observations(reward=reward, done=done, info=info)




    def reset(self):
        self.history = np.zeros([self.max_turns, 3, self.board_height, self.board_width])
        self.this_turn = 0
        self.reset_board()
        self.set_goal()
        self.append_to_state()
        self.running_reward = 0
        return self.get_historical_state()

    def render(self, mode='human', close=False):
        result = ''
        precolor_ship = '\x1b[31m'
        postcolor_ship = '\x1b[0m'
        precolor_target = '\x1b[32m'
        postcolor_target = '\x1b[0m'
        for y in range(self.board_height):
            for x in range(self.board_width):
                result += ' '
                result += precolor_target
                result += str(self.board[y, x])
                result += postcolor_target
                if self.player_location['x'] == x and self.player_location['y'] == y:
                    result += precolor_ship
                    result += 'S'
                    result += postcolor_ship
                else:
                    result += ' '
                result += '|'
            result += '\n'

        print(result)
        # Render the environment to the screen

    def reset_board(self):
        self.board = np.zeros([self.board_height, self.board_width])
        self.reward_heatmap = np.zeros([self.board_height, self.board_width])
