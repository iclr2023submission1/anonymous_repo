""" Interface with the catcher environment
"""
import numpy as np

from base_classes.environment import Environment


class Catcher(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):

        self.name = 'catcher'
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self.actions = [0, 1]
        self.action_space = np.ones(1)
        self.num_actions = 2
        self.step_size = kwargs["step_size"]

        self._height = kwargs["height"]  # 15
        self._width = kwargs["width"]  # preferably an odd number so that it's symmetrical
        self._width_paddle = 1
        self.x = np.random.randint(self._width - self._width_paddle + 1)
        self.y = self._height-1

        self._nx_block = 8   # number of different x positions of the falling blocks
        self._higher_dim_obs = kwargs["higher_dim_obs"]
        self._reverse = kwargs["reverse"]

        if self._nx_block == 1:
            self._x_block = self._width // 2
        else:
            rand = np.random.randint(self._nx_block)  # random selection of the pos for falling block
            self._x_block = rand * (
                        (self._width - 1) // (self._nx_block - 1))

    def reset(self, mode):
        if mode == Catcher.VALIDATION_MODE:
            if self._mode != Catcher.VALIDATION_MODE:
                self._mode = Catcher.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
                np.random.seed(
                    seed=11)  # Seed the generator so that the sequence of falling blocks is the same in validation
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:  # and thus mode == -1
            self._mode = -1

        self.y = self._height - 1
        self.x = np.random.randint(self._width - self._width_paddle + 1)  # self._width//2
        if self._nx_block == 1:
            self._x_block = self._width // 2
        else:
            rand = np.random.randint(self._nx_block)  # random selection of the pos for falling block
            self._x_block = rand * (
                        (self._width - 1) // (self._nx_block - 1))  # traduction in a number in [0,self._width] of rand

        return [1 * [self._height * [self._width * [0]]]]  # [0,0,1]+[0,1,0]

    def step(self, action, dont_take_reward=False):
        """Applies the agent action [action] on the environment.
        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier
            included between 0 included and nActions() excluded.
        """

        if action == 0:
            self.x = max(self.x - self.step_size, 0)
        if action == 1:
            self.x = min(self.x + self.step_size, self._width - self._width_paddle)

        self.y = self.y - 1

        if self.y == 0 and self.x > self._x_block - 1 - self._width_paddle and self.x <= self._x_block + 1:
            self.reward = 1
        elif self.y == 0:
            self.reward = -1
        else:
            self.reward = 0

        self._mode_score += self.reward
        return self.reward

    def inputDimensions(self):
        if self._higher_dim_obs:
            return [(1, (self._height + 2) * 3, (self._width + 2) * 3)]
        else:
            return [(1, self._height, self._width)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return 2

    def observe(self):
        obs = self.get_observation(self.y, self._x_block, self.x)
        return [obs]

    def observe_multiple_balls(self):
        obs1 = self.get_observation(2, 2, self.x)
        obs2 = self.get_observation(7, 2, self.x)
        obs3 = self.get_observation(13, 2, self.x)
        obs4 = self.get_observation(2, 2, self.x)
        obs5 = self.get_observation(2, 7, self.x)
        obs6 = self.get_observation(2, 13, self.x)
        return [obs1], [obs2], [obs3], [obs4], [obs5], [obs6]

    def observe_three_balls(self):
        obs1 = self.get_observation(13, 2, 12)
        obs2 = self.get_observation(13, 13, 12)
        obs3 = self.get_observation(2, 13, 12)
        return [obs1], [obs2], [obs3]

    def observe_three_agents(self):
        obs1 = self.get_observation(7, 7, 2)
        obs2 = self.get_observation(7, 7, 7)
        obs3 = self.get_observation(7, 7, 12)
        return [obs1], [obs2], [obs3]

    def observe_multiple_agents(self):
        obs1 = self.get_observation(7, 7, 2)
        obs2 = self.get_observation(7, 7, 4)
        obs3 = self.get_observation(7, 7, 6)
        obs4 = self.get_observation(7, 7, 8)
        obs5 = self.get_observation(7, 7, 10)
        obs6 = self.get_observation(7, 7, 12)
        return [obs1], [obs2], [obs3], [obs4], [obs5], [obs6]

    def get_observation(self, y, x_block, x):
        obs = np.zeros((self._height, self._width))
        obs[y, x_block] = 0.5
        obs[0, x - self._width_paddle + 1:x + 1] = 1

        if self._higher_dim_obs:
            y_t = (1 + y) * 3
            x_block_t = (1 + x_block) * 3
            x_t = (1 + x) * 3
            obs = np.zeros(((self._height + 2) * 3, (self._width + 2) * 3))
            ball = np.array([[0, 0, 0.6, 0.8, 0.6, 0, 0],
                             [0., 0.6, 0.9, 1, 0.9, 0.6, 0],
                             [0., 0.85, 1, 1, 1, 0.85, 0.],
                             [0, 0.6, 0.9, 1, 0.9, 0.6, 0],
                             [0, 0, 0.6, 0.85, 0.6, 0, 0]])
            paddle = np.array([[0.5, 0.95, 1, 1, 1, 0.95, 0.5],
                               [0.9, 1, 1, 1, 1, 1, 0.9],
                               [0., 0., 0, 0, 0, 0., 0.]])

            obs[y_t - 2:y_t + 3, x_block_t - 3:x_block_t + 4] = ball
            obs[3:6, x_t - 3:x_t + 4] = paddle

        if self._reverse:
            obs = -obs

        return obs

    def inTerminalState(self):
        if self.y == 0:
            return True
        else:
            return False
