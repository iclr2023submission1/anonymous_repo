
import numpy as np
from base_classes.environment import Environment
import a_star_path_finding as pf

class Maze(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):

        self.name = 'maze'
        self._random_state = rng
        self._mode = 1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._episode_steps = 0
        self.actions = [0, 1, 2, 3]
        self.num_actions = len(self.actions)
        self._size_maze = kwargs.get('maze_size', 8)
        self._higher_dim_obs = kwargs.get('higher_dim_obs', False)
        self._reverse = kwargs.get('reverse', False)
        self.map_type = kwargs.get('map_type', 'simple_map')  # If simple_map = True, it creates the default map
        self.random_start = kwargs.get('random_start', True)

        self.action_space = np.ones(1)

        self._n_walls = int((self._size_maze - 2) ** 2 / 3.)
        if self.map_type == 'path_finding':
            self._n_walls =10
        self._n_rewards = 3
        self.create_map()
        self.intern_dim = 3

    def create_map(self, same_agent_pos=False, agent_pos=[5, 5]):
        valid_map = False
        while valid_map == False:
            # Agent
            if self.map_type == 'no_walls':
                self._pos_agent = [int((self._size_maze/2) -1), int((self._size_maze/2) -1)]

            # Walls
            self._pos_walls = []
            self._pos_rewards = []

            for i in range(self._size_maze):
                self._pos_walls.append([i, 0])
                self._pos_walls.append([i, self._size_maze - 1])
            for j in range(self._size_maze - 2):
                self._pos_walls.append([0, j + 1])
                self._pos_walls.append([self._size_maze - 1, j + 1])

            if self.map_type == 'path_finding' and self._size_maze ==8:
                self._n_rewards = 1
                n = 0
                if self.random_start:
                    self._pos_agent = [np.random.randint(2, 6), np.random.randint(1, 5)]  # (2, 6) and (1, 5)
                else:
                    self._pos_agent = [np.random.randint(5, 6), np.random.randint(1, 2)]   # (2, 6) and (1, 5)
                potential_reward = [1, 6]
                self._pos_rewards.append(potential_reward)
                if same_agent_pos:
                    self._pos_agent = agent_pos.copy()

                while n < self._n_walls:
                    potential_wall = [self._random_state.randint(1, self._size_maze - 1),
                                      self._random_state.randint(1, self._size_maze - 1)]
                    if potential_wall not in self._pos_walls and potential_wall not in self._pos_rewards and potential_wall != self._pos_agent:
                        self._pos_walls.append(potential_wall)
                        n += 1

            if self.map_type == 'simple_map':
                wall_locations = [[1, 4], [2, 4], [3, 4], [5, 4], [6, 4], [1, 3], [2, 3], [3, 3], [5, 3], [6, 3]]
                self._pos_agent = [4, 4]

                for loc in wall_locations:
                    self._pos_walls.append(loc)

            if self.map_type == 'simple_map2':
                wall_locations = [[4, 1], [4, 2], [4, 3], [4, 5], [4, 6]]
                self._pos_agent = [4, 4]

                for loc in wall_locations:
                    self._pos_walls.append(loc)

            if self.map_type == 'simple_map3':
                wall_locations = [[4, 1], [4, 2], [4, 3], [4, 5], [4, 6], [3, 1], [3, 2], [3, 3], [3, 5], [3, 6]]
                self._pos_agent = [4, 4]

                for loc in wall_locations:
                    self._pos_walls.append(loc)

            if self.map_type == 'simple_map4':
                wall_locations = [[1, 1], [1, 2], [2, 1], [2, 2], [5, 6], [5, 5], [6, 6], [6, 5]]
                self._pos_agent = [4, 4]

                for loc in wall_locations:
                    self._pos_walls.append(loc)

            if (self.map_type =='random') or (self.map_type == 'random_with_rewards'):
                n = 0
                self._pos_agent = [np.random.randint(1, self._size_maze -2), np.random.randint(1, self._size_maze -2)]
                if same_agent_pos:
                    self._pos_agent = agent_pos.copy()
                while n < self._n_walls:
                    potential_wall = [self._random_state.randint(1, self._size_maze - 1),
                                      self._random_state.randint(1, self._size_maze - 1)]
                    if potential_wall not in self._pos_walls and potential_wall != self._pos_agent:
                        self._pos_walls.append(potential_wall)
                        n += 1
                n = 0
                while n < self._n_rewards:

                    potential_reward = [self._random_state.randint(1, self._size_maze - 1),
                                        self._random_state.randint(1, self._size_maze - 1)]
                    if (
                            potential_reward not in self._pos_rewards and potential_reward not in self._pos_walls and potential_reward != self._pos_agent):
                        self._pos_rewards.append(potential_reward)
                        n += 1

            if self.map_type == 'random' or self.map_type == 'path_finding':
                valid_map = self.is_valid_map(self._pos_agent, self._pos_walls, self._pos_rewards)

                if valid_map and self.map_type =='random':
                    self._pos_rewards = []
            elif self.map_type == 'random_with_rewards':
                valid_map = self.is_valid_map(self._pos_agent, self._pos_walls, self._pos_rewards)
            else:
                valid_map = True

    def is_valid_map(self, pos_agent, pos_walls, pos_rewards):
        a = pf.AStar()
        walls = [tuple(w) for w in pos_walls]
        start = tuple(pos_agent)
        for r in pos_rewards:
            end = tuple(r)
            a.init_grid(self._size_maze, self._size_maze, walls, start, end)
            maze = a
            optimal_path = maze.solve()
            if optimal_path is None:
                return False

        return True

    def reset(self, mode):
        self._episode_steps = 0
        self._mode = mode
        self.create_map()

        if mode == Maze.VALIDATION_MODE:
            if self._mode != Maze.VALIDATION_MODE:
                self._mode = Maze.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0

            else:
                self._mode_episode_count += 1

        return [1 * [self._size_maze * [self._size_maze * [0]]]]

    def step(self, action, dont_take_reward=False):
        self._episode_steps += 1
        action = self.actions[action]

        reward = -0.1

        # Make it a dense reward for debugging purposes
        """if self._pos_rewards != []:
            reward += -1 * (np.linalg.norm(self._pos_rewards, ord=2) - np.linalg.norm(self._pos_agent, ord=2))"""

        if action == 0:
            if [self._pos_agent[0] + 1, self._pos_agent[1]] not in self._pos_walls:
                self._pos_agent[0] = self._pos_agent[0] + 1
        elif action == 1:
            if [self._pos_agent[0], self._pos_agent[1] + 1] not in self._pos_walls:
                self._pos_agent[1] = self._pos_agent[1] + 1
        elif action == 2:
            if [self._pos_agent[0] - 1, self._pos_agent[1]] not in self._pos_walls:
                self._pos_agent[0] = self._pos_agent[0] - 1
        elif action == 3:
            if [self._pos_agent[0], self._pos_agent[1] - 1] not in self._pos_walls:
                self._pos_agent[1] = self._pos_agent[1] - 1

        if self._pos_agent in self._pos_rewards:
            reward = 1
            if not dont_take_reward:
                self._pos_rewards.remove(self._pos_agent)

        self._mode_score += reward
        return reward

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        print("test_data_set.observations.shape")
        print(test_data_set.observations()[0][0:1])

        print("self._mode_score:" + str(self._mode_score) + ".")

    def inputDimensions(self):
        if self._higher_dim_obs == True:
            return [(1, self._size_maze * 6, self._size_maze * 6)]
        else:
            return [(1, self._size_maze, self._size_maze)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return len(self.actions)

    def observe(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        for coord_wall in self._pos_walls:
            self._map[coord_wall[0], coord_wall[1]] = 1
        for coord_reward in self._pos_rewards:
            self._map[coord_reward[0], coord_reward[1]] = 2
        self._map[self._pos_agent[0], self._pos_agent[1]] = 0.5

        if self._higher_dim_obs == True:
            indices_reward = np.argwhere(self._map == 2)
            indices_agent = np.argwhere(self._map == 0.5)
            self._map = self._map / 1.
            self._map = np.repeat(np.repeat(self._map, 6, axis=0), 6, axis=1)
            # agent repr
            agent_obs = np.zeros((6, 6))
            agent_obs[0, 2] = 0.8
            agent_obs[1, 0:5] = 0.9
            agent_obs[2, 1:4] = 0.9
            agent_obs[3, 1:4] = 0.9
            agent_obs[4, 1] = 0.9
            agent_obs[4, 3] = 0.9
            agent_obs[5, 0:2] = 0.9
            agent_obs[5, 3:5] = 0.9

            # # reward repr
            # if self.map_type == 'path_finding':
            #     reward_obs = np.zeros((6, 6))
            # else:
            reward_obs = np.zeros((6, 6))
            reward_obs[:, 1] = 0.7
            reward_obs[0, 1:4] = 0.6
            reward_obs[1, 3] = 0.7
            reward_obs[2, 1:4] = 0.6
            reward_obs[4, 2] = 0.7
            reward_obs[5, 2:4] = 0.7

            for i in indices_reward:
                self._map[i[0] * 6:(i[0] + 1) * 6:, i[1] * 6:(i[1] + 1) * 6] = reward_obs

            for i in indices_agent:
                self._map[i[0] * 6:(i[0] + 1) * 6:, i[1] * 6:(i[1] + 1) * 6] = agent_obs
            self._map = (self._map * 2) - 1  # scaling

        else:
            self._map = self._map / 2.
            self._map[self._map == 0.5] = 0.99  # agent
            self._map[self._map == 1.] = 0.5  # reward

        if self._reverse:
            self._map = -self._map  # 1-self._map

        return [self._map]

    def inTerminalState(self):
        if self.map_type == 'path_finding':
            if self._mode >= 0 and (self._episode_steps >= 50 or self._pos_rewards == []):
                return True
            else:
                return False

        if self.map_type =='random':
            if self._mode >= 0 and self._episode_steps >= 50:
                return True
            else:
                return False
        if self.map_type == 'random_with_rewards':
            if self._mode >=0 and (self._episode_steps >=50 or self._pos_rewards == []):
                return True
            else:
                return False
        else:
            if self._mode >= 0 and self._episode_steps >= 50:
                return True
            else:
                return False
