import copy
import numpy as np
from replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
import torch
import random as r
import os
from os import path
from environments.maze_env import Maze


def strtobool(v):
  return str(v).lower() in ("yes", "true", "t", "1")


def to_numpy(tensor):
    if tensor is None:
        return None
    elif tensor.nelement() == 0:
        return np.array([])
    else:
        return tensor.cpu().detach().numpy()


def fill_buffer(buffer, num_transitions, env, noreset=False):
    if noreset:
        dont_take_reward = True
    else:
        dont_take_reward = False
        if env.name == 'maze':
            env.create_map()
            mode=1
        elif env.name == 'catcher':
            mode=-1
        else:
            pass

    """Fill buffer using random actions"""
    for i in range(num_transitions):

        done = False
        state = env.observe()[0]
        action = env.actions[r.randrange(env.num_actions)]
        reward = env.step(action, dont_take_reward=dont_take_reward)
        next_state = env.observe()[0]

        if not noreset:
            if env.inTerminalState():
                env.reset(mode=mode)
                done = True

        buffer.add(state, action, reward, next_state, done)


def fill_buffer_limited_multimaze(buffer, num_transitions_per_maze, env, num_mazes=10):
    """Fill buffer using random actions"""
    for i in range(num_transitions_per_maze):

        done = False
        state = env.observe()[0]
        action = env.actions[r.randrange(env.num_actions)]
        reward = env.step(action)
        next_state = env.observe()[0]

        if env.inTerminalState():
            env.reset(mode=1)
            done = True

        buffer.add(state, action, reward, next_state, done)


def visualize_buffer_batch(agent, steps=100):
    """Visualize what the states in the buffer look like"""
    for i in range(steps):
        STATE, _, _, _, _ = agent.buffer.sample(1)
        STATE = to_numpy(STATE)
        img = plt.imshow(STATE[0], cmap='gray')
        plt.pause(0.1)
        plt.draw()
    plt.close()


def get_batch_with_every_state(device, env, noreset=False):
    """Returns a list with 1 of every possible state in the environment """
    temporary_buffer = ReplayBuffer(env.observe()[0].shape, env.action_space.shape[0], int(10000), device)
    fill_buffer(temporary_buffer, 5000, env, noreset=noreset)
    STATE, _, _, _, _ = temporary_buffer.sample(5000)
    STATE = to_numpy(STATE)
    every_state = []

    for i in range(5000):
        is_in = any(np.array_equal(STATE[i], j) for j in every_state)
        if not is_in:
            every_state.append(STATE[i])

    every_state = np.asarray(every_state)
    every_state = torch.from_numpy(every_state)

    del temporary_buffer

    return every_state


def get_same_agent_states(env, copy_env_for_plot=None, num_maps=4, pos=[5, 5]):
    """Returns a list with 1 of every possible state in the environment """
    maps = dict()

    for i in range(num_maps):
        maps['%s' % (i+1)] = []
        env.create_map(same_agent_pos=True, agent_pos=pos)
        rewards = copy.deepcopy(env._pos_rewards)
        if copy_env_for_plot is not None:
            copy_env_for_plot = copy.deepcopy(env)

        STATE = env.observe()
        maps['%s' % (i+1)].append(STATE)
        for j in range(10000):
            action = env.actions[r.randrange(env.num_actions)]
            env.step(action)
            STATE = env.observe()
            is_in = any(np.array_equal(STATE, j) for j in maps['%s' % (i+1)])
            if not is_in:
                maps['%s' % (i+1)].append(STATE)
                # Debugging!!!:
                # img = plt.imshow(STATE[0], cmap='gray')
                # plt.pause(0.05)
                # plt.draw()
            if env.inTerminalState:
                env._episode_steps=0
                env._agent_pos = [5, 5]
                env._pos_rewards = copy.deepcopy(rewards)

        maps['%s' % (i+1)] = np.asarray(maps['%s' % (i+1)])
        maps['%s' % (i+1)] = torch.from_numpy(maps['%s' % (i+1)])

    # TODO check if this works with the print function etc.
    return maps


def get_run_directory_fourmaze(args, agent):

    run_name = str(agent.name)+ \
                    'lr_encoder'+str(args.lr_encoder) + \
                    'lr_sa'+str(args.lr_sa) + \
                    'sa_scaler'+str(args.sa_scaler) + \
                    'entropy_scaler'+str(args.entropy_scaler) + \
                    'prediction_delta'+str(args.delta) + \
                    'iterations'+str(args.iterations)

    run_directory = os.path.join(args.folder_directory, run_name)

    return run_directory


def get_run_directory_multimaze_modes(args, agent):

    folder_directory = args.folder_directory
    run_name = str(agent.name)+ \
                    'lr_encoder'+str(args.lr_encoder) + \
                    'lr_sa'+str(args.lr_sa) + \
                    'batch_size'+str(args.batch_size) + \
                    'sa_scaler'+str(args.sa_scaler) + \
                    'entropy_scaler'+str(args.entropy_scaler) + \
               'mode_' + str(args.mode) + \
                'depth' + str(args.depth) + \
               'lr_reward' + str(args.lr_reward) + \
               'lr_discount' + str(args.lr_discount) + \
               'eps_start' + str(args.eps_start) + \
               'randbuffer_only' + str(args.randombuffer_only) + \
               'iterations'+str(args.iterations)

    run_directory = os.path.join(folder_directory, run_name)
    return run_directory


def get_run_directory_multimaze(args, agent):

    folder_directory = args.folder_directory
    run_name = str(agent.name)+ \
                    'lr_encoder'+str(args.lr_encoder) + \
                    'lr_sa'+str(args.lr_sa) + \
                    'batch_size'+str(args.batch_size) + \
                    'sa_scaler'+str(args.sa_scaler) + \
                    'entropy_scaler'+str(args.entropy_scaler) + \
               'iterations'+str(args.iterations)

    run_directory = os.path.join(folder_directory, run_name)
    return run_directory


def get_run_directory_catcher(args, agent):

    folder_directory = args.folder_directory
    run_name = str(agent.name)+ \
                    'lr_encoder'+str(args.lr_encoder) + \
                    'lr_sa'+str(args.lr_sa) + \
                    'batch_size'+str(args.batch_size) + \
               'adversarial' + str(args.adversarial) + \
               'sa_scaler'+str(args.sa_scaler) + \
                    'entropy_scaler'+str(args.entropy_scaler) + \
               'iterations'+str(args.iterations)

    run_directory = os.path.join(folder_directory, run_name)
    return run_directory


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def check_run_directory(run_directory):

    if path.exists(run_directory):
        ints = np.arange(1, 999)
        i = 0
        new_directory = str(run_directory)
        while path.exists(new_directory):
            new_directory = run_directory +'_'+str(ints[i])
            i += 1
        return new_directory
    else:
        return run_directory


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.asarray(smoothed)


def check_amount_of_possible_mazes(maze_resets=1e6, high_dim=True, only_mazes=False):
    """Returns a list with 1 of every possible state in the environment """
    list_of_mazes = []
    list_of_unique_mazes = []
    environment = Maze(np.random.RandomState(123456), higher_dim_obs=high_dim, map_type='path_finding', maze_size=8, random_start=False)
    environment.create_map()
    for i in range(maze_resets):
        environment.reset(mode=1)
        list_of_mazes.append(environment.observe()[0])
        if i % 5000000 ==0:
            unique_mazes = np.unique(list_of_mazes, axis=0)
            print("The amount of collected mazes is:", i)
            print("The amount of unique mazes is:", len(unique_mazes))


        if len(list_of_mazes) % 10000==0:
            print("mazes collected", len(list_of_mazes))
    if only_mazes:
        return list_of_mazes

    for i in range(maze_resets):
        is_in = any(np.array_equal(list_of_mazes[i], j) for j in list_of_unique_mazes)
        if not is_in:
            list_of_unique_mazes.append(list_of_mazes[i])
            if len(list_of_unique_mazes) % 10000 == 0:
                print("unique mazes collected", len(list_of_unique_mazes))

    unique_mazes = np.asarray(list_of_unique_mazes)
    num = len(unique_mazes)
    return unique_mazes, num
