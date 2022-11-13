
from utils import strtobool, check_run_directory, get_run_directory_multimaze, fill_buffer, to_numpy
from print_functions import print_graph_with_same_agent_states
import torch
import matplotlib.pyplot as plt
import numpy as np
from agents.unsupervised_agent_multimaze_modes import Agent_Modes_Pathfinding
import time
import argparse
from environments.maze_env import Maze
import os

if os.path.isdir(os.getcwd() + '/runs'):
    pass
else:
    os.mkdir(os.getcwd() + '/runs')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_description', type=str, default='test_multimaze_modes')
    parser.add_argument('--iterations', type=int, default='500000')
    parser.add_argument('--random_samples', type=int, default=50000)
    parser.add_argument('--neuron_dim', type=int, default=200)
    parser.add_argument('--lr_dqn', type=float, default=4e-5)
    parser.add_argument('--lr_encoder', type=float, default=2e-5)
    parser.add_argument('--lr_sa', type=float, default=4e-5)
    parser.add_argument('--lr_reward', type=float, default=5e-5)
    parser.add_argument('--lr_discount', type=float, default=5e-5)
    parser.add_argument('--breadth', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--entropy_pixels', type=int, default=15)
    parser.add_argument('--sa_scaler', type=int, default=16)
    parser.add_argument('--s_scaler', type=int, default=4)
    parser.add_argument('--rd_scaler', type=int, default=2)
    parser.add_argument('--dqn_scaler', type=int, default=16)
    parser.add_argument('--entropy_scaler', type=int, default=10)
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--q_loss', type=strtobool, default=False)
    parser.add_argument('--tau', type=float, default=0.02)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--eps_start', type=float, default=1)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--detach_walls', type=strtobool, default=True)
    parser.add_argument('--delta', type=strtobool, default=True)
    parser.add_argument('--onehot', type=strtobool, default=True)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--pretrain_iterations', type=int, default=50000)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--showplot', type=strtobool, default=False)
    parser.add_argument('--interval_iterations', type=int, default=25000)
    parser.add_argument('--run_directory', type=str, default='')
    parser.add_argument('--encoder_name', type=str, default='encoder.pt')
    parser.add_argument('--forward_name', type=str, default='forward_predictor.pt')
    args = parser.parse_args()

project_directory = os.getcwd() + '/runs'
folder_directory = project_directory +'/'+ args.run_description
folder_directory = check_run_directory(run_directory=folder_directory)
os.mkdir(folder_directory)
args.folder_directory = folder_directory

start_time = time.time()
reward = dict()
iterations = dict()

for s in range(args.seeds):

    reward['%s' % s] = []
    iterations['%s' % s] = []

    rng = np.random.RandomState(123456)
    env_multimaze = Maze(rng, higher_dim_obs=True, map_type='path_finding', maze_size=8, random_start=True)
    env_multimaze.create_map()

    # Create the agent
    agent = Agent_Modes_Pathfinding(env_multimaze, args=args)

    # Make the directory specific for this environment and set of hyperparameters
    run_directory = get_run_directory_multimaze(args, agent)
    run_directory = run_directory +'_'+args.mode
    run_directory = check_run_directory(run_directory=run_directory)
    os.mkdir(run_directory)

    fig = plt.figure()
    fig.set_size_inches(w=2, h=2)

    if args.mode == 'pretrain':

        # Fill the replay buffer
        fill_buffer(agent.buffer, args.random_samples, agent.env_multimaze)

        # visualize_buffer_batch(agent)

        for i in range(args.pretrain_iterations+1):
          if agent.iterations !=0 and agent.iterations % args.interval_iterations==0:
              print(time.time()-start_time, 'Seconds past since beginning of the script')
              print_graph_with_same_agent_states(agent,  args=args, run_directory=run_directory)
              torch.save(agent.encoder.state_dict(), run_directory+'/encoder.pt')   # Save the encoder for transfer learning
              torch.save(agent.agent_forward_state_action.state_dict(), run_directory+'/forward_predictor.pt')
              torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

          agent.q_loss = False
          agent.unsupervised_learning()

        lowest_reward = -10
        agent.env_multimaze.random_start = False

        for j in range(args.iterations+1):
            agent.run_agent(unsupervised=False, encoder_updates=False)
            agent.train_predictor()
            if j != 0 and j % 10000 == 0:
                print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory)
                torch.save(agent.encoder.state_dict(), run_directory + '/encoder.pt')
                torch.save(agent.agent_forward_state_action.state_dict(), run_directory + '/forward_predictor.pt')
                reward['%s' % s].append(to_numpy(agent.output['average_reward']))
                iterations['%s' % s].append(j)
                if agent.output['average_reward'] >= lowest_reward:
                    lowest_reward = agent.output['average_reward']
                    torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

    if args.mode == 'pretrain_saved_model':

        # Fill the replay buffer
        fill_buffer(agent.buffer, args.random_samples, agent.env_multimaze)
        # visualize_buffer_batch(agent)
        agent.encoder.load_state_dict(torch.load(os.getcwd() + '/saved_models/' + args.encoder_name, map_location='cuda:0'))
        agent.agent_forward_state_action.load_state_dict(torch.load(os.getcwd() + '/saved_models/' + args.forward_name, map_location='cuda:0'))

        lowest_reward = -10
        agent.env_multimaze.random_start = False

        for j in range(args.iterations+1):
            agent.run_agent(unsupervised=False, encoder_updates=False)
            agent.iterations += 1
            if j != 0 and j % 10000 == 0:
                print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory)
                reward['%s' % s].append(to_numpy(agent.output['average_reward']))
                iterations['%s' % s].append(j)
                if agent.output['average_reward'] >= lowest_reward:
                    lowest_reward = agent.output['average_reward']
                    torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

    if args.mode == 'end-to-end':

        # Fill the replay buffer
        fill_buffer(agent.buffer, args.random_samples, agent.env_multimaze)
        # visualize_buffer_batch(agent)
        lowest_reward = -10
        agent.env_multimaze.random_start = False

        for i in range(args.iterations+1):
          if agent.iterations !=0 and agent.iterations % 10000==0:
              print(time.time()-start_time, 'Seconds past since beginning of the script')
              print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory)
              torch.save(agent.encoder.state_dict(), run_directory+'/encoder.pt')   # Save the encoder for transfer learning
              torch.save(agent.agent_forward_state_action.state_dict(), run_directory+'/forward_predictor.pt')
              reward['%s' % s].append(to_numpy(agent.output['average_reward']))
              iterations['%s' % s].append(agent.iterations)
              if agent.output['average_reward'] >= lowest_reward:
                  lowest_reward = agent.output['average_reward']
                  torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

          agent.q_loss = True
          agent.run_agent(unsupervised=True, encoder_updates=False)

    if args.mode == 'dqn_only':

        # Fill the replay buffer
        fill_buffer(agent.buffer, args.random_samples, agent.env_multimaze)
        # visualize_buffer_batch(agent)

        lowest_reward = -10
        agent.env_multimaze.random_start = False

        for j in range(args.iterations +1):
            agent.run_agent(unsupervised=False, encoder_updates=True)
            agent.iterations += 1
            if j != 0 and j % 10000 == 0:
                print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory, blue_ordered=True, transitions=False)
                torch.save(agent.encoder.state_dict(), run_directory + '/encoder.pt')  # Save the encoder for transfer learning
                torch.save(agent.agent_forward_state_action.state_dict(), run_directory + '/forward_predictor.pt')
                reward['%s' % s].append(to_numpy(agent.output['average_reward']))
                iterations['%s' % s].append(j)
                if agent.output['average_reward'] >= lowest_reward:
                    lowest_reward = agent.output['average_reward']
                    torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

    if args.mode == 'pretrain_planning':

        # Fill the replay buffer
        fill_buffer(agent.buffer, args.random_samples, agent.env_multimaze)
        # visualize_buffer_batch(agent)
        agent.planning=True
        for i in range(args.pretrain_iterations+1):
          if agent.iterations !=0 and agent.iterations % 25000==0:
              print(time.time()-start_time, 'Seconds past since beginning of the script')
              print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory)
              torch.save(agent.encoder.state_dict(), run_directory+'/encoder.pt')   # Save the encoder for transfer learning
              torch.save(agent.agent_forward_state_action.state_dict(), run_directory+'/forward_predictor.pt')
              torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

          agent.q_loss = False
          agent.unsupervised_learning()

        lowest_reward = -10
        agent.env_multimaze.random_start = False

        for j in range(args.iterations+1):
            agent.run_agent(unsupervised=False, encoder_updates=False)
            agent.train_predictor(prediction=True)
            if j != 0 and j % 10000 == 0:
                print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory)
                reward['%s' % s].append(to_numpy(agent.output['average_reward']))
                iterations['%s' % s].append(j)
                if agent.output['average_reward'] >= lowest_reward:
                    lowest_reward = agent.output['average_reward']
                    torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

    if args.mode == 'pretrain_planning_saved_model':

        # Fill the replay buffer
        fill_buffer(agent.buffer, args.random_samples, agent.env_multimaze)
        # visualize_buffer_batch(agent)

        lowest_reward = -10
        agent.env_multimaze.random_start = False
        agent.encoder.load_state_dict(torch.load(os.getcwd() + '/saved_models/' + args.encoder_name, map_location='cuda:0'))
        agent.agent_forward_state_action.load_state_dict(torch.load(os.getcwd() + '/saved_models/' + args.forward_name, map_location='cuda:0'))
        agent.planning = True
        for j in range(args.iterations+1):
            agent.run_agent(unsupervised=False, encoder_updates=False)
            agent.train_predictor(prediction=False)
            if j != 0 and j % 10000 == 0:
                print_graph_with_same_agent_states(agent,  args=args, run_directory=run_directory)
                reward['%s' % s].append(to_numpy(agent.output['average_reward']))
                iterations['%s' % s].append(j)
                if agent.output['average_reward'] >= lowest_reward:
                    lowest_reward = agent.output['average_reward']
                    torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

    if args.mode == 'ablation_inverse':

        # Fill the replay buffer
        fill_buffer(agent.buffer, args.random_samples, agent.env_multimaze)
        # visualize_buffer_batch(agent)

        for i in range(args.pretrain_iterations+1):
          if agent.iterations !=0 and agent.iterations % args.interval_iterations==0:
              print(time.time()-start_time, 'Seconds past since beginning of the script')
              print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory, transitions=False, blue_ordered=True)
              torch.save(agent.encoder.state_dict(), run_directory+'/encoder.pt')   # Save the encoder for transfer learning
              torch.save(agent.agent_forward_state_action.state_dict(), run_directory+'/forward_predictor.pt')
              torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

          agent.q_loss = False
          agent.inverse_learning()

        lowest_reward = -10
        agent.env_multimaze.random_start = False

        for j in range(args.iterations+1):
            agent.run_agent(unsupervised=False, encoder_updates=False)
            agent.iterations += 1
            if j != 0 and j % 10000 == 0:
                print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory, transitions=False, blue_ordered=True)
                reward['%s' % s].append(to_numpy(agent.output['average_reward']))
                iterations['%s' % s].append(j)
                if agent.output['average_reward'] >= lowest_reward:
                    lowest_reward = agent.output['average_reward']
                    torch.save(agent.dqn_network.state_dict(), run_directory + '/dqn_network.pt')

fig = plt.figure()
fig.set_size_inches(w=5, h=5)

# Find Mean
sum_mean = 0
for p in range(args.seeds):
    reward['%s' % p] = np.array(reward['%s' % p])
    iterations['%s' % p] = np.array(iterations['%s' % p])
    sum_mean = sum_mean + reward['%s' % p]
mean = sum_mean/args.seeds

# Find std_dev
sum_distance = 0
distance = dict()
for p in range(args.seeds):
    distance['%s' % p] = np.absolute(reward['%s' % p] - mean)
    sum_distance = sum_distance + distance['%s' % p]
sum_distance_divided = sum_distance/args.seeds
std_dev = np.sqrt(sum_distance_divided)

# Plot with error bars
plt.plot(iterations['0'], mean)
plt.fill_between(iterations['0'], mean-std_dev, mean+std_dev, alpha=0.25)

# Save arrays
np.save(run_directory + '/' + args.mode +'_' + str(args.seeds) + 'seeds' + '_iterations' + '.npy', iterations['0'])
np.save(run_directory + '/' + args.mode +'_' + str(args.seeds) + 'seeds' + '_std_dev' + '.npy', std_dev)
np.save(run_directory + '/' + args.mode +'_' + str(args.seeds) + 'seeds' + '_mean' + '.npy', mean)

# plt.show()









