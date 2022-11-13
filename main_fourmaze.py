
import os
from environments.maze_env import Maze
from agents.unsupervised_agent_fourmaze import Agent_Fourmaze
from utils import get_run_directory_fourmaze, check_run_directory, strtobool, fill_buffer
import numpy as np
import time
import argparse
from print_functions import print_graph_with_all_states_and_transitions_3D_fourmaze

if os.path.isdir(os.getcwd() + '/runs'):
    pass
else:
    os.mkdir(os.getcwd() + '/runs')


parser = argparse.ArgumentParser()
parser.add_argument('--run_description', type=str, default='test_fourmaze')
parser.add_argument('--iterations', type=int, default=50000)
parser.add_argument('--agent_dim', type=int, default=2)
parser.add_argument('--wall_dim', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--detach_forward', type=strtobool, default=True)
parser.add_argument('--lr_sa', type=float, default=1e-3)
parser.add_argument('--lr_s', type=float, default=5e-5)
parser.add_argument('--sa_scaler', type=float, default=4)
parser.add_argument('--lr_encoder', type=float, default=5e-5)
parser.add_argument('--entropy_scaler', type=int, default=15)
parser.add_argument('--format', type=str, default='png')
parser.add_argument('--detach_walls', type=strtobool, default=True)
parser.add_argument('--delta', type=strtobool, default=True)
parser.add_argument('--onehot', type=strtobool, default=True)
parser.add_argument('--seeds', type=int, default=5)
parser.add_argument('--showplot', type=strtobool, default=False)
parser.add_argument('--interval_iterations', type=int, default=250)
args = parser.parse_args()

project_directory = os.getcwd() + '/runs'
folder_directory = project_directory + '/' + args.run_description
folder_directory = check_run_directory(run_directory=folder_directory)
os.mkdir(folder_directory)
args.folder_directory = folder_directory

start_time = time.time()

higher_dim_bool = True
rng = np.random.RandomState(123456)
env1 = Maze(rng, higher_dim_obs=higher_dim_bool, map_type='no_walls', maze_size=8)
env2 = Maze(rng, higher_dim_obs=higher_dim_bool, map_type='simple_map', maze_size=8)
env3 = Maze(rng, higher_dim_obs=higher_dim_bool, map_type='simple_map2', maze_size=8)
env4 = Maze(rng, higher_dim_obs=higher_dim_bool, map_type='simple_map3', maze_size=8)

env1.create_map()
env2.create_map()
env3.create_map()
env4.create_map()

# Create the agent
agent = Agent_Fourmaze(env1, env2, env3, env4, args=args)

# Fill the replay buffer with an equal amount of samples from the four mazes
fill_buffer(agent.buffer, 5000, env1)
fill_buffer(agent.buffer, 5000, env2)
fill_buffer(agent.buffer, 5000, env3)
fill_buffer(agent.buffer, 5000, env4)

# Make the directory specific for this environment and set of hyperparameters
run_directory = get_run_directory_fourmaze(args, agent)
run_directory = check_run_directory(run_directory=run_directory)
os.mkdir(run_directory)

# Main training loop
for i in range(args.iterations + 500):
    if agent.iterations % args.interval_iterations == 0:
        print_graph_with_all_states_and_transitions_3D_fourmaze(agent, agent.device, agent.encoder,
                                                                agent.agent_forward_state_action, agent.every_state_env1,
                                                                agent.every_state_env2, agent.every_state_env3,
                                                                agent.every_state_env4, delta=agent.prediction_delta,
                                                                args=args, run_directory=run_directory)
    # Train the agent for an iteration
    agent.mlp_learn()


#TODO GitHub Token: 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
