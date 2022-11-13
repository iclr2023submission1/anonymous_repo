
import torch
from print_functions import print_graph_with_same_agent_states
from agents.unsupervised_agent_multimaze import Agent_Multimaze_Pathfinding
from utils import strtobool, check_run_directory, get_run_directory_multimaze, fill_buffer, visualize_buffer_batch
import time
import numpy as np
import argparse
from environments.maze_env import Maze
import os

if os.path.isdir(os.getcwd() + '/runs'):
    pass
else:
    os.mkdir(os.getcwd() + '/runs')

parser = argparse.ArgumentParser()
parser.add_argument('--run_description', type=str, default='test_multimaze')
parser.add_argument('--iterations', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--stop_representation', type=int, default=50000)
parser.add_argument('--detach_forward', type=strtobool, default=True)
parser.add_argument('--lr_sa', type=float, default=4e-5)
parser.add_argument('--lr_s', type=float, default=1e-5)
parser.add_argument('--lr_encoder', type=float, default=2e-5)
parser.add_argument('--neuron_dim', type=int, default=200)
parser.add_argument('--entropy_pixels', type=int, default=15)
parser.add_argument('--sa_scaler', type=float, default=16)
parser.add_argument('--s_scaler', type=int, default=4)
parser.add_argument('--entropy_scaler', type=int, default=10)
parser.add_argument('--adversarial', type=strtobool, default=False)
parser.add_argument('--format', type=str, default='png')
parser.add_argument('--detach_walls', type=strtobool, default=True)
parser.add_argument('--delta', type=strtobool, default=True)
parser.add_argument('--onehot', type=strtobool, default=True)
parser.add_argument('--seeds', type=int, default=20)
parser.add_argument('--GPU', type=int, default=0)
parser.add_argument('--showplot', type=strtobool, default=False)
parser.add_argument('--interval_iterations', type=int, default=5000)
parser.add_argument('--run_directory', type=str, default='')
args = parser.parse_args()

# Create run directory
project_directory = os.getcwd() + '/runs'
folder_directory = project_directory +'/'+ args.run_description
folder_directory = check_run_directory(run_directory=folder_directory)
os.mkdir(folder_directory)
args.folder_directory = folder_directory

for i in range(args.seeds):

    start_time = time.time()

    rng = np.random.RandomState(123456)
    random_start = True
    env_multimaze = Maze(rng, higher_dim_obs=True, map_type='path_finding', maze_size=8, random_start=random_start)

    env_multimaze.create_map()

    # Create the agent
    agent = Agent_Multimaze_Pathfinding(env_multimaze, args=args)

    # Make the directory specific for this environment and set of hyperparameters
    run_directory = get_run_directory_multimaze(args, agent)
    run_directory = check_run_directory(run_directory=run_directory)
    os.mkdir(run_directory)

    # Fill the replay buffer
    fill_buffer(agent.buffer, 50000, agent.env_multimaze)

    ## Uncomment for visual and debugging purposes only
    # visualize_buffer_batch(agent)

    for j in range(args.iterations+500):
        if agent.iterations !=0 and agent.iterations %args.interval_iterations==0:

            # Save the representation
            print_graph_with_same_agent_states(agent, args=args, run_directory=run_directory)

            # Save the encoder
            torch.save(agent.encoder.state_dict(), run_directory+'/encoder.pt')

            # Save the predictor
            torch.save(agent.agent_forward_state_action.state_dict(), run_directory+'/forward_predictor.pt')

        # Train the encoder / predictors
        if agent.iterations <= args.stop_representation:
            agent.mlp_learn()
        else:
            agent.train_predictor_only()
