
import os
import numpy as np
from environments.catcher_env import Catcher
from agents.unsupervised_agent_catcher import Agent_Catcher
from utils import get_run_directory_catcher, strtobool, check_run_directory, fill_buffer
import argparse
from print_functions import print_featuremaps_halfagent_halfball

if os.path.isdir(os.getcwd() + '/runs'):
    pass
else:
    os.mkdir(os.getcwd() + '/runs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_description', type=str, default='test_catcher')
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--agent_dim', type=int, default=1)
    parser.add_argument('--lr_encoder', type=float, default=2e-5)
    parser.add_argument('--lr_sa', type=float, default=4e-5)
    parser.add_argument('--lr_s', type=float, default=1e-5)
    parser.add_argument('--lr_adv', type=float, default=1e-3)
    parser.add_argument('--adversarial', type=strtobool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sa_scaler', type=int, default=4)
    parser.add_argument('--s_scaler', type=int, default=4)
    parser.add_argument('--neuron_dim', type=int, default=200)
    parser.add_argument('--entropy_scaler', type=int, default=5)
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--detach_walls', type=strtobool, default=True)
    parser.add_argument('--delta', type=strtobool, default=True)
    parser.add_argument('--onehot', type=strtobool, default=True)
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

# Create environment
rng = np.random.RandomState(123456)
env = Catcher(rng, higher_dim_obs=True, reverse=False, step_size=2, height=15, width=15)
agent = Agent_Catcher(env, args)

run_directory = get_run_directory_catcher(args, agent)
run_directory = check_run_directory(run_directory=run_directory)
os.mkdir(run_directory)

# Fill the replay buffer
fill_buffer(agent.buffer, 25000, env)

# Training loop
for i in range(args.iterations + 500):
    if agent.iterations % args.interval_iterations == 0:
        print_featuremaps_halfagent_halfball(agent, args=args, run_directory=run_directory,
                                                 width_inches=5, h_inches=10,
                                             extra_name='adversarial' if agent.adversarial else 'normal')
    # Train for an iteration
    agent.mlp_learn()
