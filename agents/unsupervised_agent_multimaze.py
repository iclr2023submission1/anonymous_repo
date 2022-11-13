import torch.nn.functional as F
import torch.nn as nn
import torch
from environments.maze_env import Maze
from networks import EncoderDMC_8x8wall, TransitionModel
from utils import get_same_agent_states, to_numpy
from replaybuffer import ReplayBuffer
from losses import compute_entropy_loss_featuremaps_randompixels, compute_entropy_loss_single_trajectory, compute_entropy_loss_multiple_trajectories
import numpy as np


class Agent_Multimaze_Pathfinding:

    def __init__(self, env_multimaze, max_memory_size=1e6, args=None):

        if torch.cuda.is_available():
            if args.GPU == 0:
                self.device = 'cuda:0'
            elif args.GPU == 1:
                self.device = 'cuda:1'
            elif args.GPU == 2:
                self.device = 'cuda:2'
            elif args.GPU == 3:
                self.device = 'cuda:3'
            else:
                self.device = 'cuda'
        else:
            self.device = 'cpu'
            print('WARNING: No GPU detected, running script on CPU only!!!')

        self.name = 'multimaze_path_finding'

        self.env_multimaze = env_multimaze
        self.eval_env = Maze(np.random.RandomState(123456), higher_dim_obs=True, maze_size=8, map_type='path_finding', random_start=False)

        # Make a dictionary with four random environments with all possible agent states
        self.states_same_agent = get_same_agent_states(self.eval_env, num_maps=4, pos=[5, 5])
        for i in range(4):
            self.states_same_agent['%s' % (i+1)] = self.states_same_agent['%s' % (i+1)].to(self.device).float()

        self.loss = nn.MSELoss()
        self.agent_dim = 2
        self.batch_size = args.batch_size
        self.output = dict()
        self.iterations = 0
        self.adversarial = args.adversarial
        self.feature_entropy_int = args.entropy_pixels

        self.entropy_scaler = args.entropy_scaler

        self.lr = args.lr_encoder
        self.lr_sa = args.lr_sa
        self.lr_s = args.lr_s

        self.detach_walls = args.detach_walls
        self.onehot = args.onehot
        self.action_dim = 4 if self.onehot else 1

        self.stop_representation = args.stop_representation
        self.sa_transition_scaler = args.sa_scaler
        self.s_transition_scaler = args.s_scaler
        self.prediction_delta = args.delta

        # Encoder
        self.encoder = EncoderDMC_8x8wall(obs_channels=1, latent_dim=self.agent_dim, neuron_dim=args.neuron_dim).to(self.device)

        self.wall_dim = self.encoder.wall_size

        # Mlp Prediction Functions
        self.agent_forward_state_action = TransitionModel(self.agent_dim + self.wall_dim, action_dim=self.action_dim, scale=self.sa_transition_scaler,
                                                                       prediction_dim=self.agent_dim).to(self.device)
        self.wall_stationary_forward_state = TransitionModel(self.wall_dim, action_dim=0, scale=self.s_transition_scaler,
                                                                 prediction_dim=self.wall_dim).to(self.device)
        self.adversarial_predictor = TransitionModel(self.agent_dim, action_dim=0, scale=self.sa_transition_scaler, prediction_dim=self.wall_dim).to(self.device)

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, )
        self.state_action_optimizer = torch.optim.Adam(self.agent_forward_state_action.parameters(), lr=self.lr_sa)
        self.stationary_state_optimizer = torch.optim.Adam(self.wall_stationary_forward_state.parameters(), lr=self.lr_s)
        self.adversarial_predictor_optimizer = torch.optim.Adam(self.adversarial_predictor.parameters(), lr=1e-3)

        # Replay Buffer
        self.buffer = ReplayBuffer(self.env_multimaze.observe()[0].shape, env_multimaze.action_space.shape[0], int(max_memory_size), self.device)

    def mlp_learn(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Remove the gradients inside the model parameters
        self.encoder_optimizer.zero_grad()
        self.state_action_optimizer.zero_grad()
        self.stationary_state_optimizer.zero_grad()
        if self.adversarial:
            self.adversarial_predictor_optimizer.zero_grad()

        # One-hot action encodings
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=4)

        # Current latents (Z_t)
        agent_latent, wall_features= self.encoder(STATE)
        wall_flattened = wall_features.flatten(1)

        # Detached latents
        detached_agent_latent = agent_latent.clone().detach()
        detached_wall_flattened = wall_flattened.clone().detach()

        # Next latents (Z_t+1)
        next_agent_latent, next_wall_features= self.encoder(NEXT_STATE)
        next_wall_features_flattened = next_wall_features.flatten(1)

        # Forward prediction state + action
        state_action_prediction = self.agent_forward_state_action(torch.cat((detached_agent_latent,
                                                                             detached_wall_flattened if self.detach_walls
                                                                             else wall_flattened, ACTION), dim=1))

        # Forward prediction of the wall state
        state_prediction_wall = self.wall_stationary_forward_state(wall_flattened)

        # Residual prediction
        if self.prediction_delta:
            state_action_prediction += agent_latent
            state_prediction_wall += wall_flattened

        # Optional Adversarial_loss
        if self.adversarial:
            self.adversarial_predictor_optimizer.zero_grad()
            _, adversarial_wall_features = self.encoder(STATE, detach='base')
            adversarial_prediction = self.adversarial_predictor(agent_latent)
            adversarial_loss = nn.MSELoss()(adversarial_prediction, adversarial_wall_features.flatten(1))
            adversarial_loss.backward(retain_graph=True)
            self.adversarial_predictor_optimizer.step()
            for param in self.encoder.parameters():
                param.grad *= -1

        # Forward prediction losses
        loss_state_action = self.loss(state_action_prediction, next_agent_latent)
        loss_wall_predictor = self.loss(state_prediction_wall, next_wall_features_flattened)

        # Entropy loss to avoid representation collapse
        loss_entropy = 0.5*compute_entropy_loss_featuremaps_randompixels(self)
        loss_entropy += 0.5*compute_entropy_loss_multiple_trajectories(self)

        # The loss function
        loss = loss_entropy + loss_wall_predictor + loss_state_action

        # Backprop the loss
        loss.backward()

        # Take optimization step
        self.encoder_optimizer.step()
        self.state_action_optimizer.step()
        self.stationary_state_optimizer.step()

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)
            print(' The entropy loss is: ', to_numpy(loss_entropy))
            print(' The state-action prediction is: ', to_numpy(state_action_prediction[0]))
            print(' The actual state is: ', to_numpy(next_agent_latent[0]))
            print(' The state-action prediction LOSS is: ', to_numpy(loss_state_action))
            print(' The wall prediction LOSS is: ', to_numpy(loss_wall_predictor))
            if self.adversarial:
                print(' The ADVERSARIAL prediction LOSS is: ', to_numpy(adversarial_loss))

        self.iterations += 1

    def train_predictor_only(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Remove the gradients inside the model parameters
        self.state_action_optimizer.zero_grad()

        # One-hot action encodings
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=4)

        # Current latents (Z_t)
        agent_latent, wall_features = self.encoder(STATE)
        wall_flattened = wall_features.flatten(1)

        # Detached latents
        detached_agent_latent = agent_latent.clone().detach()
        detached_wall_flattened = wall_flattened.clone().detach()

        # Next latents (Z_t+1)
        next_agent_latent, next_wall_features= self.encoder(NEXT_STATE)

        # Forward prediction state + action
        state_action_prediction = self.agent_forward_state_action(torch.cat((detached_agent_latent, detached_wall_flattened if self.detach_walls else wall_flattened, ACTION), dim=1))

        # Residual prediction
        if self.prediction_delta:
            state_action_prediction += agent_latent

        # Forward prediction losses
        loss_state_action = nn.MSELoss()(state_action_prediction, next_agent_latent)
        loss_state_action.backward()

        # Take optimization steps
        self.state_action_optimizer.step()

        self.iterations+=1

        # Print the loss every 500 iterations
        if self.iterations % 500 == 0:
            print(' The state-action prediction LOSS is: ', to_numpy(loss_state_action))
