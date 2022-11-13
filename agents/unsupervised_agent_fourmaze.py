from networks import EncoderDMC, TransitionModel
from utils import get_batch_with_every_state, to_numpy
from replaybuffer import ReplayBuffer
from losses import compute_entropy_loss
import torch
import torch.nn.functional as F
import torch.nn as nn


class Agent_Fourmaze:

    def __init__(self, env1, env2, env3, env4, args=None):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.name = 'fourmaze_3states'
        self.env1 = env1
        self.env2 = env2
        self.env3 = env3
        self.env4 = env4

        # Gather every state in every environment, for plotting purposes
        self.every_state_env1 = get_batch_with_every_state(self.device, self.env1).to(self.device)
        self.every_state_env2 = get_batch_with_every_state(self.device, self.env2).to(self.device)
        self.every_state_env3 = get_batch_with_every_state(self.device, self.env3).to(self.device)
        self.every_state_env4 = get_batch_with_every_state(self.device, self.env4).to(self.device)

        self.agent_dim = args.agent_dim
        self.wall_dim = args.wall_dim
        self.batch_size = args.batch_size

        self.entropy_scaler = args.entropy_scaler
        self.detach_walls = args.detach_walls

        self.lr = args.lr_encoder
        self.lr_sa = args.lr_sa
        self.lr_s = args.lr_s

        self.onehot = args.onehot
        self.action_dim = 4 if self.onehot else 1

        self.sa_transition_scaler = args.sa_scaler
        self.prediction_delta = args.delta

        self.output = dict()
        self.iterations = 0

        # Convolutional encoder
        self.encoder = EncoderDMC(latent_dim=self.agent_dim+self.wall_dim).to(self.device)

        # Mlp Prediction Functions
        self.agent_forward_state_action = TransitionModel(self.agent_dim+self.wall_dim, action_dim=self.action_dim,
                                                          scale=self.sa_transition_scaler,
                                                               prediction_dim=self.agent_dim).to(self.device)
        self.wall_stationary_forward_state = TransitionModel(self.wall_dim, action_dim=0, scale=self.sa_transition_scaler,
                                                             prediction_dim=self.wall_dim).to(self.device)

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, )
        self.state_action_optimizer = torch.optim.Adam(self.agent_forward_state_action.parameters(), lr=self.lr_sa)
        self.stationary_state_optimizer = torch.optim.Adam(self.wall_stationary_forward_state.parameters(), lr=self.lr_s)

        # Replay Buffer
        self.buffer = ReplayBuffer(self.env1.observe()[0].shape, env1.action_space.shape[0], int(1e6), self.device)

    def mlp_learn(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Remove the gradients inside the model parameters
        self.encoder_optimizer.zero_grad()
        self.state_action_optimizer.zero_grad()
        self.stationary_state_optimizer.zero_grad()

        # One-hot action encodings
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=self.action_dim)

        # Current latents (Z_t)
        full_latent = self.encoder(STATE)
        latent_agent = full_latent[:, 0:2]
        latent_wall = full_latent[:, 2].unsqueeze(1)
        latent_agent_detached = latent_agent.clone().detach()
        latent_wall_detached = latent_wall.clone().detach()

        # Next latents (Z_t+1)
        next_latent = self.encoder(NEXT_STATE)
        next_latent_agent = next_latent[:, 0:2]
        next_latent_wall = next_latent[:, 2].unsqueeze(1)

        # Forward prediction state + action
        state_action_prediction = self.agent_forward_state_action(torch.cat((latent_agent_detached,
                                                                             latent_wall_detached, ACTION), 1))

        # Forward prediction state
        state_prediction_wall = self.wall_stationary_forward_state(latent_wall)

        # Residual prediction
        if self.prediction_delta:
            state_action_prediction += latent_agent
            state_prediction_wall += latent_wall

        # Forward prediction losses
        loss_state_action = nn.MSELoss()(state_action_prediction, next_latent_agent)
        loss_wall_predictor = nn.MSELoss()(state_prediction_wall, next_latent_wall)

        # Entropy loss to avoid representation collapse
        loss_entropy = compute_entropy_loss(self)

        # The loss function
        loss = loss_entropy + loss_state_action + loss_wall_predictor

        # Backprop the loss
        loss.backward()

        # Take optimization steps
        self.encoder_optimizer.step()
        self.state_action_optimizer.step()
        self.stationary_state_optimizer.step()

        self.output['loss_ent'] = loss_entropy
        self.output['loss_sa'] = loss_state_action
        self.output['loss_wall_prediction'] = loss_wall_predictor

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)
            print(' The entropy loss is: ', to_numpy(loss_entropy))
            print(' The state-action prediction is: ', to_numpy(state_action_prediction[0]))
            print(' The actual state is: ', to_numpy(next_latent[0]))
            print(' The state-action prediction LOSS is: ', to_numpy(loss_state_action))
            print(' The wall prediction LOSS is: ', to_numpy(loss_wall_predictor))

        self.iterations += 1

