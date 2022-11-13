from networks import EncoderDMC_half_features_catcher, TransitionModel
from utils import to_numpy
import torch.nn.functional as F
import torch.nn as nn
from replaybuffer import ReplayBuffer
from losses import compute_entropy_loss_featuremaps_randompixels
import torch
import numpy as np


class Agent_Catcher:

    def __init__(self, env, args=None):

        if torch.cuda.is_available():
            if args.GPU == 0:
                self.device = 'cuda:0'
            elif args.GPU == 1:
                self.device = 'cuda:1'
            else:
                self.device = 'cuda'
        else:
            self.device = 'cpu'
            print('WARNING: No GPU detected, running script on CPU only!!!')

        self.name = 'catcher_twoagent'

        self.env = env
        self.batch_size = args.batch_size
        self.output = dict()
        self.onehot = args.onehot
        self.iterations = 0
        self.entropy_scaler = args.entropy_scaler
        self.detach_wall = args.detach_walls
        self.adversarial = args.adversarial
        self.lr = args.lr_encoder
        self.lr_sa = args.lr_sa
        self.lr_s = args.lr_s

        self.lr_adv = args.lr_adv
        self.feature_entropy_int = 15

        self.agent_transition_scaler = args.sa_scaler
        self.prediction_delta = args.delta

        self.agent_dim = args.agent_dim
        self.action_dim = 2 if self.onehot else 1

        self.encoder = EncoderDMC_half_features_catcher(1, self.agent_dim, neuron_dim=args.neuron_dim).to(self.device)

        test_state, test_feature = self.encoder(torch.from_numpy(env.observe()[0]).to(self.device).
                                                unsqueeze(0).unsqueeze(0).float())
        self.ball_dim = int(len(test_feature[0][0].flatten(0)))
        self.ball_grid = np.sqrt(self.ball_dim)

        # Mlp Forward Prediction Functions
        self.agent_forward_state_action = TransitionModel(self.agent_dim + self.ball_dim, action_dim=self.action_dim,
                                                          scale=self.agent_transition_scaler,
                                                               prediction_dim=self.agent_dim).to(self.device)
        self.wall_stationary_forward_state = TransitionModel(self.ball_dim, action_dim=0, scale=self.agent_transition_scaler,
                                                             prediction_dim=self.ball_dim).to(self.device)
        if self.adversarial:
            self.adversarial_predictor = TransitionModel(self.agent_dim, action_dim=0, scale=self.agent_transition_scaler,
                                                             prediction_dim=self.ball_dim).to(self.device)
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.state_action_optimizer = torch.optim.Adam(self.agent_forward_state_action.parameters(), lr=self.lr_sa)
        self.stationary_state_optimizer = torch.optim.Adam(self.wall_stationary_forward_state.parameters(), lr=self.lr_s)
        if self.adversarial:
            self.adversarial_predictor_optimizer = torch.optim.Adam(self.adversarial_predictor.parameters(), self.lr_adv)
        # Replay Buffer
        self.buffer = ReplayBuffer(self.env.observe()[0].shape, env.action_space.shape[0],
                                   capacity=int(1e6), device=self.device)

    def mlp_learn(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Remove the gradients inside the model parameters
        self.encoder_optimizer.zero_grad()
        self.state_action_optimizer.zero_grad()
        self.stationary_state_optimizer.zero_grad()
        if self.adversarial:
            self.adversarial_predictor_optimizer.zero_grad()
            _, adversarial_ball_features = self.encoder(STATE, detach='base')
            adversarial_ball_features = adversarial_ball_features.flatten(1)

        # One-hot action encodings
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=self.action_dim)

        # Current latents (Z_t)
        agent_state , ball_features = self.encoder(STATE)
        detached_ball_features = ball_features.clone().detach()
        detached_agent_state = agent_state.clone().detach()

        # Next latents (Z_t+1)
        next_agent_state, next_ball_features = self.encoder(NEXT_STATE)

        ball_features = ball_features.flatten(1)
        detached_ball_features = detached_ball_features.flatten(1)
        next_ball_features = next_ball_features.flatten(1)

        # Forward prediction state + action
        state_action_prediction = self.agent_forward_state_action(torch.cat((detached_agent_state,
                                                                             (detached_ball_features if self.detach_wall
                                                                              else ball_features), ACTION), 1))

        # Forward prediction state
        state_prediction_ball = self.wall_stationary_forward_state(ball_features)

        # Adversarial prediction
        if self.adversarial:
            adversarial_prediction = self.adversarial_predictor(agent_state)

        # Residual prediction
        if self.prediction_delta:
            state_action_prediction += agent_state
            state_prediction_ball += ball_features

        # Forward prediction losses
        loss_state_action = nn.MSELoss()(state_action_prediction, next_agent_state)
        loss_ball_predictor = nn.MSELoss()(state_prediction_ball, next_ball_features)

        # Entropy loss to avoid representation collapse
        loss_entropy = compute_entropy_loss_featuremaps_randompixels(self)

        # The loss function
        loss = loss_entropy + loss_state_action + loss_ball_predictor

        # Adversarial_loss
        if self.adversarial:
            adversarial_loss = nn.MSELoss()(adversarial_prediction, adversarial_ball_features)
            adversarial_loss.backward(retain_graph=True)
            for param in self.encoder.parameters():
                param.grad *= -1

        # Backprop the loss
        loss.backward()

        # Take optimization steps
        self.encoder_optimizer.step()
        self.state_action_optimizer.step()
        self.stationary_state_optimizer.step()
        if self.adversarial:
            self.adversarial_predictor_optimizer.step()

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)
            print(' The entropy loss is: ', to_numpy(loss_entropy))
            print(' The state-action prediction LOSS is: ', to_numpy(loss_state_action))
            print(' The ball prediction LOSS is: ', to_numpy(loss_ball_predictor))
            if self.adversarial:
                print(' The Adversarial BALL prediction LOSS is: ', to_numpy(adversarial_loss))

        self.iterations += 1
