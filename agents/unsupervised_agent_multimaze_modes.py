import torch.optim
from environments.maze_env import Maze
import numpy as np
import random as r
from networks import EncoderDMC_8x8wall, TransitionModel, InversePredictionModel, DQNmodel
from utils import get_same_agent_states, to_numpy
from replaybuffer import ReplayBuffer
import torch.nn as nn
import torch.nn.functional as F
from losses import compute_entropy_loss_featuremaps_randompixels, compute_DQN_loss_Hasselt, compute_entropy_loss_single_trajectory, compute_entropy_loss_multiple_trajectories


class Agent_Modes_Pathfinding:

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
            elif args.GPU == 99:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            print('WARNING: No GPU detected, running script on CPU only!!!')

        self.name = 'modes_path_finding'

        self.env_multimaze = env_multimaze
        self.eval_env = Maze(np.random.RandomState(123456), higher_dim_obs=True, maze_size=8, map_type='path_finding', random_start=False)
        self.states_same_agent = get_same_agent_states(self.eval_env, num_maps=4, pos=[5, 5])
        for i in range(4):
            self.states_same_agent['%s' % (i+1)] = self.states_same_agent['%s' % (i+1)].to(self.device).float()

        self.eps = args.eps_start
        self.eps_end = args.eps
        self.gamma = args.gamma
        self.tau = args.tau
        self.loss = nn.MSELoss()
        self.depth = args.depth
        self.planning = False
        self.breadth = args.breadth
        self.neuron_dim = args.neuron_dim
        self.feature_entropy_int = args.entropy_pixels

        self.agent_dim = 2
        self.batch_size = args.batch_size
        self.output = dict()
        self.iterations = 0

        self.q_loss = False

        self.entropy_scaler = args.entropy_scaler
        self.lr = args.lr_encoder
        self.lr_sa = args.lr_sa
        self.lr_reward = args.lr_reward
        self.lr_discount = args.lr_discount
        self.dqn_lr = args.lr_dqn
        self.onehot = args.onehot
        self.action_dim = 4 if self.onehot else 1

        self.sa_transition_scaler = args.sa_scaler
        self.s_transition_scaler = args.s_scaler
        self.rd_scaler = args.rd_scaler
        self.prediction_delta = args.delta

        # Encoders
        self.encoder = EncoderDMC_8x8wall(obs_channels=1, latent_dim=2, neuron_dim=args.neuron_dim).to(self.device)
        self.target_encoder = EncoderDMC_8x8wall(obs_channels=1, latent_dim=2, neuron_dim=args.neuron_dim).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        self.wall_dim = self.encoder.wall_size
        self.dqn_scaler = args.dqn_scaler

        # Mlp networks
        self.agent_forward_state_action = TransitionModel(self.agent_dim + self.wall_dim, action_dim=self.action_dim, scale=self.sa_transition_scaler,
                                                                       prediction_dim=self.agent_dim).to(self.device)
        self.wall_stationary_forward_state = TransitionModel(self.wall_dim, action_dim=0, scale=self.s_transition_scaler,
                                                                 prediction_dim=self.wall_dim).to(self.device)

        self.reward_predictor = TransitionModel(latent_dim=2+36, action_dim=self.action_dim, scale=self.rd_scaler,
                                                prediction_dim=1).to(self.device)
        self.discount_predictor = TransitionModel(latent_dim=2+36, action_dim=self.action_dim, scale=self.rd_scaler,
                                                prediction_dim=1).to(self.device)

        self.inverse_predictor = InversePredictionModel(input_dim=2*self.agent_dim+36, num_actions=4, scale=self.sa_transition_scaler, activation='tanh', final_activation=False).to(self.device)

        self.dqn_network = DQNmodel(self.wall_dim+self.agent_dim, scale=self.dqn_scaler, prediction_dim=4).to(self.device)
        self.target_network = DQNmodel(self.wall_dim+self.agent_dim, scale=self.dqn_scaler, prediction_dim=4).to(self.device)
        self.target_network.load_state_dict(self.dqn_network.state_dict())

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, )
        self.dqn_optimizer = torch.optim.Adam(self.dqn_network.parameters(), lr=self.dqn_lr)
        self.state_action_optimizer = torch.optim.Adam(self.agent_forward_state_action.parameters(), lr=self.lr_sa)
        self.stationary_state_optimizer = torch.optim.Adam(self.wall_stationary_forward_state.parameters(), lr=1e-5)
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=self.lr_reward)
        self.discount_predictor_optimizer = torch.optim.Adam(self.discount_predictor.parameters(), lr=self.lr_discount)
        self.inverse_predictor_optimizer = torch.optim.Adam(self.inverse_predictor.parameters(), lr=self.lr_sa)

        # Replay Buffer
        self.buffer = ReplayBuffer(self.env_multimaze.observe()[0].shape, env_multimaze.action_space.shape[0], int(max_memory_size), self.device)

    def unsupervised_learning(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Remove the gradients inside the model parameters
        self.encoder_optimizer.zero_grad()
        self.state_action_optimizer.zero_grad()
        self.stationary_state_optimizer.zero_grad()
        if self.planning:
            self.reward_predictor_optimizer.zero_grad()
            self.discount_predictor_optimizer.zero_grad()
        if self.q_loss:
            self.dqn_optimizer.zero_grad()
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=4)

        # Current latents (Z_t)
        agent_latent, wall_features= self.encoder(STATE)
        wall_flattened = wall_features.flatten(1)

        # Detached latents
        detached_agent_latent = agent_latent.clone().detach()
        detached_wall_flattened = wall_flattened.clone().detach()

        # Next latents (Z_t+1), from either an EMA target encoder or the actual encoder
        next_agent_latent, next_wall_features= self.encoder(NEXT_STATE)
        next_wall_features_flattened = next_wall_features.flatten(1)

        # Forward prediction state + action
        state_action_prediction = self.agent_forward_state_action(torch.cat((detached_agent_latent, detached_wall_flattened, ACTION), dim=1))

        # Forward prediction of the wall state
        state_prediction_wall = self.wall_stationary_forward_state(wall_flattened)

        if self.planning:
            reward_prediction = self.reward_predictor(torch.cat((detached_agent_latent, detached_wall_flattened, ACTION), dim=1))
            discount_prediction = self.discount_predictor(torch.cat((detached_agent_latent, detached_wall_flattened, ACTION), dim=1))
            discount_factors = self.gamma * (1 - DONE.int())
            reward_loss = nn.MSELoss()(reward_prediction, REWARD)
            discount_loss = nn.MSELoss()(discount_prediction, discount_factors)

        if self.prediction_delta:
            state_action_prediction += agent_latent
            state_prediction_wall += wall_flattened

        loss_state_action = self.loss(state_action_prediction, next_agent_latent)  # Detaching the next agent latent removes structure in the latent space!!!
        loss_wall_predictor = self.loss(state_prediction_wall, next_wall_features_flattened)

        # Entropy loss to avoid representation collapse
        loss_entropy = 0.5*compute_entropy_loss_featuremaps_randompixels(self)
        loss_entropy += 0.5*compute_entropy_loss_multiple_trajectories(self)
        # loss_entropy += compute_entropy_loss_single_trajectory(self)

        with torch.no_grad():
            next_agent_q , next_wall_q = self.target_encoder(NEXT_STATE)

        if self.q_loss:
            q_loss = compute_DQN_loss_Hasselt(self, agent_latent, wall_features, ACTION,
                                           REWARD, next_agent_q,
                                           next_wall_q, DONE)
        # The loss function
        loss = loss_entropy + loss_state_action + loss_wall_predictor
        if self.planning:
            loss+= reward_loss + discount_loss
        if self.q_loss:
            loss+= q_loss
        # Backprop the loss
        loss.backward()
        # Take optimization step
        self.encoder_optimizer.step()
        self.state_action_optimizer.step()
        self.stationary_state_optimizer.step()
        if self.planning:
            self.reward_predictor_optimizer.step()
            self.discount_predictor_optimizer.step()
        if self.q_loss:
            self.dqn_optimizer.step()

        # target network update
        for target_param, param in zip(self.target_network.parameters(), self.dqn_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)
            print(' The entropy loss is: ', to_numpy(loss_entropy))
            print(' The state-action prediction is: ', to_numpy(state_action_prediction[0]))
            print(' The actual state is: ', to_numpy(next_agent_latent[0]))
            print(' The state-action prediction LOSS is: ', to_numpy(loss_state_action))
            print(' The wall prediction LOSS is: ', to_numpy(loss_wall_predictor))
            if self.q_loss:
                print(' The Q-LOSS is', to_numpy(q_loss))
            # print(' The reward prediction LOSS is', to_numpy(loss_reward_predictor))

        # self.output['reward_loss'] = loss_reward_predictor
        self.output['controllable_loss'] = loss_state_action
        self.output['uncontrollable_loss'] = loss_wall_predictor
        self.output['entropy_loss'] = loss_entropy
        if self.planning:
            self.output['reward_loss'] = reward_loss
            self.output['discount_loss'] = discount_loss

        if self.q_loss:
            self.output['q_loss'] = q_loss

        self.iterations += 1

    def train_predictor(self, prediction=True):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Onehot encoding
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=4)

        self.state_action_optimizer.zero_grad()
        if self.planning:
            self.reward_predictor_optimizer.zero_grad()
            self.discount_predictor_optimizer.zero_grad()

        agent_latent, wall_features = self.encoder(STATE)
        next_agent_latent, next_wall_features= self.encoder(NEXT_STATE) #, detach='both')  # Always detach the target???

        wall_flattened = wall_features.flatten(1)
        detached_agent_latent = agent_latent.clone().detach()
        detached_wall_flattened = wall_flattened.clone().detach()

        if prediction:
            state_action_prediction = self.agent_forward_state_action(torch.cat((detached_agent_latent, detached_wall_flattened, ACTION), dim=1))
            if self.prediction_delta:
                state_action_prediction += agent_latent
            loss_state_action = nn.MSELoss()(state_action_prediction, next_agent_latent)  # Detaching the next agent latent removes structure in the latent space!!!
            self.output['controllable_loss'] = loss_state_action

        if self.planning:
            reward_prediction = self.reward_predictor(torch.cat((detached_agent_latent, detached_wall_flattened, ACTION), dim=1))
            discount_prediction = self.discount_predictor(torch.cat((detached_agent_latent, detached_wall_flattened, ACTION), dim=1))
            discount_factors = self.gamma * (1 - DONE.int())
            reward_loss = nn.MSELoss()(reward_prediction, REWARD)
            discount_loss = nn.MSELoss()(discount_prediction, discount_factors)
            self.output['reward_loss'] = reward_loss
            self.output['discount_loss'] = discount_loss

        if prediction and (not self.planning):
            loss = loss_state_action
        elif prediction and self.planning:
            loss = loss_state_action + reward_loss + discount_loss
        elif (not prediction) and self.planning:
            loss = reward_loss + discount_loss

        loss.backward()
        self.state_action_optimizer.step()
        if self.planning:
            self.discount_predictor_optimizer.step()
            self.reward_predictor_optimizer.step()

        self.iterations+=1
        if self.iterations % 500 == 0 and prediction:
            print(' The state-action prediction LOSS is: ', to_numpy(loss_state_action))

    def learn_DQN(self, encoder_updates):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Remove the gradients inside the model parameters
        self.dqn_optimizer.zero_grad()
        if encoder_updates:
            self.encoder_optimizer.zero_grad()
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=4)

        agent_latent, wall_features = self.encoder(STATE)

        with torch.no_grad():
            next_agent_latent, next_wall_features = self.encoder(NEXT_STATE)

        # TODO: Ablation, detach or no detach of the next Q-value?

        # next_agent_latent, next_wall_features = self.encoder(NEXT_STATE)

        q_loss = compute_DQN_loss_Hasselt(self, agent_latent, wall_features, ACTION,
                                       REWARD, next_agent_latent,
                                       next_wall_features, DONE)

        total_loss = q_loss

        # Backprop the loss
        total_loss.backward()
        # Take optimization step
        self.dqn_optimizer.step()
        if encoder_updates:
            self.encoder_optimizer.step()

        # target network update
        for target_param, param in zip(self.target_network.parameters(), self.dqn_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)
            print(' The Q-LOSS is', to_numpy(q_loss))

        self.output['q_loss'] = q_loss

    def get_action(self, controllable_latent, uncontrollable_features):

        if np.random.rand() < self.eps:
            return self.env_multimaze.actions[r.randrange(4)]
        elif self.planning:
            with torch.no_grad():
                if self.depth ==3:
                    action = self.get_action_with_planning_d3_correct(controllable_latent, uncontrollable_features)
                    return action
        elif not self.planning:
            with torch.no_grad():
                q_vals = self.dqn_network(controllable_latent, uncontrollable_features)
            action = np.argmax(to_numpy(q_vals))
            return action

    def run_agent(self, unsupervised=False, encoder_updates=False):

        done = False
        state = self.env_multimaze.observe()[0]
        controllable_latent, uncontrollable_features = self.encoder(torch.as_tensor(state).unsqueeze(0).unsqueeze(0).float().to(self.device))
        action = self.get_action(controllable_latent=controllable_latent, uncontrollable_features=uncontrollable_features)
        reward = self.env_multimaze.step(action)
        next_state = self.env_multimaze.observe()[0]

        if self.env_multimaze.inTerminalState():
            self.env_multimaze.reset(1)
            done = True
        self.buffer.add(state, action, reward, next_state, done)

        if unsupervised:
            self.unsupervised_learning()
        elif not unsupervised:
            self.learn_DQN(encoder_updates)

        if self.iterations % 10000 == 0:
            if self.iterations ==0:
                self.evaluate(eval_episodes=5)
            else:
                self.evaluate()

        self.eps = max(self.eps_end, self.eps - 0.8/50000)

    def evaluate(self, eval_episodes=100, give_value=False):

        self.eval_env.reset(1)
        average_reward = []
        Average_reward = 0

        for i in range(eval_episodes):
            reward = []
            done=False
            while not done:

                state = self.eval_env.observe()[0]
                controllable_latent, wall_features = self.encoder(torch.as_tensor(state).unsqueeze(0).unsqueeze(0).float().to(self.device))
                action = self.get_action(controllable_latent=controllable_latent, uncontrollable_features=wall_features)
                reward_t = self.eval_env.step(action, dont_take_reward=False)
                reward.append(reward_t)

                if self.eval_env.inTerminalState():
                    self.eval_env.reset(1)
                    done = True
                    reward = sum(reward)
                    average_reward.append(reward)

                # if self.iterations >= 390000:
                #     img = plt.imshow(state, cmap='gray')
                #     plt.pause(0.05)
                #     plt.draw()

        Average_reward += sum(average_reward)/len(average_reward)

        self.output['average_reward'] = torch.as_tensor(Average_reward)

        print('The AVERAGE REWARD is:', Average_reward)

        if give_value:
            return Average_reward

    def inverse_learning(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # Remove the gradients inside the model parameters
        self.encoder_optimizer.zero_grad()
        self.inverse_predictor_optimizer.zero_grad()
        self.stationary_state_optimizer.zero_grad()

        if self.q_loss:
            self.dqn_optimizer.zero_grad()
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

        # Forward prediction of the wall state
        state_prediction_wall = self.wall_stationary_forward_state(wall_flattened)

        if self.prediction_delta:
            state_prediction_wall += wall_flattened

        # Forward prediction state + action
        action_prediction = self.inverse_predictor(torch.cat((agent_latent, next_agent_latent, detached_wall_flattened), dim=1))
        loss_inverse_prediction = nn.CrossEntropyLoss()(action_prediction, torch.argmax(ACTION, dim=1))
        loss_wall_predictor = self.loss(state_prediction_wall, next_wall_features_flattened)

        # Entropy loss to avoid representation collapse
        loss_entropy = 0.5*compute_entropy_loss_featuremaps_randompixels(self)
        loss_entropy += 0.5*compute_entropy_loss_multiple_trajectories(self)

        with torch.no_grad():
            next_agent_q , next_wall_q = self.target_encoder(NEXT_STATE)

        if self.q_loss:
            q_loss = compute_DQN_loss_Hasselt(self, agent_latent, wall_features, ACTION,
                                           REWARD, next_agent_q,
                                           next_wall_q, DONE)
        # The loss function
        loss = loss_entropy + loss_inverse_prediction + loss_wall_predictor
        if self.q_loss:
            loss+= q_loss

        # Backprop the loss
        loss.backward()
        # Take optimization step
        self.encoder_optimizer.step()
        self.inverse_predictor_optimizer.step()
        self.stationary_state_optimizer.step()

        if self.q_loss:
            self.dqn_optimizer.step()
        # target network update
        for target_param, param in zip(self.target_network.parameters(), self.dqn_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)
            print(' The entropy loss is: ', to_numpy(loss_entropy))
            print(' The inverse prediction is: ', to_numpy(torch.argmax(action_prediction[0])))
            print(' The actual action is: ', to_numpy(torch.argmax(ACTION[0])))
            print(' The inverse prediction LOSS is: ', to_numpy(loss_inverse_prediction))
            print(' The wall prediction LOSS is: ', to_numpy(loss_wall_predictor))
            if self.q_loss:
                print(' The Q-LOSS is', to_numpy(q_loss))

        self.output['inverse_loss'] = loss_inverse_prediction
        self.output['uncontrollable_loss'] = loss_wall_predictor
        self.output['entropy_loss'] = loss_entropy

        if self.q_loss:
            self.output['q_loss'] = q_loss

        self.iterations += 1

    def get_action_with_planning_d3_correct(self, controllable_latent, uncontrollable_features):
        
        # Basic planning algorithm. Not the fastest, but easy to debug.

        static_features = uncontrollable_features
        preferred_action = 0
        highest_cumulative_Q = torch.tensor(-9999)

        with torch.no_grad():
            q_vals1 = self.dqn_network(controllable_latent, static_features)
            top_actions1 = torch.topk(q_vals1, 4).indices.view(4, 1)
            # Look at the top actions beginning from the initial state
            for action1 in top_actions1:
                # Take the top action, and get the resulting state after each action, d=1
                if self.onehot:
                    action1 = F.one_hot(action1, num_classes=4).squeeze(0)
                State1 = controllable_latent + self.agent_forward_state_action(torch.cat((controllable_latent, static_features.flatten(1), action1.unsqueeze(0)), dim=1))
                Reward1 = self.reward_predictor(torch.cat((controllable_latent, static_features.flatten(1), action1.unsqueeze(0)), dim=1))
                Discount1 = self.discount_predictor(torch.cat((controllable_latent, static_features.flatten(1), action1.unsqueeze(0)), dim=1))

                # Find the q-values at the new state
                q_vals2 = self.dqn_network(State1, static_features)
                top_actions2 = torch.topk(q_vals2, self.breadth).indices.view(self.breadth, 1)
                # Start the search from each state in depth 2
                for action2 in top_actions2:
                    if self.onehot:
                        action2 = F.one_hot(action2, num_classes=4).squeeze(0)
                    State2 = State1 + self.agent_forward_state_action(torch.cat((State1, static_features.flatten(1), action2.unsqueeze(0)), dim=1))
                    Reward2 = self.reward_predictor(torch.cat((State1, static_features.flatten(1), action2.unsqueeze(0)), dim=1))
                    Discount2 = self.discount_predictor(torch.cat((State1, static_features.flatten(1), action2.unsqueeze(0)), dim=1))
                    q_vals3 = self.dqn_network(State2, static_features)
                    top_actions3 = torch.topk(q_vals3, self.breadth).indices.view(self.breadth, 1)
                    for action3 in top_actions3:
                        if self.onehot:
                            action3 = F.one_hot(action3, num_classes=4).squeeze(0)
                        State3 = State2 + self.agent_forward_state_action(torch.cat((State2, static_features.flatten(1), action3.unsqueeze(0)), dim=1))
                        Reward3 = self.reward_predictor(torch.cat((State2, static_features.flatten(1), action3.unsqueeze(0)), dim=1))
                        Discount3 = self.discount_predictor(torch.cat((State2, static_features.flatten(1), action3.unsqueeze(0)), dim=1))
                        q_vals4 = self.dqn_network(State3, static_features)
                        Q_d3 = Reward1 + Discount1*(Reward2 + Discount2*(Reward3 +Discount3*(torch.max(q_vals4))))
                        Q_d2 = Reward1 + Discount1*(Reward2 + Discount2*(torch.max(q_vals3)))
                        Q_d1 = Reward1 + Discount1*(torch.max(q_vals2))
                        Q_d0 = q_vals1[0][torch.argmax(action1)]
                        Value = Q_d0 + Q_d1 + Q_d2 + Q_d3

                        if Value >= highest_cumulative_Q:
                            highest_cumulative_Q = Value
                            preferred_action = action1

            return torch.argmax(preferred_action).cpu()
