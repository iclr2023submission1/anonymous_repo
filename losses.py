import torch
import numpy as np
import random
import torch.nn.functional as F


def compute_entropy_loss(agent):

    random_states, _, _, _, _ = agent.buffer.sample(agent.batch_size)
    random_states = agent.encoder(random_states)
    random_states_rolled = torch.roll(random_states, 1, dims=0)
    difference_states = random_states-random_states_rolled

    # normal random states loss
    loss = torch.exp(-agent.entropy_scaler * torch.norm(difference_states, dim=1, p=2)).mean()

    return loss


def compute_entropy_loss_featuremaps_randompixels(agent):

    STATE, ACTION, REWARD, NEXT_STATE, DONE = agent.buffer.sample(agent.batch_size)

    random_states_agent, random_features_wall = agent.encoder(STATE)

    id = torch.tensor(random.sample(range(len(random_features_wall[0].flatten(0))), agent.feature_entropy_int))
    average_wall_features_array = random_features_wall.flatten(1)[:, id]

    random_states_agent_rolled = torch.roll(random_states_agent, 1, dims=0)
    average_wall_features_array_rolled = torch.roll(average_wall_features_array, 1, dims=0)

    concatenated_states = torch.cat((random_states_agent, average_wall_features_array), dim=1)
    concatenated_states_rolled = torch.cat((random_states_agent_rolled, average_wall_features_array_rolled), dim=1)
    difference_states = concatenated_states - concatenated_states_rolled
    loss = torch.exp(-agent.entropy_scaler * torch.norm(difference_states, dim=1, p=2)).mean()

    return loss


def compute_entropy_loss_single_trajectory(agent):

    trajectory = agent.buffer.sample_trajectory()[0]   # gets a tuple of states and features, so need to take the 0-th entry
    trajectory = trajectory
    encoded_trajectory = agent.encoder(trajectory)[0]
    # Roll the states uniformly random so we eventually get all combinations
    encoded_trajectory_rolled = torch.roll(encoded_trajectory, np.random.randint(low=1, high=len(encoded_trajectory-1)), dims=0)

    loss = torch.exp(-agent.entropy_scaler * torch.norm(encoded_trajectory_rolled - encoded_trajectory, dim=1, p=2)).mean()

    return loss


def compute_DQN_loss_Hasselt(agent, controllable_latent, uncontrollable_features, actions, rewards,
                   next_controllable_latent, next_uncontrollable_features, dones):

    # Change actions to long format for the gather function
    actions = actions.long()
    if agent.onehot:
        actions = torch.argmax(actions, dim=1).unsqueeze(1)
    # Compute the Q-value estimates corresponding to the actions in the batch
    Q = agent.dqn_network(controllable_latent, uncontrollable_features)
    next_Q = agent.dqn_network(next_controllable_latent, next_uncontrollable_features)
    target_actions = torch.argmax(next_Q, dim=1).unsqueeze(1)
    # We use the target Q-network and fill in the actions from the 'original' Q-network
    target_Q = agent.target_network(next_controllable_latent, next_uncontrollable_features)
    target_Q_value = target_Q.gather(1, target_actions)
    # Current timestep Q-values with the actions from the minibatch
    Q = Q.gather(1, actions)
    Q_target = rewards + (1 - dones.int()) * agent.gamma * target_Q_value
    loss = F.mse_loss(Q, Q_target.detach())

    return loss


def compute_entropy_loss_multiple_trajectories(agent):

    trajectory1 = agent.buffer.sample_trajectory()[0]
    encoded_trajectory1 = agent.encoder(trajectory1)[0]

    trajectory2 = agent.buffer.sample_trajectory()[0]
    encoded_trajectory2 = agent.encoder(trajectory2)[0]

    trajectory4 = agent.buffer.sample_trajectory()[0]
    encoded_trajectory4 = agent.encoder(trajectory4)[0]

    trajectory3 = agent.buffer.sample_trajectory()[0]
    encoded_trajectory3 = agent.encoder(trajectory3)[0]

    # Roll the states uniformly random so we eventually get all combinations
    encoded_trajectory_rolled1 = torch.roll(encoded_trajectory1, np.random.randint(low=1, high=len(encoded_trajectory1-1)), dims=0)

    loss1 = torch.exp(-agent.entropy_scaler * torch.norm(encoded_trajectory_rolled1 - encoded_trajectory1, dim=1, p=2)).mean()

    encoded_trajectory_rolled2 = torch.roll(encoded_trajectory2, np.random.randint(low=1, high=len(encoded_trajectory2-1)), dims=0)

    loss2 = torch.exp(-agent.entropy_scaler * torch.norm(encoded_trajectory_rolled2 - encoded_trajectory2, dim=1, p=2)).mean()

    encoded_trajectory_rolled3 = torch.roll(encoded_trajectory3, np.random.randint(low=1, high=len(encoded_trajectory3-1)), dims=0)

    loss3 = torch.exp(-agent.entropy_scaler * torch.norm(encoded_trajectory_rolled3 - encoded_trajectory3, dim=1, p=2)).mean()

    encoded_trajectory_rolled4 = torch.roll(encoded_trajectory4, np.random.randint(low=1, high=len(encoded_trajectory4-1)), dims=0)

    loss4 = torch.exp(-agent.entropy_scaler * torch.norm(encoded_trajectory_rolled4 - encoded_trajectory4, dim=1, p=2)).mean()

    return 0.25*(loss1 + loss2 + loss3 + loss4)

#         loss_entropy = 0.5*compute_entropy_loss_featuremaps_randompixels(self)
#         loss_entropy += 0.5*compute_entropy_loss_multiple_trajectories(self)