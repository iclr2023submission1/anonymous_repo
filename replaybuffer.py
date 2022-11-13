import numpy as np
import torch

# Buffer structure adopted from Denis Yarats, SAC-AE (https://github.com/denisyarats/pytorch_sac_ae)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, obs_shape[0], obs_shape[1]), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_shape[0], obs_shape[1]), dtype=np.float32)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=bool)

        self.idx = 0
        self.full = False

        self.id = None

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, dones

    def sample_trajectory(self):

        if self.id is None:
            self.id = np.where(self.dones[0:self.idx])[0]  # Find the end of all the trajectories (dones=True)

        random_start = np.random.randint(low=0, high=len(self.id)-2, size=1)              # Find the start of a trajectory in the buffer
        random_start_id = self.id[random_start]+1
        random_end_id = self.id[random_start + 1] # new trajectory starts 1 past the done index
        trajectory_ids = np.arange(random_start_id, random_end_id)  # Make all the ids

        obses = self.obses[trajectory_ids]
        next_obses = self.next_obses[trajectory_ids]
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[trajectory_ids], device=self.device)
        rewards = torch.as_tensor(self.rewards[trajectory_ids], device=self.device)
        dones = torch.as_tensor(self.dones[trajectory_ids], device=self.device)

        return obses, actions, rewards, next_obses, dones
