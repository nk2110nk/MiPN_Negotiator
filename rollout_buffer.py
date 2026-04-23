"""Rollout buffer for PPO training."""

import numpy as np
import torch
from typing import NamedTuple
from gymnasium import spaces


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def get_action_dim(action_space) -> int:
    """Get the dimension of the action space."""
    if isinstance(action_space, tuple):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return sum(list(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(observation_space):
    """Get the shape of the observation space."""
    if isinstance(observation_space, tuple):
        return observation_space
    elif isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


class RolloutBuffer:
    """Rollout buffer for collecting experience data during PPO training."""
    
    def __init__(
        self, 
        buffer_size, 
        n_envs, 
        obs_space, 
        action_space, 
        device,
        gamma=0.99,
        gae_lambda=1
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.observation_space = obs_space
        self.obs_dim = get_obs_shape(obs_space)
        if isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = len(action_space.nvec)
        else:
            self.action_dim = get_action_dim(action_space)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def reset(self):
        """Reset the buffer."""
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.generator_ready = False
    
    def empty_cache(self):
        """Clear buffers from memory."""
        del self.observations
        del self.actions
        del self.rewards
        del self.returns
        del self.episode_starts
        del self.values
        del self.log_probs
        del self.advantages
        torch.cuda.empty_cache()

    def add(self, obs, action, reward, episode_start, value, log_prob):
        """Add experience to buffer."""
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_dim))

        # Reshape to handle multi-dim and discrete action spaces
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.cpu().flatten()
        self.log_probs[self.pos] = log_prob.cpu()
        self.pos += 1

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray):
        """Compute returns and advantages using GAE."""
        last_values = last_values.cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            '''delta = r_t + gamma * V(s_{t+1}) - V(s_t)'''
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            '''A = delta(t) + gamma * lamda * delta(t+1)'''
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    @staticmethod
    def swap_and_flatten(arr):
        """Swap and flatten buffer."""
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size):
        """Get batches from the buffer."""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            self.observations = self.swap_and_flatten(self.observations)
            self.actions = self.swap_and_flatten(self.actions)
            self.values = self.swap_and_flatten(self.values)
            self.log_probs = self.swap_and_flatten(self.log_probs)
            self.advantages = self.swap_and_flatten(self.advantages)
            self.returns = self.swap_and_flatten(self.returns)
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def to_torch(self, array):
        """Convert array to torch tensor."""
        return torch.as_tensor(array, device=self.device)

    def _get_samples(self, batch_inds):
        """Get samples for a batch."""
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
