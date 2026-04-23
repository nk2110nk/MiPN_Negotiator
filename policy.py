import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional, NamedTuple
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from gymnasium import spaces
import torch.nn.functional as F


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


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class MiPNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for MiPN (Multi-issue Partial negotiatioN).
    Extracts features from vectorized bids and time.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input = np.prod(observation_space.shape)
        
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class MiPNPolicy(ActorCriticPolicy):
    """
    Multi-Issue Partial Negotiation Policy Network.
    
    This policy network outputs multiple discrete actions:
    - Issue actions: Value selection for each issue
    - Accept action: Accept (0) or Reject (1)
    
    The architecture follows the MiPN framework with:
    - Shared Feature Extractor
    - Issue-level policy heads (one per issue)
    - Accept/Reject policy head
    - Value network
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule=None,
        net_arch: Optional[Dict[str, Any]] = None,
        activation_fn = nn.ReLU,
        features_dim: int = 64,
        device: torch.device = None,
        **kwargs
    ):
        if net_arch is None:
            net_arch = {"pi": [64, 64], "vf": [64, 64]}
        
        self.features_dim = features_dim
        self.action_space_type = type(action_space)
        self.device = device
        
        # lr_schedule が None の場合の処理
        if lr_schedule is None:
            lr_schedule = lambda progress: 3e-4
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """Build the shared feature extractor and policy/value networks."""
        # Shared feature extractor
        self.features_extractor = MiPNFeaturesExtractor(
            self.observation_space,
            features_dim=self.features_dim
        )
        
        # Get dimensions from net_arch
        pi_net_arch = self.net_arch["pi"]
        vf_net_arch = self.net_arch["vf"]
        
        # Build shared layers
        pi_layers = []
        vf_layers = []
        
        input_dim = self.features_dim
        
        # Policy network shared layers
        for hidden_dim in pi_net_arch[:-1]:
            pi_layers.append(nn.Linear(input_dim, hidden_dim))
            pi_layers.append(self.activation_fn())
            input_dim = hidden_dim
        
        # Value network shared layers
        input_dim = self.features_dim
        for hidden_dim in vf_net_arch[:-1]:
            vf_layers.append(nn.Linear(input_dim, hidden_dim))
            vf_layers.append(self.activation_fn())
            input_dim = hidden_dim
        
        self.policy_net = nn.Sequential(*pi_layers)
        self.value_net = nn.Sequential(*vf_layers)
        
        # Issue-specific policy heads
        pi_last_dim = pi_net_arch[-1] if pi_net_arch else self.features_dim
        vf_last_dim = vf_net_arch[-1] if vf_net_arch else self.features_dim
        
        # For MultiDiscrete action space
        if isinstance(self.action_space, spaces.MultiDiscrete):
            self.issue_heads = nn.ModuleList([
                nn.Linear(pi_last_dim, n_actions)
                for n_actions in self.action_space.nvec[:-1]  # All except the last (accept) action
            ])
            self.accept_head = nn.Linear(pi_last_dim, self.action_space.nvec[-1])
        else:
            raise ValueError(f"MiPNPolicy only supports MultiDiscrete action space, got {type(self.action_space)}")
        
        # Value head
        self.value_head = nn.Linear(vf_last_dim, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Returns:
            actions: Action sampled from the policy
            values: Value estimation
            log_probs: Log probability of the action
        """
        # Extract features
        features = self.features_extractor(obs)
        
        # Policy network
        pi_features = self.policy_net(features)
        vf_features = self.value_net(features)
        
        # Get action logits from issue heads
        issue_logits = [head(pi_features) for head in self.issue_heads]
        accept_logits = self.accept_head(pi_features)
        
        # Create distributions and sample actions
        distributions = []
        actions = []
        log_probs_list = []
        
        for logits in issue_logits:
            dist = torch.distributions.Categorical(logits=logits)
            distributions.append(dist)
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            actions.append(action)
            log_probs_list.append(dist.log_prob(action))
        
        # Accept action
        accept_dist = torch.distributions.Categorical(logits=accept_logits)
        distributions.append(accept_dist)
        if deterministic:
            accept_action = accept_dist.probs.argmax(dim=-1)
        else:
            accept_action = accept_dist.sample()
        actions.append(accept_action)
        log_probs_list.append(accept_dist.log_prob(accept_action))
        
        # Stack actions
        actions = torch.stack(actions, dim=-1)
        log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
        
        # Value estimation
        values = self.value_head(vf_features)
        
        return actions, values, log_probs
    
    def predict_values(self, x):
        """Predict values for given observations."""
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        
        features = self.features_extractor(x)
        vf_features = self.value_net(features)
        values = self.value_head(vf_features)
        return values
    
    def sample(self, obs, is_first_offer=None):
        """
        Sample actions from the policy.
        
        Args:
            obs: Observation tensor
            is_first_offer: List of booleans indicating if first offer
            
        Returns:
            actions: Sampled actions
            values: Value estimations
            log_probs: Log probabilities of the actions
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        actions, values, log_probs = self.forward(obs, deterministic=False)
        
        # Process first offer constraint if needed
        if is_first_offer is not None:
            # Additional logic for first offer handling if needed
            pass
        
        return actions, values, log_probs
    
    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Predict action without value and log_prob."""
        features = self.features_extractor(observation)
        pi_features = self.policy_net(features)
        
        # Get actions from issue heads
        actions = []
        for head in self.issue_heads:
            logits = head(pi_features)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            actions.append(action)
        
        # Accept action
        accept_logits = self.accept_head(pi_features)
        if deterministic:
            accept_action = accept_logits.argmax(dim=-1)
        else:
            accept_dist = torch.distributions.Categorical(logits=accept_logits)
            accept_action = accept_dist.sample()
        actions.append(accept_action)
        
        # Stack actions
        actions = torch.stack(actions, dim=-1)
        return actions
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the action taken by the agent.
        
        Returns:
            values: Value estimation
            log_probs: Log probability of the action
            entropy: Entropy of the policy
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        features = self.features_extractor(obs)
        pi_features = self.policy_net(features)
        vf_features = self.value_net(features)
        
        # Get action logits
        issue_logits = [head(pi_features) for head in self.issue_heads]
        accept_logits = self.accept_head(pi_features)
        
        # Compute log probabilities
        log_probs_list = []
        entropies_list = []
        
        # Issue actions
        for i, logits in enumerate(issue_logits):
            dist = torch.distributions.Categorical(logits=logits)
            log_probs_list.append(dist.log_prob(actions[:, i]))
            entropies_list.append(dist.entropy())
        
        # Accept action
        accept_dist = torch.distributions.Categorical(logits=accept_logits)
        log_probs_list.append(accept_dist.log_prob(actions[:, -1]))
        entropies_list.append(accept_dist.entropy())
        
        log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies_list, dim=-1).sum(dim=-1)
        
        # Value estimation
        values = self.value_head(vf_features)
        
        return values, log_probs, entropy
    
    def get_distribution(
        self,
        obs: torch.Tensor,
    ):
        """Get the distribution of actions."""
        features = self.features_extractor(obs)
        pi_features = self.policy_net(features)
        
        issue_logits = [head(pi_features) for head in self.issue_heads]
        accept_logits = self.accept_head(pi_features)
        
        distributions = []
        for logits in issue_logits:
            distributions.append(torch.distributions.Categorical(logits=logits))
        
        accept_dist = torch.distributions.Categorical(logits=accept_logits)
        distributions.append(accept_dist)
        
        return distributions
