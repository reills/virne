# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch.nn as nn
import torch.nn.functional as F
from ..neural_network import ResNetBlock
from virne.solver.learning.rl_base.policy_base import ActorCriticBase


class ActorCritic(ActorCriticBase):
    
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(feature_dim, action_dim, embedding_dim)
        self.critic = Critic(feature_dim, action_dim, embedding_dim)


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[1, feature_dim], stride=[1, 1]),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(action_dim, action_dim),
            # nn.ReLU(inplace=False),
        )
        # n_mid_channcels = action_dim * 2
        # self.net = nn.Sequential(
        #     # + channcel
        #     nn.Conv2d(action_dim, n_mid_channcels, kernel_size=[1, 1], stride=[1, 1]),
        #     ResNetBlock(n_mid_channcels),
        #     # - feature
        #     nn.Conv2d(n_mid_channcels, n_mid_channcels, kernel_size=[feature_dim, 1], stride=[1, 1]),
        #     ResNetBlock(n_mid_channcels),
        #     # - channcel
        #     nn.Conv2d(n_mid_channcels, action_dim, kernel_size=[1, 1], stride=[1, 1]),
        #     ResNetBlock(action_dim),
        #     # last
        #     nn.Conv2d(action_dim, action_dim, kernel_size=[1, 1], stride=[1, 1]),
        #     nn.ReLU(inplace=False),
        #     nn.Flatten(),
        # )
    def forward(self, obs):
        """Return logits of actions"""
        x = obs['p_net_x']
        action_logits = self.net(x)
        # action_logits = self.relu(out)
        return action_logits

class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim, embedding_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[1, feature_dim], stride=[1, 1]),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(action_dim, 1)
        )
        
    def forward(self, obs):
        """Return logits of actions"""
        x = obs['p_net_x']
        values = self.net(x)
        return values