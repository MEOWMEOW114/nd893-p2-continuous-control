#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the Q-networks used by the agents
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


'''
Begin help functions
'''


def hidden_init(layer):
    # source: The other layers were initialized from uniform distributions
    # [− 1/sqrt(f) , 1/sqrt(f) ] where f is the fan-in of the layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

'''
End help functions
'''


class Policy(nn.Module):
    '''
    '''

    def __init__(self, action_size, ActorBody, CriticBody):
        '''
        '''
        super(Policy, self).__init__()
        self.actor_body = ActorBody
        self.critic_body = CriticBody
        self.std = torch.Tensor(nn.Parameter(torch.ones(1, action_size)))

    def forward(self, states, actions=None):
        '''
        '''
        estimated_actions = self.actor_body(states)
        estimated_values = self.critic_body(states)
        # pdb.set_trace()
        dist = torch.distributions.Normal(estimated_actions, self.std)
        i_dim = 2
        if isinstance(actions, type(None)):
            i_dim = 1
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=i_dim, keepdim=True)
        # entropy_loss = torch.Tensor(np.zeros((log_prob.size(0), 1)))
        entropy_loss = dist.entropy()
        entropy_loss = torch.sum(entropy_loss, dim=i_dim)/4
        # print(actions)
        return actions, log_prob, entropy_loss, estimated_values


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400,
                 fc2_units=300):
        """Initialize parameters and build model.
        :param state_size: int. Dimension of each state
        :param action_size: int. Dimension of each action
        :param seed: int. Random seed
        :param fc1_units: int. Number of nodes in first hidden layer
        :param fc2_units: int. Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # source: The low-dimensional networks had 2 hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc1))
        # source: The final layer weights and biases of the actor and were
        # initialized from a uniform distribution [−3 × 10−3, 3 × 10−3]
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        """
        # source: used the rectified non-linearity for all hidden layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # source The final output layer of the actor was a tanh layer,
        # to bound the actions
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400,
                 fc2_units=300):
        """Initialize parameters and build model.
        :param state_size: int. Dimension of each state
        :param action_size: int. Dimension of each action
        :param seed: int. Random seed
        :param fcs1_units: int. Nb of nodes in the first hiddenlayer
        :param fc2_units: int. Nb of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values
        :param state: tuple.
        :param action: tuple.
        """
        xs = F.relu(self.fcs1(state))
        x = F.relu(self.fc2(xs))
        return self.fc3(x)
