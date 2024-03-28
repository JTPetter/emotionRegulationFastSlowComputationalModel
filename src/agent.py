import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
from scipy.special import softmax

# Set up logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class QTableAgent:
    '''
    Tabular Q-learning agent
    p 131 of Sutton and Barto
    '''

    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):

        self.n_states = n_states
        self.n_actions = n_actions

        # Initialize q-table randomly
        # self.qtable = np.random.rand(n_states, n_actions)

        # Initialize q-table with zeros
        self.qtable = np.zeros((n_states, n_actions))

        # Initialize q-table with ones
        # self.qtable = np.ones((n_states, n_actions))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Used for Upper confidence bound policy (UCB)
        self.t = 0
        self.action_history = np.zeros(self.n_actions)

    def choose_action(self, state_id, policy, **kwargs):

        if policy == 'softmax_q':  # Take actions according to probabilities assigned by softmax transformed q-values
            action = random.choices(np.arange(self.n_actions), weights=softmax(self.qtable[state_id, :]))[0]
        elif policy == 'epsilon_greedy':  # Epsilon-greedy policy
            if np.random.rand(1)[0] > self.epsilon:
                action = np.random.choice(np.flatnonzero(self.qtable[state_id, :] == self.qtable[state_id, :].max()))
            else:
                action = np.random.randint(0, self.n_actions)
        elif policy == 'ucb':  # Upper-confidence bound action selection, p35 sutton & barto
            assert 'c' in kwargs.keys()
            assert kwargs['c'] > 0
            uncertainty = np.sqrt((np.log(self.t / self.action_history)))
            ucb_actions = self.qtable[state_id, :] + kwargs['c'] * uncertainty
            logger.debug(ucb_actions)
            action = np.argmax(ucb_actions)
        else:
            raise NotImplementedError()

        self.action_history[action] += 1
        self.t += 1
        return action

    def update(self, state_id, next_state_id, action_id, reward):
        q = self.qtable[state_id, action_id] + self.alpha * (reward + self.gamma * np.max(self.qtable[next_state_id, :]) - self.qtable[state_id, action_id])
        self.qtable[state_id, action_id] = q



