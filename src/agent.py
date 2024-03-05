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

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Modify the agent
class DQNAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = QNetwork(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            state = torch.tensor([state], dtype=torch.float32)
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def update(self, state, next_state, action, reward):
        state = torch.tensor([state], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)

        model_output = self.model(state)
        current_q = model_output[action]
        max_next_q = torch.max(self.model(next_state)).item()
        expected_q = reward + self.gamma * max_next_q
        loss = nn.MSELoss()(current_q.unsqueeze(0), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sum_of_q_values(self, all_states):
        total_q_value_sum = 0.0

        # Process each state through the neural network
        for state in all_states:
            # Convert state to tensor
            state_tensor = torch.tensor([state], dtype=torch.float32)

            # Get Q-values for this state
            q_values = self.model(state_tensor)

            # Sum Q-values and add to total sum
            total_q_value_sum += q_values.sum().item()

        return total_q_value_sum




