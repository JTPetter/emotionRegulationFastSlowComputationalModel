import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import path
import os
import logging
import sys

from environment_no_LT import Stimulus, AgentStatus, EmotionEnv
from agent import QTableAgent


def bin_low_high(value):
    if value > 5:
        return 2
    elif value > 0:
        return 1
    else:
        return 0

# Set up logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

#Parameters for grid search
grid_parameters = {
    'N_STIMULI': [800],
    'STIMULUS_MAX_OCCURRENCE': [5],
    'alpha': [.1],
    'gamma': [.99],
    'epsilon': [1],
    'disengage_benefit': [2],
    'engage_benefit': [2],
    'engage_adaptation': [2],
    'SEED': [np.arange(21, 51)],
    'PERCENTAGE_RESOLVABLE_STIMULI': [.5]    # 0 to 1
}

n_grid_parameters = len(grid_parameters)
grid = np.array(np.meshgrid(grid_parameters['N_STIMULI'], grid_parameters['STIMULUS_MAX_OCCURRENCE'], grid_parameters['alpha'],
                            grid_parameters['gamma'], grid_parameters['epsilon'], grid_parameters['disengage_benefit'],
                            grid_parameters['engage_benefit'], grid_parameters['engage_adaptation'], grid_parameters['SEED'],
                            grid_parameters['PERCENTAGE_RESOLVABLE_STIMULI']))
grid = grid.reshape(n_grid_parameters, int(grid.size/n_grid_parameters)).T

file_name = "Theory_no_LT"      # the first part of the file name, automatically appended with the respective simulation value and data description
                                    #DONT USE NUMBERS IN FILE NAME
folder_path = "../datasets/Theory_no_LT_partTwo" + file_name   # where to save the data
os.makedirs(folder_path)     # create a folder

for row in np.arange(0, len(grid)):

    SEED = int(grid[row, 8])
    N_RUNS = 50000
    N_STIMULI = int(grid[row, 0])
    N_ACTIONS = 2
    N_STATES = 3
    STIMULUS_MAX_OCCURRENCE = int(grid[row, 1])
    STIMULUS_INT_MIN = 1
    STIMULUS_INT_MAX = 10
    DECAY_TIME = N_RUNS * 1    # How much of the total run is used for exploring
    PERCENTAGE_RESOLVABLE_STIMULI = grid[row, 9]
    TIME_EQUATION_EXPONENT = 2

    alpha = grid[row, 2]
    gamma = grid[row, 3]
    epsilon = grid[row, 4]
    DECAY_FACTOR = epsilon/DECAY_TIME  # how much epsilon is lowered each step

    disengage_benefit = grid[row, 5]
    engage_adaptation = grid[row, 7]
    engage_benefit = grid[row, 6]

    random.seed(SEED)
    np.random.seed(SEED)

    stimuli_list = []
    resolvable_ids = np.random.choice(range(N_STIMULI), size=int(N_STIMULI * PERCENTAGE_RESOLVABLE_STIMULI),
                                      replace=False)
    for i in range(N_STIMULI):
        id = i
        emo_intensity = np.random.randint(STIMULUS_INT_MIN, STIMULUS_INT_MAX + 1)
        p_occurrence = np.random.uniform(0, 1, 1)
        stimuli_list.append(Stimulus(id=id, emo_intensity=emo_intensity, p_occurrence=p_occurrence, resolvable=(i in resolvable_ids)))

    p_sum = sum(stimulus.p_occurrence for stimulus in stimuli_list)
    for stimulus in stimuli_list:
        stimulus.p_occurrence = stimulus.p_occurrence / p_sum


    agent_status = AgentStatus()

    env = EmotionEnv(engage_benefit=engage_benefit,
                     disengage_benefit=disengage_benefit,
                     engage_adaptation=engage_adaptation,
                     stimulus_max_occurrence=STIMULUS_MAX_OCCURRENCE,
                     stimuli=stimuli_list,
                     agent_status=agent_status,
                     time_equation_exponent=TIME_EQUATION_EXPONENT
                     )
    env.reset()

    agent = QTableAgent(N_STATES, n_actions=N_ACTIONS, alpha=alpha, gamma=gamma, epsilon=epsilon)

    action = 1 # the first action
    state = bin_low_high(env.agent_status.current_emo_intensity)    #the first state

    # Record actions and rewards
    action_counts = np.zeros((N_STATES, agent.n_actions))
    reward_counts = np.zeros((N_RUNS, agent.n_actions))
    qTable_update_amount = []

    # Run Training
    for i in range(N_RUNS):
        next_state, reward, done, info = env.step(action)
        next_state = bin_low_high(next_state)
        agent.update(state, next_state, action, reward)
        logger.debug(f'action: {action}, reward: {reward}, step: {i}')
        if i % 100 == 0:
            print(row, '/', len(grid), '_____', round(i / (N_RUNS) * 100, 2) , '%', sep='')
        state = bin_low_high(env.agent_status.current_emo_intensity)
        action = agent.choose_action(state, policy="epsilon_greedy")
        if agent.epsilon > 0.1:   #cap epsilon at .1
            agent.epsilon -= DECAY_FACTOR
        #print(agent.qtable)


    # Run Simulation
    agent.alpha = 0
    for i in range(10000):
        next_state, reward, done, info = env.step(action)
        next_state = bin_low_high(next_state)
        previous_qTable_sum = np.sum(agent.qtable)  # qTable values sum before updating
        agent.update(state, next_state, action, reward)
        qTable_update_amount.append(np.sum(agent.qtable) - previous_qTable_sum)  #how much the qTable changed from the update
        logger.debug(f'action: {action}, reward: {reward}, step: {i}')
        if i % 100 == 0:
            print(row, '/', len(grid), '_____', round(i / (N_RUNS) * 100, 2) , '%', sep='')
        action_counts[state, action] += 1
        reward_counts[i, action] += reward
        state = bin_low_high(env.agent_status.current_emo_intensity)
        action = agent.choose_action(state, policy="epsilon_greedy")



    # Plot choices
    states = np.arange(0, N_STATES)
    plt.plot(states, action_counts[:, 0], marker='', color='blue', linewidth=2, label='disengage')
    plt.plot(states, action_counts[:, 1], marker='', color='red', linewidth=2, label='engage')
    plt.ylim([0, np.max(action_counts)])
    plt.legend()
    #plt.show()


    # plot qTable update amount
    time = np.arange(0, 10000)
    plt.plot(time, qTable_update_amount, marker='', color='olive', linewidth=2)
    #plt.show()



    #set options for pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


    #to write the actions to csv
    df1 = pd.DataFrame({'disengage': action_counts[:, 0], 'engage': action_counts[:, 1]})
    file_name1 = folder_path + '/' + file_name + '_' + str(row) + '_actionPerIntensity' '.csv'
    df1.to_csv(file_name1)


