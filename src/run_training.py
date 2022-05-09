import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import path
import os
import logging
import sys
import sklearn

from environment import Stimulus, AgentStatus, EmotionEnv
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
    'N_STIMULI': [300],
    'STIMULUS_MAX_OCCURRENCE': [7],
    'alpha': [.1],
    'gamma': [.99],
    'epsilon': [1],
    'disengage_benefit': [.5],
    'engage_benefit': [.3],
    'engage_adaptation': [1.5],
    'SEED': [123],
    'PERCENTAGE_RESOLVABLE_STIMULI': [1]    # 0 to 1
}

n_grid_parameters = len(grid_parameters)
grid = np.array(np.meshgrid(grid_parameters['N_STIMULI'], grid_parameters['STIMULUS_MAX_OCCURRENCE'], grid_parameters['alpha'],
                            grid_parameters['gamma'], grid_parameters['epsilon'], grid_parameters['disengage_benefit'],
                            grid_parameters['engage_benefit'], grid_parameters['engage_adaptation'], grid_parameters['SEED'],
                            grid_parameters['PERCENTAGE_RESOLVABLE_STIMULI']))
grid = grid.reshape(n_grid_parameters, int(grid.size/n_grid_parameters)).T

# file_name = "ConfirmMaxOccurenceOne"      # the first part of the file name, automatically appended with the respective simulation value and data description
#                                     #DONT USE NUMBERS IN FILE NAME
# folder_path = "../datasets/" + file_name   # where to save the data
# os.makedirs(folder_path)     # create a folder

for row in np.arange(0, len(grid)):

    SEED = int(grid[row, 8])
    N_RUNS = 60000
    N_STIMULI = int(grid[row, 0])
    N_ACTIONS = 3
    N_STATES = 3
    STIMULUS_MAX_OCCURRENCE = int(grid[row, 1])
    STIMULUS_INT_MIN = 1
    STIMULUS_INT_MAX = 10
    DECAY_TIME = N_RUNS * .7    # How much of the total run is used for exploring
    PERCENTAGE_RESOLVABLE_STIMULI = grid[row, 9]

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
                     agent_status=agent_status
                     )
    env.reset()

    agent = QTableAgent(N_STATES, n_actions=N_ACTIONS, alpha=alpha, gamma=gamma, epsilon=epsilon)

    action = 0 # the first action
    state = bin_low_high(env.agent_status.current_emo_intensity)    #the first state

    # Record actions and rewards
    action_counts = np.zeros((N_STATES, agent.n_actions))
    reward_counts = np.zeros((N_RUNS, agent.n_actions))
    qTable_update_amount = []

    # Run simulation
    for i in range(N_RUNS):
        #env.render()
        next_state, reward, done, info = env.step(action)
        next_state = bin_low_high(next_state)
        #print(state, next_state)
        previous_qTable_sum = np.sum(agent.qtable)  # qTable values sum before updating
        agent.update(state, next_state, action, reward)
        qTable_update_amount.append(np.sum(agent.qtable) - previous_qTable_sum)  #how much the qTable changed from the update
        logger.debug(f'action: {action}, reward: {reward}, step: {i}')
        if i % 100 == 0:
            print(row, '/', len(grid), '_____', round(i / (N_RUNS) * 100, 2) , '%', sep='')
        action_counts[state, action] += 1
        reward_counts[i, action] += reward
        state = bin_low_high(env.agent_status.current_emo_intensity)  # env.get_original_intensity(agent_status.current_id)
        action = agent.choose_action(state, policy="epsilon_greedy")
        if agent.epsilon > 0.1:   #cap epsilon at .1
            agent.epsilon -= DECAY_FACTOR
        #print(agent.qtable)


    # #create data for running the same stimulus 3 times with every action
    # intensity_values = np.zeros((30, N_ACTIONS))
    # stimuli_list2 = [Stimulus(id=21894721947, emo_intensity=9, p_occurrence=1)]
    # agent_status2 = AgentStatus()
    # env2 = EmotionEnv(engage_benefit=engage_benefit,
    #                   disengage_benefit=disengage_benefit,
    #                   engage_adaptation=engage_adaptation,
    #                   stimuli=stimuli_list2,
    #                   agent_status=agent_status2
    #                   )
    # env2.reset()

    #
    # for ac in [0, 1, 2]:
    #     action = ac
    #     for i in np.arange(0, 30, 1):
    #         next_state, reward, done, info = env2.step(action)
    #         logger.debug(f'action: {action}, reward: {reward}, step: {i}')
    #         intensity_values[i, ac] = agent_status2.current_emo_intensity
    #         env2.render()

    # #Create balanced qTable with current settings
    # agent_status3 = AgentStatus()
    #
    # env3 = EmotionEnv(engage_benefit=engage_benefit,
    #                          disengage_benefit=disengage_benefit,
    #                          engage_adaptation=engage_adaptation,
    #                          stimuli=stimuli_list,
    #                          agent_status=agent_status3
    #                          )
    # env3.reset()
    #
    # agent2 = QTableAgent(11, n_actions=N_ACTIONS, alpha=alpha, gamma=gamma, epsilon=1)  # for balanced qTable
    #
    # action = 0  # the first action
    #
    # # Run simulation
    # for i in range(30000):
    #     next_state, reward, done, info = env3.step(action)
    #     agent2.update(state, next_state, action, reward)
    #     if i % 100 == 0:
    #         print(row, '/', len(grid), '_____', round((i + N_RUNS) /(N_RUNS + 30000) * 100, 2), '%', sep='')
    #     if done:
    #         env3.refresh_stimuli_list()
    #         env3.reset()
    #         state = env3.agent_status.current_emo_intensity #env3.get_original_intensity(agent_status.current_id)
    #         action = agent2.choose_action(state, policy="epsilon_greedy")

    #plot the emotional intensity curve per action
    # time = np.arange(0, 30)
    # plt.plot(time, intensity_values[:, 0], marker='', color='olive', linewidth=2, label='inaction')
    # plt.plot(time, intensity_values[:, 1], marker='', color='blue', linewidth=2, label='disengage')
    # plt.plot(time, intensity_values[:, 2], marker='', color='red', linewidth=2, label='engage')
    # plt.legend()
    # ax = plt.gca()
    # ax.set_ylim([0, 10])
    # plt.show()

    # Plot choices
    states = np.arange(0, N_STATES)
    #action_cumsum = np.cumsum(action_counts, axis=0)
    plt.plot(states, action_counts[:, 0], marker='', color='olive', linewidth=2, label='inaction')
    plt.plot(states, action_counts[:, 1], marker='', color='blue', linewidth=2, label='disengage')
    plt.plot(states, action_counts[:, 2], marker='', color='red', linewidth=2, label='engage')
    plt.legend()
    plt.show()


    # plot qTable update amount
    time = np.arange(0, N_RUNS)
    plt.plot(time, qTable_update_amount, marker='', color='olive', linewidth=2)
    plt.show()


    # plot rewards
    time = np.arange(0, N_RUNS)
    inaction_timeline = np.cumsum(reward_counts[:, 0] != 0) + 1
    disengage_timeline = np.cumsum(reward_counts[:, 1] != 0) + 1
    engage_timeline = np.cumsum(reward_counts[:, 2] != 0) + 1
    reward_cumsum = np.cumsum(reward_counts, axis=0)
    reward_cumsum[:, 0] / np.arange(1, N_RUNS + 1, 1)
    plt.plot(time, reward_cumsum[:, 0]/inaction_timeline, marker='', color='olive', linewidth=2, label='inaction')
    plt.plot(time, reward_cumsum[:, 1]/disengage_timeline, marker='', color='blue', linewidth=2, label='disengage')
    plt.plot(time, reward_cumsum[:, 2]/engage_timeline, marker='', color='red', linewidth=2, label='engage')
    plt.legend()
    #plt.show()





    # #set options for pandas
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    #
    # # to write parameters to csv
    # df_parameters = pd.DataFrame({'SEED': SEED, 'N_RUNS': N_RUNS, 'N_STIMULI': N_STIMULI, 'STIMULUS_INT_MIN': STIMULUS_INT_MIN,
    #                               'STIMULUS_INT_MAX': STIMULUS_INT_MAX, 'STIMULUS_MAX_OCCURRENCE': STIMULUS_MAX_OCCURRENCE,
    #                               'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon, 'disengage_benefit:': disengage_benefit,
    #                               'engage_benefit': engage_benefit, 'engage_adaptation': engage_adaptation}, index=[0])
    # file_name0 = folder_path + '/' + file_name + '_' + str(row) + '_parameters' '.csv'
    # df_parameters.to_csv(file_name0)
    # #
    # #
    # # #to write the actions to csv
    # df1 = pd.DataFrame({'inaction': action_counts[:, 0], 'disengage': action_counts[:, 1], 'engage': action_counts[:, 2]})
    # file_name1 = folder_path + '/' + file_name + '_' + str(row) + '_actionPerIntensity' '.csv'
    # df1.to_csv(file_name1)


    # #To write the rewards to csv
    # df2 = pd.DataFrame({'inaction': reward_cumsum[:, 0]/inaction_timeline, 'disengage': reward_cumsum[:, 1]/disengage_timeline,
    #                     'engage': reward_cumsum[:, 2]/engage_timeline})
    # file_name2 = folder_path + '/' + file_name + '_' + str(sv) + '_RewardsCumMean' '.csv'
    # df2.to_csv(file_name2)
    #
    #
    # #To write action trajectory to csv
    # df3 = pd.DataFrame({'inaction': intensity_values[:, 0], 'disengage': intensity_values[:, 1], 'engage': intensity_values[:, 2]})
    # file_name3 = folder_path + '/' + file_name + '_' + str(row) + '_actionTrajectory' '.csv'
    # df3.to_csv(file_name3)

    #learned value of action per intensity
    # df_learned_values = pd.DataFrame(
    #     {'inaction': agent.qtable[:, 0], 'disengage': agent.qtable[:, 1], 'engage': agent.qtable[:, 2]})
    # print(df_learned_values)
    # file_name4 = folder_path + '/' + file_name + '_' + str(row) + '_learnedValue' '.csv'
    # df_learned_values.round(decimals=2).to_csv(file_name4)
    #
    # # #expected value of action per intensity
    # df_expected_values = pd.DataFrame(
    #     {'inaction': agent2.qtable[:, 0], 'disengage': agent2.qtable[:, 1], 'engage': agent2.qtable[:, 2]})
    # file_name5 = folder_path + '/' + file_name + '_' + str(row) + '_expectedValue' '.csv'
    # df_expected_values.round(decimals=2).to_csv(file_name5)
    # #



