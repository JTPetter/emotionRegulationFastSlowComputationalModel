import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import path
import os
import sys
import SALib.sample.sobol
import SALib.analyze.sobol

from environment import Stimulus, AgentStatus, EmotionEnv
from agent import QTableAgent


# Function to calculate the score
def calculate_score(range_data, col_for_1):
    total_sum = np.sum(range_data)
    col_sum = np.sum(range_data[:, col_for_1])
    return col_sum / total_sum if total_sum != 0 else 0


# #Parameters for grid search
# grid_parameters = {
#     'N_STIMULI': [800],
#     'STIMULUS_MAX_OCCURRENCE': [5],
#     'alpha': [.1],
#     'gamma': [.99],
#     'epsilon': [1],
#     'disengage_benefit': [2],
#     'engage_benefit': [2],
#     'engage_adaptation': [2],
#     'SEED': [212334],
#     'PERCENTAGE_RESOLVABLE_STIMULI': [.5]    # 0 to 1
# }

# n_grid_parameters = len(grid_parameters)
# grid = np.array(np.meshgrid(grid_parameters['N_STIMULI'], grid_parameters['STIMULUS_MAX_OCCURRENCE'], grid_parameters['alpha'],
#                             grid_parameters['gamma'], grid_parameters['epsilon'], grid_parameters['disengage_benefit'],
#                             grid_parameters['engage_benefit'], grid_parameters['engage_adaptation'], grid_parameters['SEED'],
#                             grid_parameters['PERCENTAGE_RESOLVABLE_STIMULI']))
# grid = grid.reshape(n_grid_parameters, int(grid.size/n_grid_parameters)).T

#
N_SAMPLES = 2
problem = {
    "num_vars": 5,
    "names": ["N_STIMULI", "MAX_OCCURENCE", "alpha", "gamma", "PERCENTAGE_RESOLVABLE_STIMULI"],
    "bounds": [[100, 5000], [2, 10], [0.001, 0.3], [.9, .999], [0, 1]]
}
sample = SALib.sample.sobol.sample(problem, N_SAMPLES)
Y = np.empty([sample.shape[0]])

for run in range(len(Y)):

    current_parameters = sample[run]

    SEED = np.random.randint(1, 2349082)
    N_RUNS = 400  # runs for training
    N_STIMULI = int(current_parameters[0])
    N_ACTIONS = 2
    N_STATES = 11
    STIMULUS_MAX_OCCURRENCE = int(current_parameters[1])
    STIMULUS_INT_MIN = 1
    STIMULUS_INT_MAX = 10
    DECAY_TIME = N_RUNS * 1    # How much of the total run is used for exploring
    PERCENTAGE_RESOLVABLE_STIMULI = current_parameters[4]
    TIME_EQUATION_EXPONENT = 2  # higher exponents lead to the curves separating later but more strongly

    alpha = current_parameters[2]
    gamma = current_parameters[3]
    epsilon = 1
    DECAY_FACTOR = epsilon/DECAY_TIME  # how much epsilon is lowered each step

    disengage_benefit = 2
    engage_adaptation = 2
    engage_benefit = 2

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
        stimulus.p_occurrence /= p_sum

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

    agent = QTableAgent(11, n_actions=N_ACTIONS, alpha=alpha, gamma=gamma, epsilon=epsilon)

    action = 1 # the first action
    state = env.agent_status.current_emo_intensity

    # Run Training
    for i in range(N_RUNS):
        next_state, reward, done, info = env.step(action)
        agent.update(state, next_state, action, reward)
        if i % 100 == 0:
            print(run, '/', len(Y), '_____', round(i / (N_RUNS) * 100, 2) , '%', sep='')
        state = int(env.agent_status.current_emo_intensity)
        action = agent.choose_action(state, 'epsilon_greedy')
        if agent.epsilon > 0.1:   #cap epsilon at .1
            agent.epsilon -= DECAY_FACTOR



    # Record actions and rewards
    action_counts = np.zeros((N_STATES, agent.n_actions))
    reward_counts = np.zeros((N_RUNS, agent.n_actions))

    # Run Simulation
    agent.alpha = 0
    agent.epsilon = 0
    env.reset()
    for i in range(50):
        next_state, reward, done, info = env.step(action)
        state = int(env.agent_status.current_emo_intensity)
        state_intensity = int(env.agent_status.current_emo_intensity)
        action = agent.choose_action(state, 'epsilon_greedy')
        action_counts[state_intensity, action] += 1

    # # Plot choices
    # states = np.arange(0, 11)
    # plt.plot(states, action_counts[:, 0], marker='', color='blue', linewidth=2, label='Distraction')
    # plt.plot(states, action_counts[:, 1], marker='', color='red', linewidth=2, label='Reappraisal')
    # plt.ylim([0, np.max(action_counts)])
    # plt.legend()
    # plt.show()

    # Get single number that quantifies output
    print(action_counts)

    # Exclude the first row (intensity 0)
    action_counts = action_counts[1:]

    # Intensity 1 to 5
    range1 = action_counts[0:5]
    score1 = calculate_score(range1, 1)

    # Intensity 6 to 10
    range2 = action_counts[5:]
    score2 = calculate_score(range2, 0)

    # sum of scores
    score_sum = np.sum([score1, score2])

    print("Score for intensity 1 to 5:", score1)
    print("Score for intensity 6 to 10:", score2)
    print("Total score:", score_sum)

    Y[run] = score_sum

sensitivity = SALib.analyze.sobol.analyze(problem, Y)
print(sensitivity["S1"])
print(sensitivity["ST"])

#set options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


folder_path = "../datasets/Manuscript/Sensitivity"   # where to save the data
os.makedirs(folder_path)     # create a folder

#to write the sensitivity to csv
df1 = pd.DataFrame({'Names': ["N_STIMULI", "MAX_OCCURENCE", "alpha", "gamma", "PERCENTAGE_RESOLVABLE_STIMULI"],
                       'First order': sensitivity["S1"], 'Total order': sensitivity["ST"]})
file_name1 = folder_path + '/' + 'Sensitivity.csv'
df1.to_csv(file_name1)
