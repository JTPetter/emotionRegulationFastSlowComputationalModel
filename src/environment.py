import numpy as np
import random
import copy

import gym
from gym.spaces import Discrete, Tuple, Box, Dict


class Stimulus:

    def __init__(self, id: int, emo_intensity: int, p_occurrence: float, resolvable: bool):
        self.id = id
        self.emo_intensity = emo_intensity
        self.p_occurrence = p_occurrence
        self.encounter_counter = 0
        self.reappraisal_counter = 0
        self.resolvable = resolvable

    def get_intensity(self):
        return self.emo_intensity

    def get_p_occurrence(self):
        return self.p_occurrence

    def get_dict(self):
        return {'id': self.id, 'emo_intensity': self.emo_intensity, "p_occurrence": self.p_occurrence,
                'reappraisals': self.reappraisal_counter,
                'encounters': self.encounter_counter, 'resolvable': self.resolvable}


class AgentStatus:

    def __init__(self):
        self.stimuliAppraisals = list()
        self.current_id = None
        self.current_emo_intensity = None
        self.expected_p_occurrence = None
        self.current_encounter_counter = 0
        # self.previous_encounter = None

    def print_list(self):
        for i in range(len(self.stimuliAppraisals)):
            print(self.stimuliAppraisals[i].get_dict())

    def appraise_stimuli(self, stimulus: Stimulus):  # currently the appraisal function is 1:1, so just a copy
        if not self._check_for_previous_encounter(stimulus):
            self.stimuliAppraisals.append(copy.deepcopy(stimulus))
        self._update_emotional_state(stimulus)

    def _check_for_previous_encounter(self, stimulus):
        for i in range(0, len(self.stimuliAppraisals)):
            if self.stimuliAppraisals[i].id == stimulus.id:
                return True
        return False

    def _update_emotional_state(self, stimulus):
        for i in range(0, len(self.stimuliAppraisals)):
            if self.stimuliAppraisals[i].id == stimulus.id:
                self.current_emo_intensity = self.stimuliAppraisals[i].emo_intensity
                self.expected_occurrence = self.stimuliAppraisals[i].p_occurrence
                self.current_id = self.stimuliAppraisals[i].id
                self.stimuliAppraisals[i].encounter_counter += 1
                self.current_encounter_counter = self.stimuliAppraisals[i].encounter_counter


class EmotionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    N_ACTIONS = 3

    # # Actions
    # INACTION = 0
    # DISENGAGE = 1
    # ENGAGE = 2

    def __init__(self,
                 engage_benefit: float,
                 disengage_benefit: float,
                 engage_adaptation: float,
                 stimulus_max_occurrence: int,
                 stimuli: list,
                 agent_status: AgentStatus
                 ):

        super(EmotionEnv, self).__init__()

        self.action_space = Discrete(self.N_ACTIONS)
        self.observation_space = Dict({'stimulus_id': Discrete(len(stimuli)),
                                       'emo_intensity': Box(low=0, high=10, shape=(1,), dtype=np.float32),
                                       'emo_step:': Box(low=0, high=2, shape=(1,), dtype=np.int8)})

        self.stimuli = stimuli
        self.n_stimuli = len(stimuli)
        self.engage_benefit = engage_benefit
        self.engage_adaptation = engage_adaptation
        self.disengage_benefit = disengage_benefit
        self.agent_status = agent_status
        self.current_appraisal = None
        self.replacement_stimulus_counter = 0
        self.stimulus_max_occurrence = stimulus_max_occurrence

    def step(self, action: int) -> tuple:
        '''
        Execute one timestep in environment
        :param action: which action to take
        :return: state, reward, done, info
        '''

        for i in range(0, len(self.agent_status.stimuliAppraisals)):
            if self.agent_status.stimuliAppraisals[i].id == self.agent_status.current_id:
                self.current_appraisal = self.agent_status.stimuliAppraisals[i]

        # Take action
        if action == 1:
            self._disengage()
        elif action == 2:
            self._engage()
        elif action == 0:
            self._inaction()
        else:
            raise ValueError(f'Received invalid action {action} which is not part of the action space')

        info = None
        reward = self._get_reward()

        self.reset()
        self.refresh_stimuli_list()
        done = False

        return self.get_original_intensity(self.agent_status.current_id), reward, done, info  #

    def _inaction(self):
        return self.agent_status

    def _disengage(self):
        self.agent_status.current_emo_intensity -= self.disengage_benefit
        self.agent_status.current_emo_intensity = np.clip(self.agent_status.current_emo_intensity, 0, 10)
        return self.agent_status

    def _engage(self):
        if self.current_appraisal.resolvable:
            self.agent_status.current_emo_intensity = ((self.agent_status.current_emo_intensity ** 2) + (self.agent_status.current_emo_intensity - self.engage_benefit) * (10 - self.agent_status.current_emo_intensity)) / 10
            self.current_appraisal.emo_intensity -= self.engage_adaptation
            self.current_appraisal.reappraisal_counter += 1
        else:
            self.agent_status.current_emo_intensity -= self.engage_benefit + (self.current_appraisal.reappraisal_counter * self.engage_adaptation)
            self.current_appraisal.reappraisal_counter += 1
        self.agent_status.current_emo_intensity = np.clip(self.agent_status.current_emo_intensity, 0, 10)
        self.current_appraisal.emo_intensity = np.clip(self.current_appraisal.emo_intensity, 0, 10)
        return self.agent_status

    def _get_reward(self):
        reward = 10 - self.agent_status.current_emo_intensity
        return reward

    def reset(self):
        probs = np.array([stimulus.get_p_occurrence() for stimulus in self.stimuli]).flatten()
        new_stimulus = np.random.choice(self.stimuli, p=probs)
        self.agent_status.appraise_stimuli(new_stimulus)

    # stimulus gets replaced with a new stimulus with the same probability of occurrence and intensity, but new id
    def refresh_stimuli_list(self):
        if self.agent_status.current_encounter_counter == self.stimulus_max_occurrence:
            id_to_remove = self.agent_status.current_id
            for j in range(0, len(self.stimuli)):
                if self.stimuli[j].id == id_to_remove:
                    new_id = len(self.stimuli) + self.replacement_stimulus_counter
                    self.replacement_stimulus_counter += 1
                    self.stimuli[j] = Stimulus(id=new_id, emo_intensity=self.stimuli[j].emo_intensity,
                                               p_occurrence=self.stimuli[j].p_occurrence, resolvable=self.stimuli[j].resolvable)

    def get_original_intensity(self, stimulus_id):
        for i in range(0, len(self.agent_status.stimuliAppraisals)):
            if self.agent_status.stimuliAppraisals[i].id == stimulus_id:
                return self.agent_status.stimuliAppraisals[i].emo_intensity

    def render(self, mode='human'):
        '''

        :param mode:
        :return:
        '''
        if mode != 'human':
            raise NotImplementedError()
        for i in range(0, len(self.agent_status.stimuliAppraisals)):
            if self.agent_status.stimuliAppraisals[i].id == self.agent_status.current_id:
                self.current_appraisal = self.agent_status.stimuliAppraisals[i]

        print({'timepoint': 1, 'emo_intensity': self.agent_status.current_emo_intensity,
               'stimulus id': self.agent_status.current_id})
        print(self.current_appraisal.get_dict())
