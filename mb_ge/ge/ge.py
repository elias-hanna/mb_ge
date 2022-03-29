import numpy as np
from mb_ge.utils.element import Element
import os
import copy

class GoExplore():
    def __init__(self, params=None, gym_env=None, cell_selection_method=None,
                 go_method=None, exploration_method=None, state_archive=None):
        ## Process run parameters
        self._process_params(params)
        ## Intialize functors (do this in params?)
        self._cell_selection_method = cell_selection_method(params=params)
        self._go_method = go_method(params=params)
        self._exploration_method = exploration_method(params=params)
        ## Intialize state_archive
        self.state_archive = state_archive(params)
        self.gym_env = gym_env ## needs to be an already initialized env
        ## Transition Dataset initialization (used to train Dynamics Model)
        self.action_space_dim = self.gym_env.action_space.shape[0]
        self.observation_space_dim = self.gym_env.observation_space.shape[0]
        self.observed_transitions = []

    def _process_params(self, params):
        if 'budget' in params:
            self.budget = params['budget']
        if 'exploration_horizon' in params:
            self.h_exploration = params['exploration_horizon']
        if 'controller_type' in params:
            self.controller_type = params['controller_type']
        if 'dump_all_transitions' in params:
            self.dump_all_transitions = params['dump_all_transitions']
        if 'dump_rate' in params:
            self.dump_rate = params['dump_rate']
        else:
            self.dump_rate = 100
        if 'dump_checkpoints' in params:
            self._dump_checkpoints = params['dump_checkpoints']
        else:
            raise Exception('GoExplore _process_params error: dump_checkpoints not in params')
        if 'use_variable_model_horizon' in params:
            self._use_variable_model_horizon = params['use_variable_model_horizon']
            if self._use_variable_model_horizon:
                if 'min_horizon' in params:
                    self._min_horizon = params['min_horizon']
                else:
                    raise Exception('GoExplore _process_params error: min_horizon not in params')
                if 'max_horizon' in params:
                    self._max_horizon = params['max_horizon']
                else:
                    raise Exception('GoExplore _process_params error: max_horizon not in params')
                if 'horizon_starting_epoch' in params:
                    self._horizon_starting_epoch = params['horizon_starting_epoch']
                else:
                    raise Exception('GoExplore _process_params error: horizon_starting_epoch not in params')
                if 'horizon_ending_epoch' in params:
                    self._horizon_ending_epoch = params['horizon_ending_epoch']
                else:
                    raise Exception('GoExplore _process_params error: horizon_ending_epoch not in params')
                if 'model_update_rate' in params:
                    self.model_update_rate = params['model_update_rate']
                else:
                    self.model_update_rate = 10
                if 'steps_per_epoch' in params:
                    self.steps_per_epoch = params['steps_per_epoch']
                else:
                    self.steps_per_epoch = 1000
        if 'epoch_mode' in params:
                    self.epoch_mode = params['epoch_mode']
        else:
            self.epoch_mode = 'None'
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            import os
            curr_dir = os.getcwd()
            self.dump_path = curr_dir
            
    def append_new_transitions(self, transitions):
        A = []
        S = []
        NS = []
        for i in range(len(transitions) - 1):
            A.append(copy.copy(transitions[i][0]))
            S.append(copy.copy(transitions[i][1]))
            NS.append(copy.copy(transitions[i+1][1] - transitions[i][1]))
        A = np.array(A)
        S = np.array(S)
        NS = np.array(NS)
        new_trs = np.concatenate([S, A, NS], axis=1)
        tmp = np.zeros((len(new_trs)+len(self.observed_transitions), new_trs.shape[1]))
        if len(self.observed_transitions) > 0:
            tmp[0:len(self.observed_transitions)] = self.observed_transitions
        tmp[len(self.observed_transitions):len(self.observed_transitions) + len(new_trs)] = new_trs
        self.observed_transitions = np.unique(tmp, axis=0)
                
    def _exploration_phase(self):
        ## reset gym environment
        obs = self.gym_env.reset()
        ## add first state to state_archive
        init_elem = Element(descriptor=obs[:3], trajectory=[obs], reward=0.,
                            sim_state={'qpos': self.gym_env.sim.data.qpos,
                                       'qvel': self.gym_env.sim.data.qvel})
        self.state_archive.add(init_elem)
        itr = 0
        budget_used = 0
        done = False

        budget_dump_cpt = 0

        # Variable horizon variables
        if self._use_variable_model_horizon:
            e = 0
            x = self._min_horizon
            y = self._max_horizon
            
            a = self._horizon_starting_epoch
            b = self._horizon_ending_epoch
            next_target_budget = self.steps_per_epoch
            
        while budget_used < self.budget and not done:
            ## Update horizon length
            if self._use_variable_model_horizon:
                if e >= a: # normal case
                    self.h_exploration = int(min(max(x + ((e - a)/(b - a))*(y - x), x), y))
                elif e < a:
                    self.h_exploration = x
                elif e > b:
                    self.h_exploration = y

            ## Reset environment
            obs = self.gym_env.reset()
            ## Select a state to return from the archive
            el = self._cell_selection_method.select_element_from_cell_archive(self.state_archive)
            ## Go back to the selected state
            transitions, b_used = self._go_method.go(self.gym_env, el)
            ## Explore from the selected state
            elements, b_used_expl = self._exploration_method(self.gym_env, el, self.h_exploration)
            b_used += b_used_expl
            print(b_used)
            ## Update archive and other datasets
            for elem in elements:
                self.state_archive.add(elem)
            ## OPTIONNAL JUST HERE TO GATHER DATA FOR FULL MODEL
            if len(transitions) > 1 and (self.dump_all_transitions
                                         or self.epoch_mode == "unique_fixed_steps"):
                self.append_new_transitions(transitions)

            ## Update used budget
            budget_used += b_used
            unique_trs_observed = len(self.observed_transitions)
            itr += 1
            
            # Verbose
            to_print = f'b_used: {budget_used} | total_b: {self.budget} | current_exploration_horizon: {self.h_exploration} '
            
            # Update exploration horizon
            if self._use_variable_model_horizon:
                if self.epoch_mode == 'model_update' and itr % self.model_update_rate == 0:
                    e += 1
                elif self.epoch_mode == 'fixed_steps' and budget_used >= next_target_budget:
                    e += 1
                    next_target_budget += self.steps_per_epoch
                elif self.epoch_mode == 'unique_fixed_steps':
                    to_print += f'| unique_trs_observed: {unique_trs_observed} '
                    if unique_trs_observed >= next_target_budget:
                        e += 1
                        next_target_budget += self.steps_per_epoch
                to_print += f'| current_epoch: {e}'

            # Dump data
            # if itr%self.dump_rate == 0:
                # self.state_archive.dump_archive(self.dump_path, budget_used, itr)
            if budget_used >= self._dump_checkpoints[budget_dump_cpt]:
                self.state_archive.dump_archive(self.dump_path, budget_used,
                                                self._dump_checkpoints[budget_dump_cpt])
                budget_dump_cpt += 1
                
            # Actually print
            print(to_print)

        self.state_archive.dump_archive(self.dump_path, budget_used, 'final')

        if len(self.observed_transitions) > 1 and self.dump_all_transitions:
            np.savez_compressed(f'{self.dump_path}/results_final/all_transitions_{self.budget}',
                                np.array(self.observed_transitions))

    def __call__(self):
        return self._exploration_phase()
