import numpy as np
from mb_ge.utils.element import Element
import os
import copy
from sklearn.neighbors import KDTree

archive_list = []

archive_nb_to_add = 6
nb_nearest_neighbors = 15

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
        ## Update params
        self.e = 0 # current epoch (1 epoch = 1 model update and potential horizon update)
        self.next_target_budget = self.steps_per_epoch
        self._archive_bd_list = []
        self._archive_nb_to_add = 6
        self._nb_nearest_neighbors = 15
        self._dump_coverage = []
        self._dump_budget = []
        
    def _process_params(self, params):
        ## Algorithm params
        if 'budget' in params:
            self.budget = params['budget']
        if 'exploration_horizon' in params:
            self.h_exploration = params['exploration_horizon']
        if 'controller_type' in params:
            self.controller_type = params['controller_type']
        ## Epoch mode params
        if 'epoch_mode' in params:
            self.epoch_mode = params['epoch_mode']
        else:
            self.epoch_mode = 'None'
        if 'model_update_rate' in params:
            self.model_update_rate = params['model_update_rate']
        else:
            self.model_update_rate = 10
        if 'nb_eval_exploration' in params:
            self.nb_eval_exploration = params['nb_eval_exploration']
        else:
            self.model_update_rate = 10
        if 'steps_per_epoch' in params:
            self.steps_per_epoch = params['steps_per_epoch']
        else:
            self.steps_per_epoch = 1000
        ## Variable horizon params
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
                self.h_exploration = self._min_horizon

        ## Dump params
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            import os
            curr_dir = os.getcwd()
            self.dump_path = curr_dir
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

    def _update(self, itr, budget_used):
        to_print = ''
        # Update exploration horizon
        # if self._use_variable_model_horizon:

        if self.epoch_mode == 'model_update' and itr % self.model_update_rate == 0:
            self.e += 1
        elif self.epoch_mode == 'fixed_steps' and budget_used >= self.next_target_budget:
            self.e += 1
            self.next_target_budget += self.steps_per_epoch
        elif self.epoch_mode == 'unique_fixed_steps':
            to_print += f'| unique_trs_observed: {unique_trs_observed} '
            if unique_trs_observed >= self.next_target_budget:
                self.e += 1
                self.next_target_budget += self.steps_per_epoch
        to_print += f'| current_epoch: {self.e}'
        
        if self._use_variable_model_horizon:
            x = self._min_horizon
            y = self._max_horizon
            a = self._horizon_starting_epoch
            b = self._horizon_ending_epoch
            
            if self.e >= a: # normal case
                self.h_exploration = int(min(max(x + ((self.e - a)/(b - a))*(y - x), x), y))
            elif self.e < a:
                self.h_exploration = x
            elif self.e > b:
                self.h_exploration = y

        return to_print

    def _update_novelty(self, new_elements, no_add=False):
        gen_bd_list = []
        for new_el in new_elements:
            gen_bd_list.append(new_el.descriptor)
            # self._archive_bd_list.append(new_el.descriptor)
        archive_kdt = KDTree(self._archive_bd_list + gen_bd_list, leaf_size=30, metric='euclidean')

        if not no_add:
            self._archive_bd_list += gen_bd_list
            
        all_elements = self.state_archive.get_all_elements()

        all_elements += new_elements

        if len(self._archive_bd_list) > self._nb_nearest_neighbors:
            for el in all_elements:
                ## Get k-nearest neighbours to this ind
                k_dists, k_indexes = archive_kdt.query([el.descriptor],
                                                       k=self._nb_nearest_neighbors)
                el.novelty = sum(k_dists[0])/self._nb_nearest_neighbors

    def _dump(self, itr, budget_used, sim_budget_used, plot_disagr=False, plot_novelty=False):
        self._dump_coverage.append(len(self.state_archive._archive.keys()))
        self._dump_budget.append(budget_used)
        if budget_used >= self._dump_checkpoints[self.budget_dump_cpt]:
            os.makedirs(self.dump_path, exist_ok=True)
            path_to_file = os.path.join(self.dump_path, 'coverage_data.npz')
            np.savez(path_to_file, cells=self._dump_coverage, budget=self._dump_budget)
            self.state_archive.dump_archive(self.dump_path, budget_used,
                                            self._dump_checkpoints[self.budget_dump_cpt],
                                            plot_disagr=plot_disagr, plot_novelty=plot_novelty)
            self.budget_dump_cpt += 1

        if sim_budget_used >= self._dump_checkpoints[self.sim_budget_dump_cpt]:
            self.state_archive.dump_archive(self.dump_path, sim_budget_used,
                                            'sim_'+
                                            str(self._dump_checkpoints[self.sim_budget_dump_cpt]),
                                            plot_disagr=False, plot_novelty=True)
            self.sim_budget_dump_cpt += 1
        
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
        sim_budget_used = 0
        done = False

        self.budget_dump_cpt = 0
        self.sim_budget_dump_cpt = 0
            
        while budget_used < self.budget and not done:
            b_used = 0
            sim_b_used = 0
            ## Reset environment
            obs = self.gym_env.reset()
            ## Select a state to return to from the archive
            el = self._cell_selection_method.select_element_from_cell_archive(self.state_archive)
            transitions, sel_el_go_b = self._go_method.go(self.gym_env, el)
            ## Explore from the selected state
            elements, b_used_expl = self._exploration_method(self.gym_env, el, self.h_exploration)
            # import pdb; pdb.set_trace()
            self._update_novelty(elements, no_add=True)
            
            b_used += (sel_el_go_b)*len(elements) + b_used_expl
            sim_b_used += len(elements)*self.h_exploration
            # Select a state to add to archive from the exploration elements
            nb_of_el_to_add = self.nb_eval_exploration
            sel_els = self._cell_selection_method.select_element_from_element_list(elements,
                                                                                   k=nb_of_el_to_add)
            ## No need to "go" to selected states since its done on real system directly
            
            ## Update archive and other datasets
            for sel_el in sel_els:
                self.state_archive.add(sel_el)

            self._update_novelty(sel_els)
            
            ## OPTIONNAL JUST HERE TO GATHER DATA FOR FULL MODEL
            if len(transitions) > 1 and (self.dump_all_transitions
                                         or self.epoch_mode == "unique_fixed_steps"):
                self.append_new_transitions(transitions)

            ## Update used budget
            budget_used += b_used
            sim_budget_used += sim_b_used
            unique_trs_observed = len(self.observed_transitions)
            itr += 1
            
            # Verbose
            to_print = f'b_used: {budget_used} | total_b: {self.budget} | current_exploration_horizon: {self.h_exploration} '

            # Update epoch, exploration horizon and model if relevant 
            to_print += self._update(itr, budget_used)
            # Dump data
            self._dump(itr, budget_used, sim_budget_used, plot_novelty=True)
            # Print
            print(to_print)

        # self.state_archive.dump_archive(self.dump_path, budget_used, 'final')

        if len(self.observed_transitions) > 1 and self.dump_all_transitions:
            np.savez_compressed(f'{self.dump_path}/results_final/all_transitions_{self.budget}',
                                np.array(self.observed_transitions))

    def __call__(self):
        return self._exploration_phase()
