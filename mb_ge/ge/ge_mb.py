import numpy as np
from mb_ge.utils.element import Element
from mb_ge.ge.ge import GoExplore
import copy
import os

## local includes
from mb_ge.visualization.discretized_state_space_visualization import DiscretizedStateSpaceVisualization
from mb_ge.visualization.test_trajectories_visualization import TestTrajectoriesVisualization

class ModelBasedGoExplore(GoExplore):
    def __init__(self, params=None, gym_env=None, cell_selection_method=None,
                 transfer_selection_method=None, go_method=None, exploration_method=None,
                 state_archive=None, dynamics_model=None):
        self._dynamics_model = dynamics_model(params=params)
        params['model'] = self._dynamics_model # grab the ref to pass it to selection methods
        super().__init__(params=params, gym_env=gym_env,
                         cell_selection_method=cell_selection_method,
                         go_method=go_method, exploration_method=exploration_method,
                         state_archive=state_archive)
        self._cell_selection_method = cell_selection_method(params=params)
        self._transfer_selection_method = transfer_selection_method(params=params)
        self._actions_sampled = np.random.uniform(low=-1, high=1,
                                                  size=(self.nb_of_samples_per_state, self._action_dim))
        ## Some data fields for dump
        self._min_disagr = []
        self._max_disagr = []
        ## Visualization methods
        self._test_trajectories_visualization = TestTrajectoriesVisualization(params=params)
        self._discretized_state_space_visualization = DiscretizedStateSpaceVisualization(params=params)
        ## Need to pass whats below as a path to testtrajectories processing
        self._test_trajectories = [] # each element of each traj is in the form of (A, S, NS) 
        ## Create directories for dumps
        path_to_create = os.path.join(self.dump_path, 'results_test_trajectories_vis/disagr')
        os.makedirs(path_to_create, exist_ok=True)
        path_to_create = os.path.join(self.dump_path, 'results_test_trajectories_vis/pred_error')
        os.makedirs(path_to_create, exist_ok=True)
        path_to_create = os.path.join(self.dump_path, 'results_discretized_ss_vis')
        os.makedirs(path_to_create, exist_ok=True)
        
    def _process_params(self, params):
        super()._process_params(params)
        if 'model_update_rate' in params:
            self.model_update_rate = params['model_update_rate']
        else:
            self.model_update_rate = 10
        if 'nb_of_samples_per_state' in params:
            self.nb_of_samples_per_state = params['nb_of_samples_per_state']
        else:
            raise Exception('ModelBasedGoExplore _process_params error: nb_of_samples_per_state not in params')
        if 'action_min' in params:
            self._action_min = params['action_min']
        else:
            print('Warning: using default action min value (-1)')
            self._action_min = -1
        if 'action_max' in params:
            self._action_max = params['action_max']
        else:
            print('Warning: using default action max value (1)')
            self._action_max = 1
        if 'action_dim' in params['dynamics_model_params']:
            self._action_dim = params['dynamics_model_params']['action_dim']
        else:
            raise Exception('ModelBasedGoExplore _process_params error: action_dim not in params')

    def _correct_el(self, el, transitions):
        trajectory = []
        for t in transitions:
            trajectory.append(copy.copy(t[1]))
        el.descriptor = trajectory[-1][:3]
        el.trajectory = trajectory[-self.h_exploration:]
        el.disagreement = 0. # no disagreement on this traj since we experienced it on real system

    def _update(self, itr, budget_used, transitions):
        prev_e = self.e
        to_print = super()._update(itr, budget_used)
        # Add samples to dynamics model trainer
        for el_transitions in transitions:
            self._dynamics_model.add_samples_from_transitions(el_transitions)
        # Train the dynamics model
        if self.e - prev_e != 0:
            self._dynamics_model.train()
            self._actions_sampled = np.random.uniform(low=-1, high=1,
                                    size=(self.nb_of_samples_per_state, self._action_dim))
            all_elements = self.state_archive.get_all_elements()
            self._update_disagreement(all_elements, 'state')
            ## Various model dumps
            self._model_dump(itr, budget_used, plot=False,
                             plot_disagr=True, plot_novelty=True)
            
            
        return to_print

    def _update_state_disagr(self, elements):
        A = np.tile(self._actions_sampled, (len(elements), 1))

        all_s = []
        min_d = np.inf
        max_d = 0
        # Get all states to estimate uncertainty for
        for element in elements:
            all_s.append(element.trajectory[-1])
        S = np.repeat(all_s, self.nb_of_samples_per_state, axis=0)
        # Batch prediction
        batch_pred_delta_ns, batch_disagreement = self._dynamics_model.forward_multiple(A, S,
                                                                                        mean=True,
                                                                                        disagr=True)
        end_state_disagrs = []
        for i in range(len(elements)):
            el_disagrs = batch_disagreement[i*self.nb_of_samples_per_state:
                                            i*self.nb_of_samples_per_state+
                                            self.nb_of_samples_per_state]
            
            end_state_disagrs.append(np.mean([np.mean(disagr.detach().numpy()) for disagr in el_disagrs]))
            elements[i].end_state_disagr = end_state_disagrs[-1]
            if end_state_disagrs[-1] < min_d:
                min_d = end_state_disagrs[-1]
            if end_state_disagrs[-1] > max_d:
                max_d = end_state_disagrs[-1]
        ## Add min and max disagr
        self._min_disagr.append(min_d)
        self._max_disagr.append(max_d)
        
    def _update_trajectory_disagr(self, elements):
        all_s = []
        # Get all states to estimate uncertainty for
        for element in elements:
            element.trajectory_disagr = np.mean([np.mean(disagr.detach().numpy())
                                                 for disagr in element.disagreement])
        
    def _update_disagreement(self, elements, mode):
        if mode == 'state':
            self._update_state_disagr(elements)
        if mode == 'trajectory':
            self._update_trajectory_disagr(elements)

    def _model_dump(self, itr, budget_used, plot=False,
                    plot_disagr=False, plot_novelty=False):
        ## Dump last min-max disagr data
        path_to_file = os.path.join(self.dump_path, 'min_max_disagr.npz')
        np.savez(path_to_file, min_disagr=self._min_disagr, max_disagr=self._max_disagr)
        ## Plot test trajectories data (disagreement and prediction error)
        self._test_trajectories_visualization.dump_plots(budget_used,
                                                         itr='test_trajectories_vis')
        ## Plot discretized state space visualization
        self._discretized_state_space_visualization.dump_plots(budget_used,
                                                               itr='discretized_ss_vis')
        
    def _exploration_phase(self):
        # reset gym environment
        obs = self.gym_env.reset()
        # add first state to state_archive
        init_elem = Element(descriptor=obs[:3], trajectory=[obs], reward=0.,
                            sim_state={'qpos': self.gym_env.sim.data.qpos,
                                       'qvel': self.gym_env.sim.data.qvel})
        self.state_archive.add(init_elem)
        itr = 0
        budget_used = 0
        sim_budget_used = 0
        i_budget_used = 0
        done = False

        self.budget_dump_cpt = 0
        self.sim_budget_dump_cpt = 0
        flag = False
        while budget_used < self.budget and not done:
            b_used = 0
            sim_b_used = 0
            ## Reset environment
            obs = self.gym_env.reset()
            # Select a state to return from the archive
            el = self._cell_selection_method.select_element_from_cell_archive(self.state_archive,
                                                                              exploration_horizon=self.h_exploration)
            # Go to and Explore in imagination from the selected state
            i_elements, i_b_used = self._exploration_method(self._dynamics_model, el,
                                                            self.h_exploration, eval_on_model=True)

            # Update novelty
            self._update_novelty(i_elements, no_add=True)

            # Compute disagreement for imagined exploration elements
            # Compute trajectory disagreement for transfer selection
            self._update_disagreement(i_elements, 'state')
            
            #####################################################
            sel_i_els = self._transfer_selection_method.select_element_from_element_list(i_elements)
            # Go to the selected state(s) on real system
            transitions = []
            for sel_i_el in sel_i_els:
                loc_trans, loc_b_used = self._go_method.go(self.gym_env, sel_i_el)
                transitions.append(loc_trans)
                ## Update sim and real system budget used for each se_i_el we go to
                budget_used += loc_b_used
                sim_b_used += self.h_exploration

                # Correct sel_i_els to have the right trajectory
                self._correct_el(sel_i_el, loc_trans)

            # Update novelty
            self._update_novelty(sel_i_els)

            # Compute disagreement for transferred elements
            # Compute state disagreement for next cell selection
            self._update_disagreement(sel_i_els, 'state')  

            # Update archive and other datasets
            for sel_i_el in sel_i_els:
                self.state_archive.add(sel_i_el)
                
            ## OPTIONNAL JUST HERE TO GATHER DATA FOR FULL MODEL
            if len(transitions) > 1 and (self.dump_all_transitions
                                         or self.epoch_mode == "unique_fixed_steps"):
                for el_transitions in transitions:
                    self.append_new_transitions(el_transitions)

            # Update used budget
            i_budget_used += i_b_used
            # budget_used += b_used
            sim_budget_used += sim_b_used
            unique_trs_observed = len(self.observed_transitions)
            itr += 1
                
            # Verbose
            to_print = f'b_used: {budget_used} | i_b_used: {i_budget_used} | total_b: {self.budget} | current_exploration_horizon: {self.h_exploration} '
            
            # Update epoch, exploration horizon and model if relevant
            to_print += self._update(itr, budget_used, transitions)
            # Dump data
            self._dump(itr, budget_used, sim_budget_used, plot=True,
                       plot_novelty=True, plot_disagr=True)
            # Print
            print(to_print)

        # self.state_archive.dump_archive(self.dump_path, budget_used, 'final')

        if len(self.observed_transitions) > 1 and self.dump_all_transitions:
            np.save(f'all_transitions_{self.budget}', np.array(self.observed_transitions))
