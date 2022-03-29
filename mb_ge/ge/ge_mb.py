import numpy as np
from mb_ge.utils.element import Element
from mb_ge.ge.ge import GoExplore
import copy
import os

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

    def _process_params(self, params):
        super()._process_params(params)
        if 'model_update_rate' in params:
            self.model_update_rate = params['model_update_rate']
        else:
            self.model_update_rate = 10

    def _correct_el(self, el, transitions):
        trajectory = []
        for t in transitions:
            trajectory.append(copy.copy(t[1]))
        el.descriptor = trajectory[-1][:3]
        el.trajectory = trajectory[-self.h_exploration:]
        el.disagreement = 0. # no disagreement on this traj since we experienced it on real system

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
        i_budget_used = 0
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
            # Select a state to return from the archive
            el = self._cell_selection_method.select_element_from_cell_archive(self.state_archive)
            # Go to and Explore in imagination from the selected state
            i_elements, i_b_used = self._exploration_method(self._dynamics_model, el,
                                                            self.h_exploration, eval_on_model=True)
            # Select a state to go to from states found in imagination
            sel_i_el = self._transfer_selection_method.select_element_from_element_list(i_elements)
            # Go back to the selected state
            transitions, b_used = self._go_method.go(self.gym_env, sel_i_el)
            # Correct sel_i_el to have the right trajectory
            self._correct_el(sel_i_el, transitions)
            # Update archive and other datasets
            self.state_archive.add(sel_i_el)
            ## OPTIONNAL JUST HERE TO GATHER DATA FOR FULL MODEL
            if len(transitions) > 1 and (self.dump_all_transitions
                                         or self.epoch_mode == "unique_fixed_steps"):
                self.append_new_transitions(transitions)
            print(b_used)
            # Update used budget
            i_budget_used += i_b_used
            budget_used += b_used
            unique_trs_observed = len(self.observed_transitions)
            itr += 1

            # Train the dynamics model
            self._dynamics_model.add_samples_from_transitions(transitions)
            if itr % self.model_update_rate == 0:
                self._dynamics_model.train()
                
            # Verbose
            to_print = f'b_used: {budget_used} | i_b_used: {i_budget_used} | total_b: {self.budget} | current_exploration_horizon: {self.h_exploration} '
            
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
            # if itr % self.dump_rate == 0:
                # self.state_archive.dump_archive(self.dump_path, budget_used, itr)
            if budget_used >= self._dump_checkpoints[budget_dump_cpt]:
                self.state_archive.dump_archive(self.dump_path, budget_used,
                                                self._dump_checkpoints[budget_dump_cpt])
                budget_dump_cpt += 1
                
            # Actually print
            print(to_print)

        self.state_archive.dump_archive(self.dump_path, budget_used, 'final')

        if len(self.observed_transitions) > 1 and self.dump_all_transitions:
            np.save(f'all_transitions_{self.budget}', np.array(self.observed_transitions))
