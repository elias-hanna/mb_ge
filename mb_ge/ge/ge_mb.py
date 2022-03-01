import numpy as np
from mb_ge.utils.element import Element
from mb_ge.ge.ge import GoExplore
import copy
import os

class ModelBasedGoExplore(GoExplore):
    def __init__(self, params=None, gym_env=None, cell_selection_method=None,
                 transfer_selection_method=None, go_method=None, exploration_method=None,
                 state_archive=None, dynamics_model=None):
        super().__init__(params=params, gym_env=gym_env, selection_method=cell_selection_method,
                         go_method=go_method, exploration_method=exploration_method,
                         state_archive=state_archive)
        self._dynamics_model = dynamics_model(params=params)
        params['model'] = self._dynamics_model # grab the ref to pass it to selection methods
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

        budget_used = 0
        i_budget_used = 0
        done = False
        itr = 0
        h_max = 40
        h_min = 10
        self.h_exploration = h_min
        while budget_used < self.budget and not done:
            ## Update horizon length
            self.h_exploration = int(max(h_min,
                                     np.floor((budget_used/self.budget)*(h_max-h_min)+h_min)))
            ## Reset environment
            obs = self.gym_env.reset()
            # Select a state to return from the archive
            el = self._cell_selection_method.select_element_from_cell_archive(self.state_archive)
            # import pdb; pdb.set_trace()
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
            self.observed_transitions.append(transitions)
            # Update used budget
            i_budget_used += i_b_used
            budget_used += b_used
            itr += 1
            print(f'b_used: {budget_used} | i_b_used: {i_budget_used} | total_b: {self.budget} | current_model_horizon: {self.h_exploration}')
            # Train the dynamics model
            self._dynamics_model.add_samples_from_transitions(transitions)
            if itr % self.model_update_rate == 0:
                self._dynamics_model.train()
            if itr % self.dump_rate == 0:
                path_to_dir_to_create = os.path.join(self.dump_path, f'results_{itr}')
                os.makedirs(path_to_dir_to_create, exist_ok=True)
                self.state_archive.visualize(budget_used, itr=itr)
                for key in self.state_archive._archive.keys():
                    np.save(f'{self.dump_path}/results_{itr}/archive_cell_{key}_itr_{itr}',
                            self.state_archive._archive[key]._elements)

        path_to_dir_to_create = os.path.join(self.dump_path, f'results_final')
        os.makedirs(path_to_dir_to_create, exist_ok=True)
        self.state_archive.visualize(budget_used, itr='final')
        for key in self.state_archive._archive.keys():
            np.save(f'{self.dump_path}/results_final/archive_cell_{key}_final',
                    self.state_archive._archive[key]._elements)
        np.save(f'all_transitions_{self.budget}', np.array(self.observed_transitions))
