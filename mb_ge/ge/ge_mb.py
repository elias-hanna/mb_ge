import numpy as np
from mb_ge.utils.element import Element
from mb_ge.ge.ge import GoExplore
import copy
import os


#### TEMPORARY
from mb_ge.selection.state_disagreement_selection import StateDisagreementSelection

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
        self.horrible_thing = StateDisagreementSelection(params=params)

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

    def _update(self, itr, budget_used, transitions):
        prev_e = self.e
        to_print = super()._update(itr, budget_used)
        # Add samples to dynamics model trainer
        for el_transitions in transitions:
            self._dynamics_model.add_samples_from_transitions(el_transitions)
        # Train the dynamics model
        if self.e - prev_e != 0:
            self._dynamics_model.train()

        return to_print

    def _update_disagreement(self, elements):
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

        while budget_used < self.budget and not done:
            b_used = 0
            sim_b_used = 0
            ## Reset environment
            obs = self.gym_env.reset()
            # Select a state to return from the archive
            el = self._cell_selection_method.select_element_from_cell_archive(self.state_archive)
            # Go to and Explore in imagination from the selected state
            i_elements, i_b_used = self._exploration_method(self._dynamics_model, el,
                                                            self.h_exploration, eval_on_model=True)

            # Update novelty
            self._update_novelty(i_elements, no_add=True)
            
            ##### NEED TO BE REMOVED HORRIBLE#####
            all_elements = self.state_archive.get_all_elements()
            _, _ = self.horrible_thing._batch_eval_all_elements(all_elements)
            #####################################################
            sel_i_els = self._transfer_selection_method.select_element_from_element_list(i_elements)
            # Go to the selected state(s) on real system
            transitions = []
            for sel_i_el in sel_i_els:
                loc_trans, loc_b_used = self._go_method.go(self.gym_env, sel_i_el)
                # transitions, b_used = self._go_method.go(self.gym_env, sel_i_el)
                transitions.append(loc_trans)
                ## Update sim and real system budget used for each se_i_el we go to
                budget_used += loc_b_used
                sim_b_used += self.h_exploration

                # Correct sel_i_els to have the right trajectory
                self._correct_el(sel_i_el, loc_trans)

            # Update novelty
            self._update_novelty(sel_i_els)
            # print('REAL OBSERVED NOV: ', sel_i_el.novelty)
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
            self._dump(itr, budget_used, sim_budget_used, plot_novelty=True, plot_disagr=True)
            # Print
            print(to_print)

        # self.state_archive.dump_archive(self.dump_path, budget_used, 'final')

        if len(self.observed_transitions) > 1 and self.dump_all_transitions:
            np.save(f'all_transitions_{self.budget}', np.array(self.observed_transitions))
