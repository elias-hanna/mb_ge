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
        self._dynamics_model.add_samples_from_transitions(transitions)
        # Train the dynamics model
        if self.e - prev_e != 0:
            self._dynamics_model.train()

        return to_print
    
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
            
            # if budget_used+300 >= self._dump_checkpoints[self.budget_dump_cpt]:
            # if budget_used >= 500 and len(self._archive_bd_list) > self._nb_nearest_neighbors:
            # if False:
            #     disagr = []
            #     novelty = []
            #     ## Warning need to filter dominated solutions
            #     for el in i_elements:
            #         # loc_d = round(np.mean([np.mean(disagr.detach().numpy())
            #                                # for disagr in el.disagreement]),3)
            #         # loc_n = round(el.novelty, 3)
            #         loc_d = np.mean([np.mean(disagr.detach().numpy())
            #                          for disagr in el.disagreement])
            #         loc_n = el.novelty
            #         disagr.append(loc_d)
            #         novelty.append(loc_n)
            #     non_dominated_disagr = []
            #     non_dominated_novelty = []
            #     for i in range(len(disagr)):
            #         dominated = False
            #         loc_d = disagr[i]
            #         loc_n = novelty[i]
            #         for j in range(len(disagr)):
            #             if loc_d < disagr[j] and loc_n < novelty[j]:
            #                 dominated = True
            #                 break
            #         if not dominated:
            #             non_dominated_disagr.append(loc_d)
            #             non_dominated_novelty.append(loc_n)
                    
            #     import kneed
            #     import matplotlib.pyplot as plt
            #     ## Sort by nov
            #     # sorted_pairs = sorted(zip(novelty, disagr))
            #     # tuples = zip(*sorted_pairs)
            #     # s_nov, s_disagr = [list(t) for t in tuples] 
            #     # kneedle = kneed.KneeLocator(s_nov, s_disagr)
            #     # plt.ylabel('Mean disagreeement along trajectory')
            #     # plt.xlabel('Novelty')
            #     ## Sort by disagr
            #     # sorted_pairs = sorted(zip(disagr, novelty))
            #     if len(non_dominated_disagr) >= 2:
            #         try:
            #             sorted_pairs = sorted(zip(non_dominated_disagr, non_dominated_novelty))
            #             tuples = zip(*sorted_pairs)
            #             s_disagr, s_nov = [list(t) for t in tuples] 
            #             kneedle = kneed.KneeLocator(s_disagr, s_nov, curve='concave', direction='increasing')#, interp_method='polynomial')
            #             kneedle.plot_knee_normalized()
            #             plt.xlabel('Mean disagreeement along trajectory')
            #             plt.ylabel('Novelty')
            #             # plt.show()
            #         except Exception as e:
            #             import pdb; pdb.set_trace()
            # Select a state to go to from states found in imagination

            ##### NEED TO BE REMOVED HORRIBLE#####
            all_elements = self.state_archive.get_all_elements()
            _, _ = self.horrible_thing._batch_eval_all_elements(all_elements)
            #####################################################
            sel_i_el = self._transfer_selection_method.select_element_from_element_list(i_elements)
            # Go to the selected state on real system
            transitions, b_used = self._go_method.go(self.gym_env, sel_i_el)
            
            sim_b_used += self.h_exploration
            # Correct sel_i_el to have the right trajectory
            self._correct_el(sel_i_el, transitions)

            # Update novelty
            self._update_novelty([sel_i_el])
            # print('REAL OBSERVED NOV: ', sel_i_el.novelty)
            # Update archive and other datasets
            self.state_archive.add(sel_i_el)
            ## OPTIONNAL JUST HERE TO GATHER DATA FOR FULL MODEL
            if len(transitions) > 1 and (self.dump_all_transitions
                                         or self.epoch_mode == "unique_fixed_steps"):
                self.append_new_transitions(transitions)
            # Update used budget
            i_budget_used += i_b_used
            budget_used += b_used
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
