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
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            import os
            curr_dir = os.getcwd()
            self.dump_path = curr_dir
            
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
        while budget_used < self.budget and not done:
            obs = self.gym_env.reset()
            ## Select a state to return from the archive
            el = self._cell_selection_method.select_element_from_cell_archive(self.state_archive)
            ## Go back to the selected state
            transitions, b_used = self._go_method.go(self.gym_env, el)
            budget_used += b_used
            ## Explore from the selected state
            elements, b_used = self._exploration_method(self.gym_env, el, self.h_exploration)
            ## Update archive and other datasets
            for elem in elements:
                self.state_archive.add(elem)
            ## OPTIONNAL JUST HERE TO GATHER DATA FOR FULL MODEL
            if len(transitions) > 1 and self.dump_all_transitions:
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
                tmp[len(self.observed_transitions):
                    len(self.observed_transitions) + len(new_trs)] = new_trs
                self.observed_transitions = np.unique(tmp, axis=0)
            ## Update used budget
            budget_used += b_used
            itr += 1
            print(f'b_used: {budget_used} | total_b: {self.budget}')
            if itr%self.dump_rate == 0:
                path_to_dir_to_create = os.path.join(self.dump_path, f'results_{itr}')
                os.makedirs(path_to_dir_to_create, exist_ok=True)
                self.state_archive.visualize(budget_used, itr=itr)
                # import pdb; pdb.set_trace()
                total_num_of_els = 0
                all_max_len_desc = []
                all_max_len_params = []
                all_max_len_traj = []
                for cell in self.state_archive._archive.values():
                    total_num_of_els += len(cell._elements)
                    len_desc_in_cell = []
                    len_params_in_cell = []
                    len_traj_in_cell = []
                    for el in cell._elements:
                        if el.descriptor!=[] and el.policy_parameters!=[] and el.trajectory!=[]:
                            len_desc_in_cell.append(len(el.descriptor))
                            len_params_in_cell.append(len(el.policy_parameters))
                            len_traj_in_cell.append(len(el.trajectory))
                    if len_desc_in_cell!=[] and len_params_in_cell!=[] and len_traj_in_cell!=[]:
                        all_max_len_desc.append(max(len_desc_in_cell))
                        all_max_len_params.append(max(len_params_in_cell))
                        all_max_len_traj.append(max(len_traj_in_cell))

                # total_num_of_els = sum([len(cell._elements)
                                        # for cell in self.state_archive._archive.values()])
                # len_desc = len(list(self.state_archive._archive.values())[0]._elements[0].descriptor)
                # len_params = len(list(self.state_archive._archive.values())[0]._elements[0].policy_parameters)
                # max_traj_len = max([len(list(self.state_archive._archive.values())[0]._elements)])
                if all_max_len_desc != [] and all_max_len_params != [] and all_max_len_traj != []:
                    len_desc = max(all_max_len_desc)
                    len_params = max(all_max_len_params)
                    len_traj = max(all_max_len_traj)
                    descriptors = np.zeros((total_num_of_els, len_desc))
                    prev_descriptors = np.zeros((total_num_of_els, len_desc))
                    params = np.zeros((total_num_of_els, len_params))                
                    count = 0
                    for key in self.state_archive._archive.keys():
                        for el in self.state_archive._archive[key]._elements:
                            descriptors[count, :] = copy.copy(el.descriptor)
                            prev_descriptors[count, :] = copy.copy(el.previous_element.descriptor)
                            params[count, :] = copy.copy(el.policy_parameters)
                            count += 1
                    np.savez(f'{self.dump_path}/results_{itr}/descriptors', descriptors)
                    np.savez(f'{self.dump_path}/results_{itr}/prev_descriptors', descriptors)
                    np.savez(f'{self.dump_path}/results_{itr}/params', descriptors)

                # for key in self.state_archive._archive.keys():
                    # np.savez_compressed(f'{self.dump_path}/results_{itr}/archive_cell_{key}_itr_{itr}',
                            # self.state_archive._archive[key]._elements)

        path_to_dir_to_create = os.path.join(self.dump_path, f'results_final')
        os.makedirs(path_to_dir_to_create, exist_ok=True)
        self.state_archive.visualize(budget_used, itr='final')
        np.savez(f'{self.dump_path}/results_{itr}/descriptors', descriptors)
        np.savez(f'{self.dump_path}/results_{itr}/prev_descriptors', descriptors)
        np.savez(f'{self.dump_path}/results_{itr}/params', descriptors)
                
        # for key in self.state_archive._archive.keys():
            # np.savez_compressed(f'{self.dump_path}/results_final/archive_cell_{key}_final',
                    # self.state_archive._archive[key]._elements)
        if len(self.observed_transitions) > 1 and self.dump_all_transitions:
            np.savez_compressed(f'{self.dump_path}/results_final/all_transitions_{self.budget}',
                                np.array(self.observed_transitions))

    def __call__(self):
        return self._exploration_phase()
