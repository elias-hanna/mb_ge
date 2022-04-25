from abc import abstractmethod
import os
import numpy as np
import copy

class Archive():
    def __init__(self, params=None):
        ## Archive is a dict
        self._archive = dict()
        self._process_params(params)
        
    def _process_params(self, params):
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            curr_dir = os.getcwd()
            self.dump_path = curr_dir
    
    @abstractmethod
    def add(self, element):
        """
        Adds an element to the archive, given the archive adding rules.

        Args:
            element: element to be added to the archive
        """
        raise NotImplementedError

    def get_all_elements(self):
        all_elements = []
        for cell in self._archive.values():
            all_elements += cell._elements
        return all_elements
        
    @abstractmethod
    def compare(self, element1, element2):
        """
        Compare two elements given the archive rules.

        Args:
            element1: element to be compared to element2
            element2: element to be compared to element1

        Returns:
            result: 1 if element1 > element2, -1 if element2 > element1
        """
        raise NotImplementedError

    def dump_archive(self, dump_path, budget_used, itr):
        total_num_of_els = 0
        all_max_len_desc = []
        all_max_len_params = []
        all_max_len_traj = []
        for cell in self._archive.values():
            total_num_of_els += len(cell._elements)
            len_desc_in_cell = []
            len_params_in_cell = []
            len_traj_in_cell = []
            for el in cell._elements:
                if el.descriptor is not None and el.policy_parameters is not None \
                   and el.trajectory is not None:
                    len_desc_in_cell.append(len(el.descriptor))
                    len_params_in_cell.append(len(el.policy_parameters))
                    len_traj_in_cell.append(len(el.trajectory))
                if len_desc_in_cell!=[] and len_params_in_cell!=[] and len_traj_in_cell!=[]:
                    all_max_len_desc.append(max(len_desc_in_cell))
                    all_max_len_params.append(max(len_params_in_cell))
                    all_max_len_traj.append(max(len_traj_in_cell))
                    
        if all_max_len_desc != [] and all_max_len_params != [] and all_max_len_traj != []:
            len_desc = max(all_max_len_desc)
            len_params = max(all_max_len_params)
            len_traj = max(all_max_len_traj)
            descriptors = np.zeros((total_num_of_els, len_desc))
            prev_descriptors = np.zeros((total_num_of_els, len_desc))
            params = np.zeros((total_num_of_els, len_params))
            policy_horizon = np.zeros((total_num_of_els))
            count = 0
            for key in self._archive.keys():
                for el in self._archive[key]._elements:
                    descriptors[count, :] = copy.copy(el.descriptor)
                    if el.previous_element is not None:
                        prev_descriptors[count, :] = copy.copy(el.previous_element.descriptor)
                    else:
                        prev_descriptors[count, :] = copy.copy(el.descriptor)
                    params[count, :] = copy.copy(el.policy_parameters)
                    policy_horizon[count] = len(el.trajectory)
                    count += 1
            np.savez(f'{dump_path}/results_{itr}/descriptors', descriptors)
            np.savez(f'{dump_path}/results_{itr}/prev_descriptors', prev_descriptors)
            np.savez(f'{dump_path}/results_{itr}/params', params)
            np.savez(f'{dump_path}/results_{itr}/policy_horizon', policy_horizon)
