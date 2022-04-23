import random
import numpy as np

## Local imports
from mb_ge.selection.selection import SelectionMethod

class StateDisagreementSelection(SelectionMethod):        
    def _process_params(self, params):
        super()._process_params(params)
        if 'nb_of_samples_per_state' in params:
            self.nb_of_samples_per_state = params['nb_of_samples_per_state']
        else:
            raise Exception('StateDisagreementSelection _process_params error: nb_of_samples_per_state not in params')
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
        if 'model' in params:
            self._model = params['model']
        else:
            raise Exception('StateDisagreementSelection _process_params error: model not in params')
        if 'action_dim' in params['dynamics_model_params']:
            self._action_dim = params['dynamics_model_params']['action_dim']
        else:
            raise Exception('StateDisagreementSelection _process_params error: action_dim not in params')
                
    def select_element_from_cell_archive(self, archive, exploration_horizon=0, mode='state'):
        all_elements = archive.get_all_elements()
        # all_elements_ordered, _ = self._batch_eval_all_elements(all_elements)
        all_elements_ordered = self.get_ordered_element_list(all_elements, mode=mode)
        for selected_element in all_elements_ordered:
            if self._horizon_check(selected_element, exploration_horizon=exploration_horizon):
                return selected_element
        return None
        
    def select_element_from_element_list(self, elements, k=1, mode='state'):
        elements_ordered = self.get_ordered_element_list(elements, mode=mode)
        # elements_ordered, _ = self._batch_eval_all_elements(elements)
        return self._get_horizon_checked_element_list(elements_ordered)[:k]

    def get_ordered_element_list(self, elements, mode='state'):
        disagrs = []
        els = []
        for element in elements:
            if mode=='state':
                disagrs.append(element.end_state_disagr)
            if mode=='trajectory':
                disagrs.append(element.trajectory_disagr)
            els.append(element)

        elements_ordered = [el for _, el in sorted(zip(disagrs, els), reverse=True)]
        
        return elements_ordered
