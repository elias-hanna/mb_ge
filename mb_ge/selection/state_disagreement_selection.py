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
        
    def select_element_from_cell_archive(self, archive):
        most_disagreed_elements = []
        for cell in archive._archive.values():
            most_disagreed_elements.append(self.select_element_from_element_list(cell._elements))
        return self.select_element_from_element_list(most_disagreed_elements)
    
    def select_element_from_element_list(self, elements):
        most_disagreed_element = None
        max_disagr = None
        if len(elements) == 1:
            return elements[0]
        for element in elements:
            last_obs = element.trajectory[-1]
            mean_disagreements = []
            for _ in range(self.nb_of_samples_per_state):
                action = np.random.uniform(low=-1, high=1)
                _, disagreement = self._model.forward(action, last_obs, mean=True, disagr=True)
                mean_disagreements.append(disagreement)

            if mean_disagreements == [] or mean_disagreements[0] == []:
                continue
            mean_disagr = np.mean([np.mean(disagr.detach().numpy())
                                   for disagr in mean_disagreements])
            if most_disagreed_element is not None:
                if mean_disagr > max_disagr:
                    most_disagreed_element = element
                    max_disagr = mean_disagr
            else:
                most_disagreed_element = element
                max_disagr = mean_disagr
        return most_disagreed_element