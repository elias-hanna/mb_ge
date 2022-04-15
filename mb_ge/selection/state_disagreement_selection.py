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

    def _batch_eval_all_elements(self, elements):
        actions = np.random.uniform(low=-1, high=1,
                                    size=(self.nb_of_samples_per_state, self._action_dim))

        A = np.tile(actions, (len(elements), 1))

        all_s = []
        # Get all states to estimate uncertainty for
        for element in elements:
            all_s.append(element.trajectory[-1])
        S = np.repeat(all_s, self.nb_of_samples_per_state, axis=0)
        # Batch prediction
        batch_pred_delta_ns, batch_disagreement = self._model.forward_multiple(A, S, mean=True,
                                                                               disagr=True)
        mean_disagrs = []
        for i in range(len(elements)):
            el_disagrs = batch_disagreement[i*self.nb_of_samples_per_state:
                                            i*self.nb_of_samples_per_state+
                                            self.nb_of_samples_per_state]
            
            mean_disagrs.append(np.mean([np.mean(disagr.detach().numpy()) for disagr in el_disagrs]))
        try:
            elements_ordered = [el for _, el in sorted(zip(mean_disagrs, elements), reverse=True)]
        except Exception as e:
            import pdb; pdb.set_trace()
        return elements_ordered
                
    def select_element_from_cell_archive(self, archive):
        # most_disagreed_elements = []
        # for cell in archive._archive.values():
        #     most_disagreed_elements.append(self.select_element_from_element_list(cell._elements))
        # return self.select_element_from_element_list(most_disagreed_elements)
    
        # all_elements = []
        # for cell in archive._archive.values():
            # all_elements += cell._elements
        all_elements = archive.get_all_elements()
        all_elements_ordered = self._batch_eval_all_elements(all_elements)
        # all_elements_ordered = self.get_ordered_element_list(all_elements)
        for selected_element in all_elements_ordered:
            if self._horizon_check(selected_element):
                return selected_element
        return None
        
    def select_element_from_element_list(self, elements):
        most_disagreed_element = None
        max_disagr = None
        # if len(elements) == 1:
            # return [elements[0]]
        for element in elements:
            last_obs = element.trajectory[-1]
            mean_disagreements = []
            for _ in range(self.nb_of_samples_per_state):
                action = np.random.uniform(low=-1, high=1, size=self._action_dim)
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

    def get_ordered_element_list(self, elements):
        disagrs = []
        els = []
        # if len(elements) == 1:
            # return elements[0]
        for element in elements:
            last_obs = element.trajectory[-1]
            mean_disagreements = []
            for _ in range(self.nb_of_samples_per_state):
                action = np.random.uniform(low=-1, high=1, size=self._action_dim)
                _, disagreement = self._model.forward(action, last_obs, mean=True, disagr=True)
                mean_disagreements.append(disagreement)
            if mean_disagreements == [] or mean_disagreements[0] == []:
                continue
            mean_disagr = np.mean([np.mean(disagr.detach().numpy())
                                   for disagr in mean_disagreements])
            disagrs.append(mean_disagr)
            els.append(element)

        try:
            elements_ordered = [el for _, el in sorted(zip(disagrs, els), reverse=True)]
        except Exception as e:
            import pdb; pdb.set_trace()
        return elements_ordered
