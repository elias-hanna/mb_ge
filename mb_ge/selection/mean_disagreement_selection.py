import random
import numpy as np

## Local imports
from mb_ge.selection.selection import SelectionMethod

class MeanDisagreementSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        most_disagreed_elements = []
        for cell in archive._archive.values():
            most_disagreed_elements.append(self.select_element_from_element_list(cell._elements))
        return self.select_element_from_element_list(most_disagreed_elements)
    
    def select_element_from_element_list(self, elements):
        most_disagreed_element = None
        max_disagr = None
        for element in elements:
            disagreements = element.disagreement ## disagreement for each traj step
            mean_disagr = np.mean([np.mean(disagr.detach().numpy()) for disagr in disagreements])
            if most_disagreed_element is not None:
                if mean_disagr > max_disagr:
                    most_disagreed_element = element
                    max_disagr = mean_disagr
            else:
                most_disagreed_element = element
                max_disagr = mean_disagr
        return most_disagreed_element
