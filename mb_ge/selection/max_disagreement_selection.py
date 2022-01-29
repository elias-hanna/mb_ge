import random
import numpy as np

## Local imports
from mb_ge.selection.selection import SelectionMethod

class MaxDisagreementSelection(SelectionMethod):
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
            disagreements = element.disagreement ## disagreement for each traj step
            if disagreements == []:
                continue
            max_disagr = np.max([np.max(disagr.detach().numpy()) for disagr in disagreements])
            if most_disagreed_element is not None:
                if max_disagr > max_disagr:
                    most_disagreed_element = element
                    max_disagr = max_disagr
            else:
                most_disagreed_element = element
                max_disagr = max_disagr
        return most_disagreed_element
