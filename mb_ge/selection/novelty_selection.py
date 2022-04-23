import random
import numpy as np

## Local imports
from mb_ge.selection.selection import SelectionMethod

class NoveltySelection(SelectionMethod):                
    def select_element_from_cell_archive(self, archive, exploration_horizon=0):
        all_elements = archive.get_all_elements()
        all_elements_ordered = self.get_ordered_element_list(all_elements)
        for selected_element in all_elements_ordered:
            if self._horizon_check(selected_element, exploration_horizon=exploration_horizon):
                return selected_element
        return None
        
    def select_element_from_element_list(self, elements, k=1):
        elements_ordered = self.get_ordered_element_list(elements)
        return self._get_horizon_checked_element_list(elements_ordered)[:k]
    
    def get_ordered_element_list(self, elements):
        novelties = []
        els = []
        for element in elements:
            novelties.append(element.novelty)
            els.append(element)

        if len(set(novelties)) == 1: # case were nov aren't possibly computed return random
            return random.choices(elements, k=len(elements))

        elements_ordered = [el for _, el in sorted(zip(novelties, els), reverse=True)]
        
        return elements_ordered