import random
import math

## Local imports
from mb_ge.selection.selection import SelectionMethod

class HeuristicSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        weights = [1/(math.sqrt(0.5*cell.visit_count + 1)) for cell in archive._archive.values()]
        selected_cell_list = random.choices([cell for cell in archive._archive.values()],
                                            k=len(cell), weights=weights)
        for selected_cell in selected_cell_list:
            el_list = selected_cell._elements
            selected_element_list = self.select_element_from_element_list(el_list,
                                                                          len(el_list))
            for selected_element in selected_element_list:
                if self._horizon_check(selected_element):
                    return selected_element
        return None
    
    def select_element_from_element_list(self, elements, num_of_els):
        return random.choices(elements, k=num_of_els)
