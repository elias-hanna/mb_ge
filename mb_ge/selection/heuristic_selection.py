import random
import math

## Local imports
from mb_ge.selection.selection import SelectionMethod

class HeuristicSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        all_cells = archive._archive.values()
        weights = [1/(math.sqrt(0.5*cell.visit_count + 1)) for cell in all_cells]
        selected_cell_list = random.choices([cell for cell in all_cells],
                                            k=len(all_cells), weights=weights)
        for selected_cell in selected_cell_list:
            el_list = selected_cell._elements
            selected_element_list = self.select_element_from_element_list(el_list,
                                                                          len(el_list))
            for selected_element in selected_element_list:
                if self._horizon_check(selected_element):
                    return selected_element
        return None
    
    def select_element_from_element_list(self, elements, k=1):
        elements_ordered = random.choices(elements, k=k)
        return self._get_horizon_checked_element_list(elements_ordered)[:k]
