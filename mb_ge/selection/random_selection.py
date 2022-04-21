import random

## Local imports
from mb_ge.selection.selection import SelectionMethod

class RandomSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        # selected_cell = random.choice(list(archive._archive.values()))
        # return self.select_element_from_element_list(selected_cell._elements)
        cell_list = list(archive._archive.values())
        selected_cell_list = random.choices(cell_list, k=len(cell_list))
        for selected_cell in selected_cell_list:
            el_list = selected_cell._elements
            selected_element_list = self.select_element_from_element_list(el_list,
                                                                          k=len(el_list))
            for selected_element in selected_element_list:
                if self._horizon_check(selected_element):
                    return selected_element
        return None
    
    def select_element_from_element_list(self, elements, k=1):
        elements_ordered = random.choices(elements, k=k)
        return self._get_horizon_checked_element_list(elements_ordered)[:k]
