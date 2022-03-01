import random

## Local imports
from mb_ge.selection.selection import SelectionMethod

class RandomSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        selected_cell = random.choice(list(archive._archive.values()))
        return self.select_element_from_element_list(selected_cell._elements)
    
    def select_element_from_element_list(self, elements):
        return random.choice(elements)
