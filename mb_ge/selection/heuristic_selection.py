import random
import math

## Local imports
from mb_ge.selection.selection import SelectionMethod

class HeuristicSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        weights = [1/(math.sqrt(0.5*cell.visit_count + 1)) for cell in archive._archive.values()]
        selected_cell = random.choices([cell for cell in archive._archive.values()],
                                       weights=weights)        
        return self.select_element_from_element_list(selected_cell[0]._elements)
    
    def select_element_from_element_list(self, elements):
        return random.choice(elements)
