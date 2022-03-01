import random
import math

## Local imports
from mb_ge.selection.selection import SelectionMethod

class HeuristicSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        selected_cell = random.choices(archive._archive.values(),
                                       weights=[1/(math.sqrt(0.5*cell.visit_count + 1))
                                                for cell in archive._archive.values()])        
        return self.select_element_from_element_list(selected_cell._elements)
    
    def select_element_from_element_list(self, elements):
        return random.choice(elements)
