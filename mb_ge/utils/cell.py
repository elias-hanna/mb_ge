class Cell():
    def __init__(self):
        self._elements = list()

    def add(self, element):
        self._elements.append(element)

    def remove(self, element):
        self._elements.remove(element)

    def _shortest_traj(self):
        best_el = None
        best_traj_length = 1000000
        for el in self._elements:
            traj_length = len(el.trajectory)
            prev_el = el.previous_element
            while prev_el != None:
                traj_length += len(prev_el.trajectory)
                prev_el = prev_el.previous_element
            if traj_length < best_traj_length:
                best_traj_length = traj_length
                best_el = el
        return best_el, best_traj_length

    def get_elements(self):
        return self._elements
    
    def get_best_policy_to_cell(self, mode='shortest_traj'):
        if mode == 'shortest_traj':
            return self._shortest_traj()
