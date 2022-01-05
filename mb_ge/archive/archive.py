from abc import abstractmethod

## Archive manipulation imports
from sklearn.neighbors import KDTree

class Element():
    def __init__(self, descriptor=None, trajectory=None, reward=None, policy_parameters=None):
        self.descriptor = descriptor
        self.trajectory = trajectory
        self.reward = reward
        self.policy_parameters = policy_parameters
        self.previous_element = None ## allows to chain policies

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
        
    def get_best_policy_to_cell(self, mode='shortest_traj'):
        if mode == 'shortest_traj':
            return self._shortest_traj()

class Archive():
    def __init__(self, params=None):
        ## kdt will store either the centroids of the behaviour archive
        self._centroids_kdt = None
        self._behaviour_archive_kdt = None
        ## Archive is a dict
        self._archive = dict()

    
    def _process_params(self, params):
        pass
    
    @abstractmethod
    def add(self, element):
        """
        Adds an element to the archive, given the archive adding rules.

        Args:
            element: element to be added to the archive
        """
        raise NotImplementedError

    @abstractmethod
    def compare(self, element1, element2):
        """
        Compare two elements given the archive rules.

        Args:
            element1: element to be compared to element2
            element2: element to be compared to element1

        Returns:
            result: 1 if element1 > element2, -1 if element2 > element1
        """
        raise NotImplementedError
