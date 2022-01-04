from abc import abstractmethod

## Archive manipulation imports
from sklearn.neighbors import KDTree

class Archive():
    def __init__(self, params=None):
        ## kdt will store either the centroids of the behaviour archive
        self._centroids_kdt = KDTree(c, leaf_size=40, metric='euclidean')
        self._behaviour_archive_kdt = None

        pass

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
    def remove(self, element):
        """
        Removes an element from the archive

        Args:
            element: element to be removed from the archive
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
            result: 1 if element1 > element2, -1 if element2 > element1, 0 if element1 == element2,
                    given the archive rules.
        """
        raise NotImplementedError
