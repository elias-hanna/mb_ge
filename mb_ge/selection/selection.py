from abc import abstractmethod

class SelectionMethod():
    def __init__(self, params=None):
        self._process_params(params)

    def _process_params(self, params):
        pass
    
    @abstractmethod
    def _select_cell(self, archive):
        """
        Args: 
            archive: Archive to draw an cell (key) from. Must be of class Archive
        
        Returns: 
            element: Element that has been drawn from archive
        """
        raise NotImplementedError

    @abstractmethod
    def select_element(self, archive):
        """
        Args: 
            archive: Archive to draw an Element (value) from. Must be of class Archive
        
        Returns: 
            element: Element that has been drawn from archive
        """
        raise NotImplementedError
        
    def __call__(self, archive):
        return self._select_cell(archive)