from abc import abstractmethod

class SelectionMethod():
    def __init__(self, params=None):
        self._process_params(params)

    def _process_params(self, params):
        if 'env_max_h' in params:
            self._env_max_h = params['env_max_h']
        else:
            raise Exception('SelectionMethod _process_params error: env_max_h not in params')

    def _horizon_check(self, el):
        traj_len = len(el.trajectory)
        prev_el = el.previous_element
        while prev_el != None:
            traj_len += len(prev_el.trajectory)
            prev_el = prev_el.previous_element
        return (traj_len < self._env_max_h)

    def _get_horizon_checked_element_list(self, elements):
        checked_elements = []
        for element in elements:
            if self._horizon_check(element):
                checked_elements.append(element)
        return checked_elements
    
    @abstractmethod
    def select_element_from_cell_archive(self, archive):
        """
        Usually used to select which state to explore from.
        Args: 
            archive: Archive to draw an cell (key) from. Must be of class Archive
        
        Returns: 
            element: Element that has been drawn from archive
        """
        raise NotImplementedError

    @abstractmethod
    def select_element_from_element_list(self, elements, k=1):
        """
        Usually used to select which policies to transfer.
        Args: 
            elements: Archive to draw an Element (value) from. Must be of class Archive
        
        Returns: 
            element: Element that has been drawn from archive
        """
        raise NotImplementedError
