from abc import abstractmethod

class GoMethod():
    def __init__(self, params=None):
        self._process_params(params)

    def _process_params(self, params):
        if 'controller_type' in params:
            ## Associate an instance of controller_type with given params
            self.controller = params['controller_type'](params=params)
        else:
            raise Exception('ExplorationMethod _process_params error: controller_type not in params')

    def go(self, gym_env, el):
        """
        Returns to cell-state el on gym_env
        Args:
            gym_env: Gym environment on which state needs to be restored to el
            el: Cell-State(element) the gym environment must be restored to
        Returns:
            transitions: observed transitions while going back to the state
            budget_used: budget spent using the go method
        """
        raise NotImplementedError
    
