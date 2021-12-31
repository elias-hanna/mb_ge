## Controller import
from diversity_algorithms.controllers.fixed_structure_nn_numpy import SimpleNeuralControllerNumpy as SimpleNeuralController

## Local imports
from mb_ge.controller.controller import Controller

class NeuralNetworkController(Controller):
    def __init__(self, params=None):
        super().__init__(params)
        ## Exception error raised in super __init__ if controller_params not in params uwu
        self.controller = SimpleNeuralController(self.input_dim, self.output_dim,
                                                 params=params['controller_params'])

        self.n_params = self.controller.n_weights
        self.params = self.controller.get_parameters()

    # def _init_controller(self, params):
        
    def compute_action(self, obs):
        """
        Args:
            obs: what the controller takes as input to compute actions, must be of dimensions equal to self.input_dim
        
        Returns:
            action: action vector, must be of dimensions equal to self.output_dim
        """
        return self.controller(obs)

    def set_parameters(self, parameters):
        """
        Sets the controller parameters
        Args:
            parameters: 1 dimensionnal parameters vector
        """
        self.controller.set_parameters(parameters)

    def get_parameters(self):
        """
        Gets the controller parameters
        Returns:
            parameters: 1 dimensionnal parameters vector
        """
        return self.controller.get_parameters()
        
    def __call__(self, obs):
        return self.compute_action(obs)
