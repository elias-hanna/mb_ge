from archive import Archive

from math import floor

class FixedGridArchive(Archive):
    def __init__(self, params=None):
        super().__init__(params)
        self._process_params(params)
        
    def _process_params(self, params):
        if 'fixed_grid_min' in params:
            self._grid_min = params['fixed_grid_min']
        else:
            raise Exception('FixedGridArchive _process_params error: fixed_grid_min not in params')
        if 'fixed_grid_max' in params:
            self._grid_max = params['fixed_grid_max']
        else:
            raise Exception('FixedGridArchive _process_params error: fixed_grid_max not in params')
        if 'fixed_grid_div' in params:
            self._grid_div = params['fixed_grid_div']
        else:
            raise Exception('FixedGridArchive _process_params error: fixed_grid_div not in params')
        

    def add(self, element):
        ## WARNING: element.descriptor must be a single array
        ## Here descriptor values are normalized in [0,1] and scaled to the number of cells
        ## each dimension of the archive is divided in
        # import pdb; pdb.set_trace()
        archive_index_str = ''.join([str(i)+str(floor((element.descriptor[i]-self._grid_min
                                                       /(self._grid_max-self._grid_min))
                                               *self._grid_div))
                                     for i in range(len(element.descriptor))])

        
        
        if archive_index_str in self._archive: ## Case cell already contains an element
            if self.compare(element, self._archive[archive_index_str]):
                self._archive[archive_index_str] = element
            
        else:
            self._archive[archive_index_str] = element

    def compare(self, element1, element2):
        if element1.reward > element2.reward or len(element1.trajectory)<len(element2.trajectory):
            return 1
        return 0

if __name__ == '__main__':
    from archive import Element
    from mb_ge.controller.nn_controller import NeuralNetworkController
    import numpy as np
    controller_params = \
    {
        'controller_input_dim': 0,
        'controller_output_dim': 0,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 50
    }
    params = \
    {
        'controller_type': NeuralNetworkController, ## WARNING THIS NEED TO BE A CONTROLLER CLASS
        'controller_params': controller_params,
        
        'budget': 1000,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,

        'fixed_grid_min': 1.3,
        'fixed_grid_max': 4.1,
        'fixed_grid_div': 5,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
    }

    archive = FixedGridArchive(params)
    elements = []
    for _ in range(params['nb_eval_exploration']):
        x = np.random.uniform(low=params['policy_param_init_min'],
                              high=params['policy_param_init_max'],
                              size=10)
        rew = np.random.uniform()
        traj = [np.random.uniform(low=params['fixed_grid_min'],
                                  high=params['fixed_grid_max'],
                                  size=2)
                for _ in range(params['exploration_horizon'])]
        desc = traj[-1]
        el = Element(descriptor=desc, trajectory=traj, reward=rew, policy_parameters=x)
        archive.add(el)
        
    print(archive._archive)
