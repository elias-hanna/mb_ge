## Local imports
from mb_ge.archive.archive import Archive
from mb_ge.utils.cell import Cell

## Calculus imports
from math import floor

class FixedGridArchive(Archive):
    def __init__(self, params=None):
        super().__init__(params)
        self._process_params(params)
        
    def _process_params(self, params):
        super()._process_params(params)
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
        if 'single_element_per_cell' in params:
            self._single_element_per_cell = params['single_element_per_cell']
        if 'archive_type' in params:
            self._archive_type = params['archive_type']
            if self._archive_type not in ['element', 'cell']:
                raise Exception(f'FixedGridArchive _process_params error: unknow archive_type \
                -> {self._archive_type}')
        else:
            raise Exception('FixedGridArchive _process_params error: archive_type not in params')

    def compare(self, element1, element2):
        if element1.reward > element2.reward or \
           element1.total_trajectory_len < element2.total_trajectory_len:
            return 1
        return 0
    
    def add(self, element):
        ## WARNING: element.descriptor must be a single array
        ## Here descriptor values are normalized in [0,1] and scaled to the number of cells
        ## each dimension of the archive is divided in
        a = [str(i)+str(floor((element.descriptor[i]-self._grid_min)
                              /(self._grid_max-self._grid_min)*self._grid_div))
             for i in range(len(element.descriptor))]
        archive_index_str = ''.join([str(i)+str(floor((element.descriptor[i]-self._grid_min)
                                                      /(self._grid_max-self._grid_min)
                                                      *self._grid_div))
                                     for i in range(len(element.descriptor))])
        
        if archive_index_str in self._archive: ## Case archive already exists at index
            if self._archive_type == 'cell':
                self._archive[archive_index_str].visit_count += 1
                if self._single_element_per_cell:
                    if self.compare(element, self._archive[archive_index_str]._elements[0]):
                        self._archive[archive_index_str]._elements = []
                        print('replacing elem')
                        self._archive[archive_index_str].add(element)
                else:
                    self._archive[archive_index_str].add(element)
            elif self._archive_type == 'element':
                if self.compare(element, self._archive[archive_index_str]):
                    self._archive[archive_index_str] = element
                    
        else:
            if self._archive_type == 'cell':
                cell = Cell()
                cell.add(element)
                self._archive[archive_index_str] = cell
            elif self._archive_type == 'element':
                self._archive[archive_index_str] = element

        
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

    archive = FixedGridElementArchive(params)
    elements = []
    for _ in range(params['nb_eval_exploration']):
        x = np.random.uniform(low=params['policy_param_init_min'],
                              high=params['policy_param_init_max'],
                              size=10)
        rew = np.random.uniform()
        traj = [np.random.uniform(low=params['fixed_grid_min'],
                                  high=params['fixed_grid_max'],
                                  size=3)
                for _ in range(params['exploration_horizon'])]
        desc = traj[-1]
        el = Element(descriptor=desc, trajectory=traj, reward=rew, policy_parameters=x)
        archive.add(el)
        
    print(archive._archive)
