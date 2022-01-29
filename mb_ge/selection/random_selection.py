import random

## Local imports
from mb_ge.selection.selection import SelectionMethod

class RandomSelection(SelectionMethod):
    def select_element_from_cell_archive(self, archive):
        selected_cell = random.choice(list(archive._archive.values()))
        return self.select_element_from_element_list(selected_cell._elements)
    
    def select_element_from_element_list(self, elements):
        return random.choice(elements)
    
if __name__ == '__main__':
    from mb_ge.utils.element import Element
    from mb_ge.archive.fixed_grid_archive import FixedGridArchive
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

    selection = RandomSelection()
    print(selection(archive))

    print(selection.select_element(archive))
