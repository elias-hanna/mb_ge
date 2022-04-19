# Data manipulation includes
import numpy as np
import matplotlib
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# OS manipulation includes
import sys
import os

# Local imports
from count_bins import getBinsReachable

# MB-GE includes
from mb_ge.controller.nn_controller import NeuralNetworkController

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} path_of_folder_with_res_repetitions')
        exit()

    folderpath = sys.argv[1]

    splitted_foldername = list(filter(None, folderpath.split('/')))
    run_name = splitted_foldername[-2] + '_' +splitted_foldername[-1]
    
    controller_params = \
    {
        'controller_input_dim': 6,
        'controller_output_dim': 3,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 50
    }
    dynamics_model_params = \
    {
        'obs_dim': 6,
        'action_dim': 3,
        'dynamics_model_type': 'prob', # possible values: prob, det
        'ensemble_size': 4, # only used if dynamics_model_type == prob
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
        'train_unique_trans': False,
    }
    params = \
    {
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'action_min': -1,
        'action_max': 1,
        
        'budget': 1000000,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,

        'archive_type': 'cell',
        'single_element_per_cell': True,
        'fixed_grid_min': -0.5,
        'fixed_grid_max': 0.5,
        'fixed_grid_div': 30,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'dynamics_model_params': dynamics_model_params,
        
        'epoch_mode': 'dummy_val',
        'model_update_rate': 10,
        'steps_per_epoch': 5000, # unused if epoch_mode == model_update
        'use_variable_model_horizon': 'dummy_val',
        'min_horizon': 1, # unused if use_variable_horizon == False
        'max_horizon': 25, # unused if use_variable_horizon == False
        'horizon_starting_epoch': 20, # unused if use_variable_horizon == False
        'horizon_ending_epoch': 100, # unused if use_variable_horizon == False

        'dump_path': 'dummy_val',
        'dump_rate': 'dummy_val', # unused if dump_checkpoints used
        'dump_checkpoints': [10000, 20000, 50000, 100000, 200000, 500000, 1000000],
        'nb_of_samples_per_state':10,
        'dump_all_transitions': False,
        'env_max_h': 'dummy_val',
    }

    reachable_bins = getBinsReachable(params['fixed_grid_min'], params['fixed_grid_max'],
                                      params['fixed_grid_div'])

    rep_dirs = next(os.walk(folderpath))[1]

    number_of_reps = len(rep_dirs)

    coverage_vals = []
    budgets = []
    for rep_dir in rep_dirs:
        print(rep_dir)
        rep_path = os.path.join(folderpath, rep_dir)
        coverage_data = np.load(os.path.join(rep_path, 'coverage_data.npz'))
        coverage_vals.append(coverage_data['cells']/reachable_bins)
        budgets.append(coverage_data['budget'])

    color = cm.rainbow(np.linspace(0, 1, len(coverage_vals)))
   
    plt.figure()

    for cov, bud, c in zip(coverage_vals, budgets, color):
        plt.plot(bud, cov, c=c)

    plt.title(f"Coverage depending on number of iterations")
    # plt.savefig(f"coverage_{run_name}.jpg")

    plt.show()
