# Data manipulation includes
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Local imports
from count_bins import getBinsReachable

# MB-GE includes
from mb_ge.controller.nn_controller import NeuralNetworkController


if __name__ == '__main__':
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

    # label = [int(val) for val in max_values]

    label = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]

    coverage_data = np.load('coverage_data.npz')
    budget_spent = coverage_data['budget']
    cells = coverage_data['cells']
    import pdb; pdb.set_trace()
    coverage = cells/reachable_bins
    
    ## Compute coverage and reward mean/error
    # coverage_mean = np.nanmean(coverage_vals, axis = 0)
    # reward_mean = np.nanmean(rewarding_pis_vals, axis = 0)

    # coverage_error = np.nanstd(coverage_vals, axis = 0)
    # reward_error = np.nanstd(rewarding_pis_vals, axis = 0)

    plt.figure()

    plt.plot(budget_spent, coverage, 'k-')
    # plt.plot(label, coverage_mean, 'k-')
    # plt.fill_between(label, coverage_mean-coverage_error, coverage_mean+coverage_error,
                     # facecolor='green', alpha=0.5)
    plt.title(f"Coverage depending on number of iterations")
    # plt.savefig(f"coverage_{run_name}.jpg")

    plt.show()
