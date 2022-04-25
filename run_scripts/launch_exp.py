if __name__ == '__main__':
    # Local imports
    from mb_ge.selection.random_selection import RandomSelection
    from mb_ge.selection.heuristic_selection import HeuristicSelection
    from mb_ge.selection.mean_disagreement_selection import MeanDisagreementSelection
    from mb_ge.selection.max_disagreement_selection import MaxDisagreementSelection
    from mb_ge.selection.state_disagreement_selection import StateDisagreementSelection
    from mb_ge.selection.novelty_selection import NoveltySelection
    from mb_ge.selection.pareto_front_selection import ParetoFrontSelection

    from mb_ge.go.execute_policy_go import ExecutePolicyGo

    from mb_ge.exploration.random_exploration import RandomExploration
    from mb_ge.exploration.ns_exploration import NoveltySearchExploration

    from mb_ge.archive.fixed_grid_archive import FixedGridArchive

    from mb_ge.models.dynamics_model import DynamicsModel

    from mb_ge.controller.nn_controller import NeuralNetworkController

    from mb_ge.ge.ge import GoExplore
    from mb_ge.ge.ge_mb import ModelBasedGoExplore
    
    import gym

    import argparse
    
    parser = argparse.ArgumentParser(description='Process run parameters.')

    parser.add_argument('--algorithm', type=str, default='ge')

    parser.add_argument('--cell-selection', type=str, default='random')
    parser.add_argument('--transfer-selection', type=str, default='random')
    parser.add_argument('--exploration', type=str, default='random')

    parser.add_argument('--budget', type=int, default=100000)

    parser.add_argument('--dump-path', type=str, default='default_dump/')
    parser.add_argument('--dump-rate', type=int, default=100)

    parser.add_argument('--train-unique-trans', action='store_true')

    parser.add_argument('--variable-horizon', action='store_true')
    parser.add_argument('--epoch-mode', type=str, default='fixed_steps')

    args = parser.parse_args()

    if args.cell_selection == 'random':
        cell_selection_method = RandomSelection
    elif args.cell_selection == 'heuristic':
        cell_selection_method = HeuristicSelection
    elif args.cell_selection == 'statedisagr':
        cell_selection_method = StateDisagreementSelection
    elif args.cell_selection == 'novelty':
        cell_selection_method = NoveltySelection
    elif args.cell_selection == 'paretofront':
        cell_selection_method = ParetoFrontSelection

    if args.transfer_selection == 'random':
        transfer_selection_method = RandomSelection
    elif args.transfer_selection == 'meandisagr':
        transfer_selection_method = MeanDisagreementSelection
    elif args.transfer_selection == 'maxdisagr':
        transfer_selection_method = MaxDisagreementSelection
    elif args.transfer_selection == 'statedisagr':
        transfer_selection_method = StateDisagreementSelection
    elif args.transfer_selection == 'novelty':
        transfer_selection_method = NoveltySelection
    elif args.transfer_selection == 'paretofront':
        transfer_selection_method = ParetoFrontSelection

    if args.exploration == 'random':
        exploration_method = RandomExploration
    elif args.exploration == 'ns':
        exploration_method = NoveltySearchExploration

    go_method = ExecutePolicyGo

    epoch_mode = args.epoch_mode
    train_unique_trans = args.train_unique_trans
    use_variable_horizon = args.variable_horizon
    dump_rate = args.dump_rate
    budget = args.budget

    # if args.dump_path is not None:
    #     params['dump_path'] = args.dump_path

    state_archive_type = FixedGridArchive

    dynamics_model = DynamicsModel
    
    ## Framework methods
    env = gym.make('BallInCup3d-v0')
        
    controller_params = \
    {
        'controller_input_dim': 6,
        'controller_output_dim': 3,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 10
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
        'train_unique_trans': train_unique_trans,
    }
    params = \
    {
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'action_min': -1,
        'action_max': 1,
        
        'budget': budget,
        'exploration_horizon': 50, # usually 10
        'nb_eval_exploration': 20, # usually 10
        'nb_thread_exploration': 6, # usually 6

        'archive_type': 'cell',
        'single_element_per_cell': True,
        'fixed_grid_min': -0.4,
        'fixed_grid_max': 0.4,
        'fixed_grid_div': 100, # usually 30
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'dynamics_model_params': dynamics_model_params,
        
        'epoch_mode': epoch_mode,
        'model_update_rate': 10,
        'steps_per_epoch': 1000, # unused if epoch_mode == model_update
        'use_variable_model_horizon': use_variable_horizon,
        'min_horizon': 1, # 1 in mbpo # unused if use_variable_horizon == False
        'max_horizon': 25, # 25 in mbpo unused if use_variable_horizon == False
        'horizon_starting_epoch': 1, # 20 in mbpo # unused if use_variable_horizon == False
        'horizon_ending_epoch': 10, # 100 in mbpo # unused if use_variable_horizon == False

        'dump_path': args.dump_path,
        'dump_rate': dump_rate, # unused if dump_checkpoints used
        # 'dump_checkpoints': [10000, 20000, 50000, 100000, 200000, 500000, 1000000],
        'dump_checkpoints': [1000, 2000, 5000, 10000, 20000, 50000, 100000],# 1000000],
        'nb_of_samples_per_state': 10,
        'dump_all_transitions': False,
        'env_max_h': env.max_steps,

        'path_to_test_trajectories':'example_trajectories.npz',
    }
    

    if args.algorithm == 'ge':
        ge = GoExplore(params=params, gym_env=env, cell_selection_method=cell_selection_method,
                       go_method=go_method, exploration_method=exploration_method,
                       state_archive=state_archive_type)

    if args.algorithm == 'mb_ge':
        ge = ModelBasedGoExplore(params=params, gym_env=env,
                                 cell_selection_method=cell_selection_method,
                                 transfer_selection_method=transfer_selection_method,
                                 go_method=go_method, exploration_method=exploration_method,
                                 state_archive=state_archive_type,
                                 dynamics_model=dynamics_model)
    
    ge._exploration_phase()
