if __name__ == '__main__':
    from mb_ge.selection.random_selection import RandomSelection
    from mb_ge.selection.heuristic_selection import HeuristicSelection
    from mb_ge.selection.mean_disagreement_selection import MeanDisagreementSelection
    from mb_ge.selection.max_disagreement_selection import MaxDisagreementSelection
    from mb_ge.selection.state_disagreement_selection import StateDisagreementSelection
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
    # parser.add_argument('--epoch-mode', type=str, default='model_update')
    parser.add_argument('--budget', type=int, default=100000)
    parser.add_argument('--dump-path', type=str)
    parser.add_argument('--dump-rate', type=int)
    # parser.add_argument('--variable-horizon', action='store_true')
    parser.add_argument('--variable-horizon', type=str)#, default='model_update')

    args = parser.parse_args()

    cell_selection_method = RandomSelection
    if args.cell_selection is not None:
        if args.cell_selection == 'random':
            cell_selection_method = RandomSelection
        if args.cell_selection == 'heuristic':
            cell_selection_method = HeuristicSelection
        if args.cell_selection == 'statedisagr':
            cell_selection_method = StateDisagreementSelection

    transfer_selection_method = RandomSelection
    if args.transfer_selection is not None:
        if args.transfer_selection == 'random':
            transfer_selection_method = RandomSelection
        if args.transfer_selection == 'meandisagr':
            transfer_selection_method = MeanDisagreementSelection
        if args.transfer_selection == 'maxdisagr':
            transfer_selection_method = MaxDisagreementSelection
        if args.transfer_selection == 'statedisagr':
            transfer_selection_method = StateDisagreementSelection

    exploration_method = RandomExploration
    if args.exploration is not None:
        if args.exploration == 'random':
            exploration_method = RandomExploration
        if args.exploration == 'ns':
            exploration_method = NoveltySearchExploration

    epoch_mode = "model_update"
    # if args.epoch_mode is not None:
    #     if args.epoch_mode == 'model_update':
    #         epoch_mode = 'model_update'
    #     if args.epoch_mode == 'fixed_steps':
    #         epoch_mode = 'fixed_steps'
    #     if args.epoch_mode == 'unique_fixed_steps':
    #         epoch_mode = 'unique_fixed_steps'
    if args.variable_horizon is not None:
        use_variable_horizon = True
        if args.variable_horizon == 'model_update':
            epoch_mode = 'model_update'
        if args.variable_horizon == 'fixed_steps':
            epoch_mode = 'fixed_steps'
        if args.variable_horizon == 'unique_fixed_steps':
            epoch_mode = 'unique_fixed_steps'

    budget = 100000
    if args.budget is not None:
        budget = args.budget

    dump_rate = 50
    if args.dump_rate is not None:
        dump_rate = args.dump_rate

    use_variable_horizon = args.variable_horizon
    # if args.variable_horizon is not None:
        # use_variable_horizon = True

    ## Framework methods
    env = gym.make('BallInCup3d-v0')
        
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
    }
    params = \
    {
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'action_min': -1,
        'action_max': 1,
        
        'budget': budget,
        'exploration_horizon': 100,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,

        'archive_type': 'cell',
        'fixed_grid_min': -0.5,
        'fixed_grid_max': 0.5,
        'fixed_grid_div': 10,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'model_update_rate': 10,
        'dynamics_model_params': dynamics_model_params,

        'epoch_mode': epoch_mode,
        'steps_per_epoch': 1000,
        'use_variable_model_horizon': use_variable_horizon,
        'min_horizon': 1,
        'max_horizon': 25,
        'horizon_starting_epoch': 20,
        'horizon_ending_epoch': 100,
        
        'dump_rate': dump_rate,
        'nb_of_samples_per_state':10,
        'dump_all_transitions': False,
        'env_max_h': env.max_steps,
    }
    
    if args.dump_path is not None:
        params['dump_path'] = args.dump_path    

    go_method = ExecutePolicyGo

    state_archive_type = FixedGridArchive

    dynamics_model = DynamicsModel

    if args.algorithm is not None:
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
