if __name__ == '__main__':
    from mb_ge.selection.random_selection import RandomSelection
    from mb_ge.selection.mean_disagreement_selection import MeanDisagreementSelection
    from mb_ge.selection.max_disagreement_selection import MaxDisagreementSelection
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
    parser.add_argument('--selection', type=str, default='random')
    parser.add_argument('--exploration', type=str, default='random')

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
        
        'budget': 100000,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,

        'archive_type': 'cell',
        'fixed_grid_min': -0.5,
        'fixed_grid_max': 0.5,
        'fixed_grid_div': 5,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'model_update_rate': 10,
        'dynamics_model_params': dynamics_model_params,

        'dump_rate': 50,
    }
    
    args = parser.parse_args()

    selection_method = RandomSelection
    if args.selection is not None:
        if args.selection == 'random':
            selection_method = RandomSelection
        if args.selection == 'meandisagr':
            selection_method = MeanDisagreementSelection
        if args.selection == 'maxdisagr':
            selection_method = MaxDisagreementSelection

    exploration_method = RandomExploration
    if args.exploration is not None:
        if args.exploration == 'random':
            exploration_method = RandomExploration
        if args.exploration == 'ns':
            exploration_method = NoveltySearchExploration

    ## Framework methods
    env = gym.make('BallInCup3d-v0')

    go_method = ExecutePolicyGo

    state_archive_type = FixedGridArchive

    dynamics_model = DynamicsModel

    if args.algorithm is not None:
        if args.algorithm == 'ge':
            ge = GoExplore(params=params, gym_env=env, selection_method=selection_method,
                           go_method=go_method, exploration_method=exploration_method,
                           state_archive=state_archive_type)

        if args.algorithm == 'mb_ge':
            ge = ModelBasedGoExplore(params=params, gym_env=env, selection_method=selection_method,
                                     go_method=go_method, exploration_method=exploration_method,
                                     state_archive=state_archive_type,
                                     dynamics_model=dynamics_model)
    
    ge._exploration_phase()

    ge.state_archive.visualize(params['budget'], show=True)

