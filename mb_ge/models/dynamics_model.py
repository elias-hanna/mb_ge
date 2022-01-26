# Model Dependancies
from mb_ge.models.deterministic_model import DeterministicDynModel
from mb_ge.models.probabilistic_ensemble import ProbabilisticEnsemble
from mb_ge.utils.simple_replay_buffer import SimpleReplayBuffer

# torch import
import torch

# Other includes
import numpy as np
import copy

class DynamicsModel():
    def __init__(self, params=None):
        self._process_params(params)

        ## INIT MODEL ##
        if self._dynamics_model_type == "prob":
            from src.trainers.mbrl.mbrl import MBRLTrainer
            variant = dict(
                mbrl_kwargs=dict(
                    ensemble_size=self._ensemble_size,
                    layer_size=self._layer_size,
                    learning_rate=self._learning_rate,
                    batch_size=self._batch_size,
                )
            )
            M = variant['mbrl_kwargs']['layer_size']
            dynamics_model = ProbabilisticEnsemble(
                ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
                obs_dim=self._obs_dim,
                action_dim=self._action_dim,
                hidden_sizes=[M, M])
            dynamics_model_trainer = MBRLTrainer(ensemble=dynamics_model,
                                                 **variant['mbrl_kwargs'],)

        # ensemble somehow cant run in parallel evaluations
        elif dynamics_model_type == "det":
            from src.trainers.mbrl.mbrl_det import MBRLTrainer
            dynamics_model = DeterministicDynModel(obs_dim=self._obs_dim,
                                                   action_dim=self._action_dim,
                                                   hidden_size=self._layer_size)
            dynamics_model_trainer = MBRLTrainer(model=dynamics_model,
                                                 batch_size=self._batch_size,)
        self._dynamics_model = dynamics_model
        self._dynamics_model_trainer = dynamics_model_trainer

        # initialize replay buffer
        self._replay_buffer = SimpleReplayBuffer(max_replay_buffer_size=1000000,
                                                 observation_dim=self._batch_size,
                                                 action_dim=self._action_dim,
                                                 env_info_sizes=dict(),)

    def _process_params(self, params):
        if 'dynamics_model_params' in params:
            if 'obs_dim' in params['dynamics_model_params']:
                self._obs_dim = params['dynamics_model_params']['obs_dim']
            else:
                raise Exception('ExplorationMethod _process_params error: obs_dim not in params')
            if 'action_dim' in params['dynamics_model_params']:
                self._action_dim = params['dynamics_model_params']['action_dim']
            else:
                raise Exception('ExplorationMethod _process_params error: action_dim not in params')
            if 'dynamics_model_type' in params['dynamics_model_params']:
                self._dynamics_model_type = params['dynamics_model_params']['dynamics_model_type']
            else:
                raise Exception('ExplorationMethod _process_params error: dynamics_model_type not in params')
            if 'ensemble_size' in params['dynamics_model_params']:
                self._ensemble_size = params['dynamics_model_params']['ensemble_size']
            else:
                raise Exception('ExplorationMethod _process_params error: ensemble_size not in params')
            if 'layer_size' in params['dynamics_model_params']:
                self._layer_size = params['dynamics_model_params']['layer_size']
            else:
                raise Exception('ExplorationMethod _process_params error: layer_size not in params')
            if 'batch_size' in params['dynamics_model_params']:
                self._batch_size = params['dynamics_model_params']['batch_size']
            else:
                raise Exception('ExplorationMethod _process_params error: batch_size not in params')
            if 'learning_rate' in params['dynamics_model_params']:
                self._learning_rate = params['dynamics_model_params']['learning_rate']
            else:
                raise Exception('ExplorationMethod _process_params error: learning_rate not in params')
        else:
            raise Exception('ExplorationMethod _process_params error: dynamics_model_params not in params')

    def forward(self, a, s, mean=True, disagr=True):
        s_0 = copy.deepcopy(s)
        a_0 = copy.deepcopy(a)
        
        s_0 = np.tile(s_0,(dynamics_model.ensemble_size, 1))
        
        s_0 = ptu.from_numpy(s_0)
        a_0 = ptu.from_numpy(a_0)
        
        a_0 = a_0.repeat(dynamics_model.ensemble_size,1)
        # if probalistic dynamics model - choose output mean or sample
        if disagr:
            pred_delta_ns, _ = dynamics_model.sample_with_disagreement(
                torch.cat((
                    dynamics_model._expand_to_ts_form(s_0),
                    dynamics_model._expand_to_ts_form(a_0)), dim=-1
                ), disagreement_type="mean" if mean else "var")
            pred_delta_ns = ptu.get_numpy(pred_delta_ns)
        
        else:
            pred_delta_ns = dynamics_model.output_pred_ts_ensemble(s_0, a_0, mean=mean)
        return pred_delta_ns

    def train(self, replay_buffer, verbose=True):
        torch.set_num_threads(24)
        self._dynamics_model_trainer.train_from_buffer(replay_buffer,
                                                       holdout_pct=0.1,
                                                       max_grad_steps=200000)
        if verbose:
            print("=========================================\nDynamics Model Trainer statistics:")
            stats = dynamics_model_trainer.get_diagnostics()
            for name, value in zip(stats.keys(), stats.values()):
                print(name, ": ", value)
            print("=========================================\n")
                
    def add_samples(self, S, A, NS):
        for s, a, ns in zip(S, A):
            self._replay_buffer.add_sample(s, a, 0, 0, ns, {})
            
    # Utils methods
    def normalize_standard(self, vector, mean_vector, std_vector):
        return [(vector[i] - mean_vector[i])/std_vector[i] for i in range(len(vector))]
    
    def rescale_standard(self, vector, mean_vector, std_vector):
        return [vector[i]*std_vector[i] + mean_vector[i] for i in range(len(vector))]
