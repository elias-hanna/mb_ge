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
    def __init__(self, dynamics_model_type, obs_dim, action_dim):
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        ## INIT MODEL ##
        if dynamics_model_type == "prob":
            from src.trainers.mbrl.mbrl import MBRLTrainer
            variant = dict(
                mbrl_kwargs=dict(
                    ensemble_size=4,
                    layer_size=500,
                    learning_rate=1e-3,
                    batch_size=512,
                )
            )
            M = variant['mbrl_kwargs']['layer_size']
            dynamics_model = ProbabilisticEnsemble(
                ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=[M, M]
            )
            dynamics_model_trainer = MBRLTrainer(
                ensemble=dynamics_model,
                **variant['mbrl_kwargs'],
            )

        # ensemble somehow cant run in parallel evaluations
        elif dynamics_model_type == "det":
            from src.trainers.mbrl.mbrl_det import MBRLTrainer
            dynamics_model = DeterministicDynModel(obs_dim=obs_dim,
                                                   action_dim=action_dim,
                                                   hidden_size=500)
            dynamics_model_trainer = MBRLTrainer(
                model=dynamics_model,
                batch_size=512,)
        self._dynamics_model = dynamics_model
        self._dynamics_model_trainer = dynamics_model_trainer

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

    # Utils methods
    def normalize_standard(self, vector, mean_vector, std_vector):
        return [(vector[i] - mean_vector[i])/std_vector[i] for i in range(len(vector))]
    
    def rescale_standard(self, vector, mean_vector, std_vector):
        return [vector[i]*std_vector[i] + mean_vector[i] for i in range(len(vector))]
