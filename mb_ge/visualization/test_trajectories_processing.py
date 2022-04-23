import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from mb_ge.visualization.plot_utils import PlotUtils

class TestTrajectoriesProcessing():
    def __init__(self, params=None):
        if 'path_to_test_trajectories' in params:
            self.test_trajectories = np.load(params['path_to_test_trajectories'])
            ## /!\ Warning, test_trajectories must be of shape (nb_of_trajs, nb_of_steps, state_dim)
            ## Do some preprocessing here to put the trajs in the form of a [[(Ai, Si, NSi), ...,]]
        else:
            raise Exception('TestTrajectoriesProcessing _process_params error: path_to_test_trajectories not in params')
        if 'model' in params:
            self.model = params['model']
        else:
            raise Exception('TestTrajectoriesProcessing _process_params error: model not in params')
        if 'env_max_h' in params:
            self.env_max_h = params['env_max_h']
        else:
            raise Exception('TestTrajectoriesProcessing _process_params error: env_max_h not in params')
        if 'controller_type' in params:
            ## Associate an instance of controller_type with given params
            self.controller = params['controller_type'](params=params)
        else:
            raise Exception('ExplorationMethod _process_params error: controller_type not in params')
        if 'dynamics_model_params' in params:
            dynamics_model_params = params['dynamics_model_params']
            if 'obs_dim' in dynamics_model_params:
                self._obs_dim = dynamics_model_params['obs_dim']
            else:
                raise Exception('TestTrajectoriesProcessing _process_params error: obs_dim not in params')
            if 'action_dim' in dynamics_model_params:
                self._action_dim = dynamics_model_params['action_dim']
            else:
                raise Exception('TestTrajectoriesProcessing _process_params error: action_dim not in params')
        else:
            raise Exception('TestTrajectoriesProcessing _process_params error: dynamics_model_params not in params')
        
    def _execute_test_trajectories_on_model():
        controller_list = []

        traj_list = []
        disagreements_list = []
        prediction_errors_list = []

        pred_trajs = np.empty((len(self.test_trajectories), self.env_max_h, self.obs_dim))
        disagrs = np.empty((len(self.test_trajectories), self.env_max_h))
        pred_errors = np.empty((len(self.test_trajectories), self.env_max_h))
        
        A = np.empty((len(self.test_trajectories), self.action_dim))
        S = np.empty((len(self.test_trajectories), self.obs_dim))
        for _ in len(self.test_trajectories):
            ## Create a copy of the controller
            controller_list.append(self.controller.copy())
            ## Set controller parameters
            controller_list[-1].set_parameters(x)
            ## Init starting state
            S[i,:] = test_trajectories[i,0,:]
        
        for i in range(self.env_max_h):
            for j in range(len(test_trajectories)):
                A[j,:] = controller_list[S[i,:])
                
            batch_pred_delta_ns, batch_disagreement = model.forward_multiple(A, S, mean=True,
                                                                             disagr=True)
            for j in range(len(self.test_trajectories)):
                ## Compute mean prediction from model samples
                next_step_pred = batch_pred_delta_ns[i]
                mean_pred = [np.mean(next_step_pred[:,i]) for i in range(len(next_step_pred[0]))]
                S[i,:] += mean_pred.copy()
                pred_trajs[j,i,:] = mean_pred.copy()
                disagrs[j,i] = np.mean(batch_disagreement[i].detach().numpy())
                pred_errors[j,i] = np.linalg.norm(S[i,:]-self.test_trajectories[j,i,:])

        return pred_trajs, disagrs, pred_errors
        
    def dump_plots(self, curr_budget, itr=0, show=False):
        ## Get results of test trajectories on model on last model update
        pred_trajs, disagrs, pred_errors = self._execute_test_trajectories_on_model()

        ## Compute mean and stddev of trajs disagreement
        mean_disagr = np.mean(disagrs, axis=0)
        std_disagr = np.std(disagrs, axis=0)
        ## Compute mean and stddev of trajs prediction error
        mean_pred_error = np.mean(pred_errors, axis=0)
        std_pred_error = np.std(pred_errors, axis=0)

        ## Create fig and ax
        fig = plt.figure(figsize=(8, 8), dpi=160)
        ax = fig.add_subplot(111, projection='3d')
        ## Prepare plot 
        PlotUtils.prepare_plot(plt, fig, ax)

        ## Figure for model ensemble disagreement
        plt.figure()
        plt.plot(label, mean_disagr, 'k-')
        plt.fill_between(label, mean_disagr-std_disagr, mean_disagr+std_disagr,
                         facecolor='green', alpha=0.5)
        ## Set plot title
        plt.title(f"Mean model ensemble disagreeement along successful test trajectories")
        ## Save fig
        plt.savefig(f"{self.dump_path}/results_{itr}/test_trajectories_disagr",
                    bbox_inches='tight')
        
        ## Figure for prediction error
        plt.figure()
        plt.plot(label, mean_pred_error, 'k-')
        plt.fill_between(label, mean_pred_error-std_pred_error, mean_pred_error+std_pred_error,
                         facecolor='green', alpha=0.5)
        ## Set plot title
        plt.title(f"Mean prediction error along successful test trajectories")
        ## Save fig
        plt.savefig(f"{self.dump_path}/results_{itr}/test_trajectories_pred_error",
                    bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
