import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from mb_ge.visualization.plot_utils import PlotUtils

class DiscretizedStateSpaceVisualization():
    def __init__(self, params=None):
        self._rope_length = .3
        self._disc_step = 0.02
        self._vel_min = 0.
        self._vel_max = 1.

        if 'fixed_grid_min' in params:
            self._grid_min = params['fixed_grid_min']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: fixed_grid_min not in params')
        if 'fixed_grid_max' in params:
            self._grid_max = params['fixed_grid_max']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: fixed_grid_max not in params')
        if 'model' in params:
            self.model = params['model']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: model not in params')
        if 'env_max_h' in params:
            self.env_max_h = params['env_max_h']
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: env_max_h not in params')
        if 'controller_type' in params:
            ## Associate an instance of controller_type with given params
            self.controller = params['controller_type'](params=params)
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: controller_type not in params')
        if 'dynamics_model_params' in params:
            dynamics_model_params = params['dynamics_model_params']
            if 'obs_dim' in dynamics_model_params:
                self._obs_dim = dynamics_model_params['obs_dim']
            else:
                raise Exception('DiscretizedStateSpaceVisualization _process_params error: obs_dim not in params')
            if 'action_dim' in dynamics_model_params:
                self._action_dim = dynamics_model_params['action_dim']
            else:
                raise Exception('DiscretizedStateSpaceVisualization _process_params error: action_dim not in params')
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: dynamics_model_params not in params')
        if 'nb_of_samples_per_state' in params:
            self._samples_per_state = params['nb_of_samples_per_state']
            self._action_sampled = np.random.uniform(low=-1, high=1,
                                                     size=(self._samples_per_state,
                                                           self._action_dim))
        else:
            raise Exception('DiscretizedStateSpaceVisualization _process_params error: fixed_grid_min not in params')

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

    def _reachable(self, centroid):
        return (np.linalg.norm(centroid[:3]) < 0.3)

    def _get_centroids(self):
        centroids = []
        for x in range(self._grid_min, self._grid_max, self._disc_step):
            for y in range(self._grid_min, self._grid_max, self._disc_step):
                for z in range(self._grid_min, self._grid_max, self._disc_step):
                    for vx in range(self._vel_min, self._vel_max, self._disc_step):
                        for vy in range(self._vel_min, self._vel_max, self._disc_step):
                            for vz in range(self._vel_min, self._vel_max, self._disc_step):
                                centroid = [x, y, z, vx, vy, vz]
                                if self._reachable(centroid):
                                    centroids.append(centroid)
        return centroids

    def _get_state_disagr(self, centroids):
        A = np.tile(self._actions_sampled, (len(elements), 1))

        all_s = []
        # Get all states to estimate uncertainty for
        for centroid in centroids:
            all_s.append(centroid)
        S = np.repeat(all_s, self.nb_of_samples_per_state, axis=0)
        # Batch prediction
        batch_pred_delta_ns, batch_disagreement = self._dynamics_model.forward_multiple(A, S,
                                                                                        mean=True,
                                                                                        disagr=True)
        centroids_disagrs = []
        for i in range(len(centroids)):
            disagrs = batch_disagreement[i*self._samples_per_state:
                                         i*self._samples_per_state+
                                         self._samples_per_state]
            
            centroids_disagrs.append(np.mean([np.mean(disagr.detach().numpy())
                                          for disagr in disagrs]))
        return centroids_disagrs
    
    def dump_plots(self, curr_budget, itr=0, show=False):
        ## Discretize all state space as defined in params and get centroids
        centroids = self._get_centroids()

        ## For each centroid compute mean model ensemble disagreement over sampled actions
        ## Use self.sampled_actions
        centroids_disagrs = self._get_state_disagr(centroids)

        ## Plot data check hist below
        
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



### Time to reach a coverage plot ###
    plt.figure()

    coverage_target_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,]# 0.8, 0.9, 1.]
    ticks = [i+1 for i, v in enumerate(coverage_target_vals)]

    rr_budget_to_reach = rr_budget_to_reach.item()
    rh_budget_to_reach = rh_budget_to_reach.item()
    mrs_budget_to_reach = mrs_budget_to_reach.item()

    rr_budget_to_reach = {key: rr_budget_to_reach[key] for key in coverage_target_vals}
    rh_budget_to_reach = {key: rh_budget_to_reach[key] for key in coverage_target_vals}
    mrs_budget_to_reach = {key: mrs_budget_to_reach[key] for key in coverage_target_vals}

    width = 0.1
    p1 = [i - 1.5*width for i in ticks] 
    p2 = [i for i in ticks] 
    p3 = [i + 1.5*width for i in ticks]
    
    # import pdb; pdb.set_trace()
    bp1 = plt.boxplot(rr_budget_to_reach.values(), patch_artist=True,
                         boxprops=dict(facecolor="C0"), widths=width, positions=p1)
    bp2 = plt.boxplot(rh_budget_to_reach.values(), patch_artist=True,
                         boxprops=dict(facecolor="C1"), widths=width, positions=p2)
    bp3 = plt.boxplot(mrs_budget_to_reach.values(), patch_artist=True,
                         boxprops=dict(facecolor="C2"), widths=width, positions=p3)

    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
               ['rr', 'rh', 'mrs'], loc='upper right')
    # colors = ['pink', 'lightblue', 'lightgreen']
    # bplots = [bplot1, bplot2, bplot3]
    # for i in range(len(bplots)):
        # for patch, color in zip(bplots[i]['boxes'], repeat(colors[i])):
            # import pdb; pdb.set_trace()
            # patch.set_facecolor(color)
    
    plt.xticks(ticks=ticks, labels=[str(i) for i in coverage_target_vals])

    plt.savefig(f"budget_to_reach_all.jpg")

    ### Show the plots ###
    plt.show()
