import numpy as np
from itertools import repeat

from matplotlib import pyplot as plt

if __name__ == '__main__':

    ge_random_random_data = np.load('fixed_horizon_ge_random_random_data.npz',
                                    allow_pickle=True)
    ge_random_heuristic_data = np.load('fixed_horizon_ge_random_heuristic_data.npz',
                                       allow_pickle=True)
    mb_ge_random_statedisagr_data = np.load('fixed_horizon_mb_ge_random_statedisagr_data.npz',
                                            allow_pickle=True)


    rr_coverage_mean = ge_random_random_data['coverage_mean']
    rr_coverage_error = ge_random_random_data['coverage_error']
    rr_reward_mean = ge_random_random_data['reward_mean']
    rr_reward_error = ge_random_random_data['reward_error']
    rr_budget_to_reach = ge_random_random_data['budget_to_reach']
    
    rh_coverage_mean = ge_random_heuristic_data['coverage_mean']
    rh_coverage_error = ge_random_heuristic_data['coverage_error']
    rh_reward_mean = ge_random_heuristic_data['reward_mean']
    rh_reward_error = ge_random_heuristic_data['reward_error']
    rh_budget_to_reach = ge_random_heuristic_data['budget_to_reach']

    mrs_coverage_mean = mb_ge_random_statedisagr_data['coverage_mean']
    mrs_coverage_error = mb_ge_random_statedisagr_data['coverage_error']
    mrs_reward_mean = mb_ge_random_statedisagr_data['reward_mean']
    mrs_reward_error = mb_ge_random_statedisagr_data['reward_error']
    mrs_budget_to_reach = mb_ge_random_statedisagr_data['budget_to_reach']

    
    label = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]

    plt.figure()

    ### Coverage plot ###

    ## RR plot in black and green
    plt.plot(label, rr_coverage_mean, 'gp-', label='rr_coverage_mean')
    plt.fill_between(label, rr_coverage_mean-rr_coverage_error, rr_coverage_mean+rr_coverage_error,
                     facecolor='green', alpha=0.5)

    ## RH plot in blue and yellow
    plt.plot(label, rh_coverage_mean, 'yp-', label='rh_coverage_mean')
    plt.fill_between(label, rh_coverage_mean-rh_coverage_error, rh_coverage_mean+rh_coverage_error,
                     facecolor='yellow', alpha=0.5)

    ## MRS plot in green and purple
    plt.plot(label, mrs_coverage_mean, 'cp-', label='mrs_coverage_mean')
    plt.fill_between(label, mrs_coverage_mean-mrs_coverage_error,
                     mrs_coverage_mean+mrs_coverage_error, facecolor='cyan', alpha=0.5)
    
    plt.legend()
    plt.title(f"Coverage depending on number of iterations")
    plt.savefig(f"coverage_all.jpg")

    ### Reward plot ###

    ## RR plot in black and green
    plt.figure()
    plt.plot(label, rr_reward_mean, 'k-', label='rr_reward_mean')
    plt.fill_between(label, rr_reward_mean-rr_reward_error, rr_reward_mean+rr_reward_error,
                     facecolor='green', alpha=0.5)

    ## RH plot in blue and yellow
    plt.plot(label, rh_reward_mean, 'b-', label='rh_reward_mean')
    plt.fill_between(label, rh_reward_mean-rh_reward_error, rh_reward_mean+rh_reward_error,
                     facecolor='yellow', alpha=0.5)

    ## MRS plot in green and purple
    plt.plot(label, mrs_reward_mean, 'g-', label='mrs_reward_mean')
    plt.fill_between(label, mrs_reward_mean-mrs_reward_error, mrs_reward_mean+mrs_reward_error,
                     facecolor='yellow', alpha=0.5)

    plt.legend()
    plt.title(f"Number of rewarded policies depending on number of iterations")
    plt.savefig(f"reward_all.jpg")

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
