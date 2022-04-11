import numpy as np

from matplotlib import pyplot as plt

if __name__ == '__main__':

    ge_random_random_data = np.load('fixed_horizon_ge_random_random_data.npz')
    ge_random_heuristic_data = np.load('fixed_horizon_ge_random_heuristic_data.npz')
    mb_ge_random_statedisagr_data = np.load('fixed_horizon_mb_ge_random_statedisagr_data.npz')


    rr_coverage_mean = ge_random_random_data['coverage_mean']
    rr_coverage_error = ge_random_random_data['coverage_error']
    rr_reward_mean = ge_random_random_data['reward_mean']
    rr_reward_error = ge_random_random_data['reward_error']

    rh_coverage_mean = ge_random_heuristic_data['coverage_mean']
    rh_coverage_error = ge_random_heuristic_data['coverage_error']
    rh_reward_mean = ge_random_heuristic_data['reward_mean']
    rh_reward_error = ge_random_heuristic_data['reward_error']

    mrs_coverage_mean = mb_ge_random_statedisagr_data['coverage_mean']
    mrs_coverage_error = mb_ge_random_statedisagr_data['coverage_error']
    mrs_reward_mean = mb_ge_random_statedisagr_data['reward_mean']
    mrs_reward_error = mb_ge_random_statedisagr_data['reward_error']

    
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

    plt.show()
