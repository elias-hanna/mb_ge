class Element():
    def __init__(self, descriptor=None, trajectory=None, actions=[], reward=0.,
                 disagreement=None, policy_parameters=None, previous_element=None, sim_state=None,
                 total_trajectory_len=0):
        self.descriptor = descriptor
        self.trajectory = trajectory
        self.actions = actions
        self.reward = reward
        self.novelty = 0.
        self.disagreement = disagreement
        self.end_state_disagr = 0.
        self.trajectory_disagr = 0.
        self.policy_parameters = policy_parameters
        self.previous_element = previous_element ## allows to chain policies
        self.sim_state = sim_state ## allows to restore a sim state if using a simulator
        self.total_trajectory_len = previous_element.total_trajectory_len + len(self.trajectory) \
                                    if previous_element is not None else len(self.trajectory)

    def __lt__(self, other):
        return True
