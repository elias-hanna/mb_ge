class Element():
    def __init__(self, descriptor=None, trajectory=None, actions=None, reward=0.,
                 disagreement=None, policy_parameters=None, previous_element=None, sim_state=None):
        self.descriptor = descriptor
        self.trajectory = trajectory
        self.actions = actions
        self.reward = reward
        self.novelty = 0.
        self.disagreement = disagreement
        self.policy_parameters = policy_parameters
        self.previous_element = previous_element ## allows to chain policies
        self.sim_state = sim_state ## allows to restore a sim state if using a simulator
