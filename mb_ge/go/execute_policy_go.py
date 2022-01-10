from mb_ge.go.go_method import GoMethod

class ExecutePolicyGo(GoMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        
    def go(self, gym_env, el):
        ## reconstruct needed-policy chaining (deterministic-case)
        policies_to_chain = []
        len_under_policy = []
        budget_used = 0
        policies_to_chain.insert(0, el.policy_parameters)
        prev_el = el.previous_element
        to_check = []
        to_check_traj = []
        ## WARNING: might have to copy values
        while prev_el != None:
            if prev_el.policy_parameters is not None:
                policies_to_chain.insert(0, prev_el.policy_parameters)
                len_under_policy.insert(0, len(prev_el.trajectory))
                to_check.insert(0, prev_el.descriptor)
                to_check_traj.insert(0, prev_el.trajectory)
            prev_el = prev_el.previous_element
        # print(policies_to_chain)
        # print(len_under_policy)
        ## Replay policies from initial state to el goal state
        obs = gym_env._get_obs()
        traj = []
        for policy_params, h in zip(policies_to_chain, len_under_policy):
            self.controller.set_parameters(policy_params)
            for _ in range(h):
                traj.append(obs)
                action = self.controller(obs)
                obs, reward, done, info = gym_env.step(action)
                budget_used += 1
            import pdb; pdb.set_trace()
        return budget_used
