from mb_ge.go.go_method import GoMethod

class ExecutePolicyGo(GoMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        
    def go(self, gym_env, el):
        ## reconstruct needed-policy chaining (deterministic-case)
        policies_to_chain = []
        len_under_policy = []
        budget_used = 0
        transitions = []
        policies_to_chain.insert(0, el.policy_parameters)
        len_under_policy.insert(0, len(el.trajectory))
        prev_el = el.previous_element
        ## WARNING: might have to copy values
        while prev_el != None:
            if prev_el.policy_parameters is not None:
                policies_to_chain.insert(0, prev_el.policy_parameters)
                len_under_policy.insert(0, len(prev_el.trajectory))
            prev_el = prev_el.previous_element
        ## Replay policies from initial state to el goal state
        obs = gym_env.get_obs()
        ## Check if el is init elem
        if el.policy_parameters is None:
            transitions.append((None, obs))
            return transitions, budget_used
        for policy_params, h in zip(policies_to_chain, len_under_policy):
            self.controller.set_parameters(policy_params)
            for _ in range(h):
                action = self.controller(obs)
                transitions.append((action, obs))
                obs, reward, done, info = gym_env.step(action)
                budget_used += 1
        transitions.append((None, obs))
        return transitions, budget_used
