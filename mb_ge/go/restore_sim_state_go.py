from mb_ge.go.go_method import GoMethod

class RestoreSimStateGo(GoMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        
    def go(self, gym_env, el):
        ## reconstruct needed-policy chaining (deterministic-case)
        budget_used = 0
        prev_el = el.previous_element
        ## Restore sim state to last prev el
        gym_env.set_state(prev_el.sim_state['qpos'], prev_el.sim_state['qvel'])
        ## WARNING: might have to copy values
        while prev_el != None:
            if prev_el.policy_parameters is not None:
                budget_used += len(prev_el.trajectory)
            prev_el = prev_el.previous_element
        ## Play el policy starting from restored state
        obs = gym_env.get_obs()
        ## Check if el is init elem
        # if el.policy_parameters == []:
        if el.policy_parameters is None:
            transitions.append((None, obs))
            return transitions, budget_used
        for _ in range(len(el.trajectory)):
            action = self.controller(obs)
            transitions.append((action, obs))
            obs, reward, done, info = gym_env.step(action)
            budget_used += 1
        return [], budget_used
