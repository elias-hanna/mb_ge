from mb_ge.go.go_method import GoMethod

class RestoreSimStateGo(GoMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        
    def go(self, gym_env, el):
        ## reconstruct needed-policy chaining (deterministic-case)
        budget_used = len(el.trajectory)
        prev_el = el.previous_element
        ## Restore sim state to last prev el
        gym_env.set_state(prev_el.sim_state['qpos'], prev_el.sim_state['qvel'])
        ## WARNING: might have to copy values
        while prev_el != None:
            if prev_el.policy_parameters is not None:
                budget_used += len(prev_el.trajectory)
            prev_el = prev_el.previous_element
        return [], budget_used