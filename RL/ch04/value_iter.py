if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy
from collections import defaultdict

def value_iter_onestep(V, env, gamma):
    for state in env.states(): #모든 상태에 접근
        if state == env.goal_state: #goal에서의 value는 0
            V[state] = 0
            continue
        action_values = []
    
        for action in env.actions(): # 모든 action에 접근
            next_state = env.next_state(state,action) # s'
            r = env.reward(state,action,next_state) # r(s,a,s')
            value = r + gamma*V[next_state] # new value function
            action_values.append(value)

        V[state] = max(action_values) # 최대값
    return V
    
def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)
        
        old_V = V.copy() # 갱신 전 value function
        V = value_iter_onestep(V,env, gamma) #update

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state]) # 갱신된 양
            if delta < t: # 점진적인 갱신을 위함
                delta = t
        
        if delta < threshold:
            break

    return V

V = defaultdict(lambda: 0)
env = GridWorld()
gamma = 0.9

V = value_iter(V, env, gamma)

pi = greedy_policy(V, env, gamma)
env.render_v(V, pi)