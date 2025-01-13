if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from common.gridworld import GridWorld
from collections import defaultdict

def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        action_probs = pi[state]
        new_V = 0

        for action, action_probs in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)

            new_V += action_probs * (r + gamma * V[next_state])

        V[state] = new_V

    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy() # 갱신 전 가치 함수
        V = eval_onestep(pi, V, env, gamma)

        # 갱신된 양의 최대값 계산 
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
            

        if delta < threshold:
            break

    return V

env = GridWorld()
gamma = 0.9
pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)
