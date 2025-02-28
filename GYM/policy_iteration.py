import gym
import numpy as np
from common.make_gif import plot_animation

GAME = "FrozenLake-v1"
SAVE_PATH = f"video/{GAME}_slippery.gif"
TITLE = f"{GAME}_Policy_Iteration"

NUM_EPISODE = 200
GAMMA = 0.99
THETA = 1e-6

def policy_evaluation(env, policy, V, gamma=GAMMA, theta=THETA):
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = 0
            action = policy[state]
            for prob, next_state, reward, done in env.P[state][action]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[state] - v))
            V[state] = v
        if delta < theta:
            break
    return V

def policy_improvement(env, policy, V, gamma=GAMMA):
    policy_stable = True
    for state in range(env.observation_space.n):
        old_action = policy[state]
        action_values = np.zeros(env.action_space.n)
        
        for action in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (reward + gamma * V[next_state])
        
        new_action = np.argmax(action_values)
        if old_action != new_action:
            policy_stable = False
        policy[state] = new_action
    
    return policy, policy_stable

def policy_iteration(env, gamma=GAMMA):
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)
    V = np.zeros(env.observation_space.n)
    is_policy_stable = False
    frames = []

    while not is_policy_stable:
        V = policy_evaluation(env, policy, V, gamma)
        policy, is_policy_stable = policy_improvement(env, policy, V, gamma)
    
    return policy, V, frames

if __name__ == "__main__":
    env = gym.make(GAME, is_slippery=True, render_mode="rgb_array")
    
    state, _ = env.reset()
    optimal_policy, state_value_function, frames = policy_iteration(env, gamma=GAMMA)

    for episode in range(NUM_EPISODE):
        state, _ = env.reset()
        done = False
        
        while not done:
            frames.append(env.render())
            action = optimal_policy[state]
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state 

    plot_animation(frames, save_path=SAVE_PATH, title=TITLE, repeat=False, interval=1000)
    env.close()

    print(f"Optimal Policy: {optimal_policy}")
    print(f"State Value Function: {state_value_function}")
