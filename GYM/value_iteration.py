import gym
import numpy as np
from common.make_gif import plot_animation

GAME = "Taxi-v3"
SAVE_PATH = f"video/{GAME}.gif"
TITLE = f"{GAME}_Value_Iteration"

GAMMA = 0.99 
THETA = 1e-8 
NUM_EPISODES = 30

def value_iteration(env, gamma=GAMMA, theta=THETA):
    state_num = env.observation_space.n
    action_num = env.action_space.n

    V = np.zeros(state_num)
    while True:
        delta = 0
        for state in range(state_num):
            action_values = np.zeros(action_num)
            for action in range(action_num):
                for prob, next_state, reward, done in env.P[state][action]:
                    action_values[action] += prob * (reward + gamma * V[next_state])
            
            best_action_value = np.max(action_values)
            delta = max(delta, abs(V[state] - best_action_value))
            V[state] = best_action_value 
        
        if delta < theta:
            break
    
    policy = np.zeros(state_num, dtype=int)
    for state in range(state_num):
        action_values = np.zeros(action_num)
        for action in range(action_num):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (reward + gamma * V[next_state])
        policy[state] = np.argmax(action_values)
    
    return policy, V

if __name__ == "__main__":
    env = gym.make(GAME, render_mode="rgb_array")
    optimal_policy, state_value_function = value_iteration(env, gamma=GAMMA)

    #print(f"Optimal Policy: {optimal_policy}")
    #print(f"State Value Function: {state_value_function}")

    frames = []
    total_rewards = [] 

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            frames.append(env.render())
            action = optimal_policy[state]
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state

        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Total Reward = {episode_reward}")

    print(f"Average Reward over {NUM_EPISODES} episodes: {np.mean(total_rewards)}")
    plot_animation(frames, save_path=SAVE_PATH, title=TITLE, repeat=False, interval=1000)
    env.close()
