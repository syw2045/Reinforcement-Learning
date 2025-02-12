import gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

for i in range(10):
    done = False
    game_reward = 0
    obs, _ = env.reset()

    while not done:
        action = env.action_space.sample()
        new_obs, reward, done, truncated, info = env.step(action)
        game_reward += reward

        if done or truncated:
            print(f"Episode {i} finished, reward: {game_reward}")
            break

env.close()
