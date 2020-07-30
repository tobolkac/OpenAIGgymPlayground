import gym

if __name__ == "__main__":
    env_name = "CartPole-v1"

    env = gym.make(env_name)

    env.reset()

    for _ in range(200):
        action = env.action_space.sample()
        env.step(action)
        env.render()