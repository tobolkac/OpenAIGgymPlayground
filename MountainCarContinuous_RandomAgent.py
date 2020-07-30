import gym
import numpy as np

class Agent():
    def __init__(self, env):
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_shape = env.action_space.shape

    def get_action(self):
        action = np.random.uniform(self.action_low,
                                  self.action_high,
                                  self.action_shape)
        return action

if __name__ == "__main__":
    env_name = "MountainCarContinuous-v0"

    env = gym.make(env_name)
    agent = Agent(env)

    state = env.reset()

    for _ in range(200):
        action = agent.get_action()
        state, _, _, _ = env.step(action)
        env.render()