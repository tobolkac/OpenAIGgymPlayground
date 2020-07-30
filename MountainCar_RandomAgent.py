import gym
import random

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size:",  self.action_size)

    def get_action(self, state):
        action = random.choice(range(self.action_size))
        return action

if __name__ == "__main__":
    env_name = "MountainCar-v0"

    env = gym.make(env_name)
    agent = Agent(env)

    state = env.reset()

    for _ in range(200):
        action = agent.get_action(state)
        state, _, _, _ = env.step(action)
        env.render()