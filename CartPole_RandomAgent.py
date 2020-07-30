import gym
import random

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size:",  self.action_size)

    def get_action (self):
        action = random.choice(range(self.action_size))
        return action

if __name__ == "__main__":
    env_name = "CartPole-v1"

    env = gym.make(env_name)
    agent = Agent(env)

    env.reset()

    for _ in range(200):
        action = agent.get_action()
        env.step(action)
        env.render()