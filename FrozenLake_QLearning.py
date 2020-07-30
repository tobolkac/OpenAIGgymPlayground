import gym
import random
import numpy as np
import time
from gym.envs.registration import register

class Agent():
    def __init__(self, env, discount_rate=0.97, learning_rate=.01):
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.n

        self.epsilon = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])

    def get_action(self):
        q_state = self.q_table[state]
        action = random.choice(range(self.action_size)) if random.random() < self.epsilon else np.argmax(q_state)
        return action

    def train(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] = self.q_table[state, action] + (self.learning_rate * q_update)

        if done:
            self.epsilon = self.epsilon * 0.99

if __name__ == "__main__":
    try:
        register(
            id='FrozenLakeNoSlip-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': '4x4', 'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=0.78,
        )
    except:
        pass

    env_name = "FrozenLakeNoSlip-v0"

    env = gym.make(env_name)
    agent = Agent(env)

    total_reward = 0
    for ep in range(100):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action()
            next_state, reward, done, info = env.step(action)
            agent.train((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward

            print("state: ", state, "   action: ", action)
            print("Episode: {}, Total reward: {}".format(ep, total_reward))
            env.render()
            time.sleep(0.25)