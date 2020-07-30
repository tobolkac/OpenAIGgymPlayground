import gym
import random
import numpy as np
import tensorflow as tf
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

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def build_model(self):
        self.state_in = tf.placeholder(tf.int32, shape=[1])
        self.action_in = tf.placeholder(tf.int32, shape=[1])
        self.target_in = tf.placeholder(tf.float32, shape=[1])

        self.state = tf.one_hot(self.state_in, depth=self.state_size)
        self.action = tf.one_hot(self.action_in, depth=self.state_size)

        self.q_state = tf.layers.dense(self.state, units=self.action_size)
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)

        self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def get_action(self):
        q_state = self.session.run(self.q_state, feed_dict={self.state_in: [state]})
        action = random.choice(range(self.action_size)) if random.random() < self.epsilon else np.argmax(q_state)
        return action

    def train(self, experience):
        state, action, next_state, reward, done = ([exp] for exp in experience)

        q_next = self.session.run(self.q_state, feed_dict={self.state_in: next_state})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next)

        feed = {self.state_in: state, self.action_in: action, self.target_in: q_target}
        self.session.run(self.optimizer, feed_dict=feed)

        if experience[4]:
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