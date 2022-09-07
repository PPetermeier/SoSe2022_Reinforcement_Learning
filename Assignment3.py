import gym
import numpy as np
import seaborn as sns
import time
from dqn_agent import DQNAgent

# Variables
class RLrunner():
    def __init__(self, env=gym.make('CartPole-v1'), num_steps=20):
        self.env = env
        self.num_steps = num_steps
        self.obs = self.env.reset()
        self.config =  {
            "EPISODES": 800,
            "REPLAY_MEMORY_SIZE": 1_00_000,
            "MINIMUM_REPLAY_MEMORY": 1_000,
            "MINIBATCH_SIZE": 64,
            "UPDATE_TARGETNW_STEPS": 200,
            "LEARNING_RATE": 0.001,
            "EPSILON": 1,
            "EPSILON_DECAY": 0.99,
            "MINIMUM_EPSILON": 0.001,
            "DISCOUNT": 0.99,
            "VISUALIZATION": True
        }

    def run_cartpole(self):
        # Our model to solve the mountain-car problem.
        self.agent = DQNAgent(self.env, self.config)
        self.agent.train(self.env)

        for step in range(self.num_steps):
            action = self.env.action_space.sample()

            obs, reward, done, info = self.env.step(action)

            self.env.render()

            time.sleep(0.01)

            if done:
                self.env.reset()

        self.env.close()