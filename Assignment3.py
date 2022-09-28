import gym
from agents import DQNAgent, Dueling_DQN_Agent

# Variables
class RLrunner():
    def __init__(self, env=gym.make('CartPole-v1'), num_steps=10):
        self.env = env
        self.num_steps = num_steps
        self.obs = self.env.reset()
        self.config = {
            "EPISODES": 300,
            "REPLAY_MEMORY_SIZE": 256,
            "MINIMUM_REPLAY_MEMORY": 256,
            "MINIBATCH_SIZE": 64,
            "UPDATE_TARGETNW_STEPS": 64,
            "LEARNING_RATE": 0.001,
            "EPSILON": 0.3,
            "EPSILON_DECAY": 0.99,
            "MINIMUM_EPSILON": 0.001,
            "DISCOUNT": 0.9,
            "VISUALIZATION": False
        }

    def run_cartpole(self):
        # Our model to solve the mountain-car problem.
        self.agent = DQNAgent(self.env, self.config)
        self.reward_log = self.agent.train(self.env)
        #self.total_reward = 0

        for step in range(self.num_steps):
            action = self.env.action_space.sample()

            obs, reward, done, info = self.env.step(action)
            #self.total_reward += reward

            self.env.render()
            if done:
                self.env.reset()
                #self.reward_log.append(self.total_reward)

        self.env.close()
        return self.reward_log

    def test(self, number):
        self.agent = DQNAgent(self.env, self.config)
        self.agent.test(env=self.env, name='./models/500.0_agent', number=number)

    def run_dueling_cartpole(self):
        # Our model to solve the mountain-car problem.
        self.agent = Dueling_DQN_Agent(self.env, self.config)
        self.reward_log = self.agent.train(self.env)
        #self.total_reward = 0

        for step in range(self.num_steps):
            action = self.env.action_space.sample()

            obs, reward, done, info = self.env.step(action)
            #self.total_reward += reward

            self.env.render()
            if done:
                self.env.reset()
                #self.reward_log.append(self.total_reward)

        self.env.close()
        return self.reward_log

    def test_dueling(self, number):
        self.agent = Dueling_DQN_Agent(self.env, self.config)
        self.agent.test(self.env, name='./models/500.0_dueling_agent', number=number)

class Random_agent():
    def __init__(self, test_episodes=100):
        self.env = gym.make('CartPole-v1')
        self.test_episodes = test_episodes
        # --------- Communicate dimensions of env to agent
        self.states = len(self.env.observation_space.low)
        self.n_actions = self.env.action_space.n
        self.actions = [i for i in range(self.n_actions)]
        # visualization
        self.timestep = self.test_episodes / 10
        self.history = []
        self.reward_data = []
        self.epsilon_data = []

    def test(self):
        reward = []
        for i in range(self.test_episodes):
            ep_reward = 0
            s = self.env.reset()
            done = False

            while done != True:
                action = self.env.action_space.sample()
                s_, r, done, info = self.env.step(action)

                s = s_
                ep_reward += r

            reward.append(ep_reward)
        return reward


