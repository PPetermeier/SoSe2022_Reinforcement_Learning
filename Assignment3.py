import gym
from agents import DQNAgent, Dueling_DQN_Agent


class RLrunner():
    """
    Wrapper to orchestrate creation and training of agents as well as simulation of random agent.
    """
    def __init__(self, env=gym.make('CartPole-v1'), num_steps=10) -> None:
        """
        Creates env with gym and sets it into starting position, takes number of steps,
        and given configuration of hyperparameters, which have to changed here if needed.
        The saved configuration is the one with which the saved data was generated with.
        """
        self.env = env
        self.num_steps = num_steps
        self.obs = self.env.reset()
        self.config = {
            'EPISODES': 300,
            'REPLAY_MEMORY_SIZE': 256,
            'MINIMUM_REPLAY_MEMORY': 256,
            'MINIBATCH_SIZE': 64,
            'UPDATE_TARGETNW_STEPS': 64,
            'LEARNING_RATE': 0.001,
            'EPSILON': 0.3,
            'EPSILON_DECAY': 0.99,
            'MINIMUM_EPSILON': 0.001,
            'DISCOUNT': 0.9,
            'VISUALIZATION': False
        }

    def run_cartpole(self) -> None:
        """
        Creates a vanilla DQN agent with given config and trains it with given parameters.
        """
        self.agent = DQNAgent(self.env, self.config)
        self.agent.train(self.env) # TODO: Ich glaub das ist redundand. Testen!

        for step in range(self.num_steps):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                self.env.reset()

        self.env.close()

    def test(self, number) -> None:
        """
        Loads the saved model with env and config and runs tests with visualization of said model.
        """
        self.agent = DQNAgent(self.env, self.config)
        self.agent.test(env=self.env, name='./models/500.0_agent', number=number)

    def run_dueling_cartpole(self) -> None:
        """
        Creates a dueling-q agent with given config and trains it with given parameters.
        """
        self.agent = Dueling_DQN_Agent(self.env, self.config)
        self.agent.train(self.env)

        for step in range(self.num_steps):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                self.env.reset()

        self.env.close()

    def test_dueling(self, number) -> None:
        """
        Loads the saved model with env and config and runs tests with visualization of said model.
        """
        self.agent = Dueling_DQN_Agent(self.env, self.config)
        self.agent.test(self.env, name='./models/500.0_dueling_agent', number=number)

class Random_agent():
    """
    Class create to simulate behaviour in CartPole Environment with an agent flipping a coin.
    Uses attributes similar to given DQN agent for simplicity and readability.
    Consists mainly of the test functions running the simulation.
    """
    def __init__(self, test_episodes=100) -> None:
        self.env = gym.make('CartPole-v1')
        self.test_episodes = test_episodes
        # --------- Communicate dimensions of env to agent
        self.states = len(self.env.observation_space.low)
        self.n_actions = self.env.action_space.n
        self.actions = list(range(self.n_actions))

    def test(self) -> list:
        """
        Creates an empty list, runs through the env by sampling a random action
        out of the action space and returns the cumulative rewards
        after all runs are done as a list.
        Done to establish a baseline performance to compare both agents to
        and to create samples to create the plots with.
        """
        reward = []
        for i in range(self.test_episodes):
            ep_reward = 0
            s = self.env.reset()
            done = False

            while not done:
                action = self.env.action_space.sample()
                s_, r, done, info = self.env.step(action)

                s = s_
                ep_reward += r

            reward.append(ep_reward)
        return reward
