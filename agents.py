import numpy as np
import random
from collections import deque
import datetime

import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DQNAgent:
    def __init__(self, env, config):
                
        # used parameter within the agent
        self.episodes = config['EPISODES']

        self.epsilon = config['EPSILON']
        self.epsDecay = config['EPSILON_DECAY']
        self.minEps = config['MINIMUM_EPSILON']

        self.discount = config['DISCOUNT']
        self.miniBatchSize = config['MINIBATCH_SIZE']
        self.minReplayMem = config['MINIMUM_REPLAY_MEMORY']
        self.updateTQNW = config['UPDATE_TARGETNW_STEPS']
        self.learningRate = config['LEARNING_RATE']
        self.visualization = config['VISUALIZATION']
        
        # Replay memory to store experiences of the model with the environment
        self.replay_memory = deque(maxlen=config['REPLAY_MEMORY_SIZE'])
        
        # dimensions of the action and state space
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape

        # both q networks for online action choice and the target network
        self.model = self.create_model()
        self.targetmodel = self.model

        # counter for training steps used for updating the target network from time to time (defined in config)
        self.counterDQNTrained = 0

    def create_model(self):
        ''' DQN definition, from 2 inputs to 2 hidden layers with 24, 48 nodes with relu activation function. 
        Output layer has 3 nodes with a linear activation function '''
        state_input = Input(shape=(self.observation_dim))
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128, activation='relu')(state_h1)

        # We had to add layers here!
        output = Dense(self.action_dim, activation='linear')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        # loss function as Mean Squared Error with an adam optimizer with given learning rate
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learningRate))
        return model

    def load_model(self, name):
        ''' loads a model, that is, the weights of the DQN function approximator '''
        self.model.load_weights(name+".h5")
        
    def save_model(self, name):
        ''' saves the weights of the DQN '''
        self.model.save_weights(name+".h5")
    
    def trainDQN(self):
        self.counterDQNTrained += 1
        
        # minibatch handling for experience replay
        minibatch = random.sample(self.replay_memory, self.miniBatchSize)
        X_cur_states = [] # s
        X_next_states = [] # s'
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample # s, a, r, s'
            X_cur_states.append(cur_state)
            X_next_states.append(next_state)

        X_cur_states = np.array(X_cur_states)
        X_next_states = np.array(X_next_states)

        # action values for the current_states Q(s,a). Be aware that 3 values are outputted for each state
        cur_action_values = self.model.predict(X_cur_states, verbose=0)
        
        # action values for the next_states taken from our target network Q hat (s,a)
        next_action_values = self.targetmodel.predict(X_next_states, verbose=0)
        
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample
            if not done:
                # Q(st, at) = rt + gamma * max(Q hat(s(t+1), a(t+1)))
                cur_action_values[index][action] = reward + self.discount * np.amax(next_action_values[index])
            else:
                # Q(st, at) = rt
                cur_action_values[index][action] = reward
                
        # train the agent with new Q values for the states and the actions
        # the MSE is minimized with the adam optimizer with given learning rate
        self.model.fit(X_cur_states, cur_action_values, verbose=1)
        
        # for each updateTQNW steps we adjust the DQN update the target network with the weights
        if self.counterDQNTrained % self.updateTQNW == 0:
            self.targetmodel.set_weights(self.model.get_weights())

    def train(self, env):
        ''' the actual training of the agent '''
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/'+ current_time
        summary_writer = tf.summary.create_file_writer(logdir)

        # for data gathering
        max_reward = -999999
        scores_deque = deque(maxlen=100)

        for episode in range(self.episodes):
            cur_state = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                episode_length += 1
                # set VISUALIZATION = True if want to see agent while training. But makes training a bit slower. 
                # Default is showing the agent after each 50 episodes
                if self.visualization:
                    env.render()

                if(np.random.uniform(0, 1) < self.epsilon):
                    # Take random action
                    action = np.random.randint(0, self.action_dim)
                else:
                    # Take action that maximizes the total reward
                    action = np.argmax(self.model.predict(np.expand_dims(cur_state, axis=0), verbose=0)[0])

                next_state, reward, done, _ = env.step(action)

                episode_reward += reward

                if done:
                    if (episode_reward > max_reward):
                        self.save_model(str(episode_reward)+"_agent")
           #    self.save_model("Episode_"+str(episode)+"_agent")
                # Add experience to replay memory buffer
                self.replay_memory.append((cur_state, action, reward, next_state, done))
                cur_state = next_state

                # only train DQN if at least once the replay memory is full
                if(len(self.replay_memory) < self.minReplayMem):
                    continue

                self.trainDQN()


            if(self.epsilon > self.minEps and len(self.replay_memory) > self.minReplayMem):
                self.epsilon *= self.epsDecay

            # some bookkeeping.
            scores_deque.append(episode_reward)
            max_reward = max(episode_reward, max_reward)
            # tensorboard log filling
            with summary_writer.as_default():
                tf.summary.scalar('reward', episode_reward, step=episode)
                tf.summary.scalar('epsilon', self.epsilon, step=episode)
                tf.summary.scalar('Timesteps per episode', episode_length, step=episode)
                tf.summary.scalar('Avg score over 100 episode', np.mean(scores_deque), step=episode)

    def test(self, env, name, number):
        ''' load the weights of the DQN and perform 10 steps in the environment '''
        # create and load weights of the model
        self.load_model(name)

        # Number of episodes in which agent manages to won the game before time is over
        episodes_won = 0
        # Number of episodes for which we want to test the agent
        TOTAL_EPISODES = number

        for _ in range(TOTAL_EPISODES):
            episode_reward = 0
            cur_state = env.reset()[0]
            done = False
            episode_len = 0
            while not done:
                env.render()
                episode_len += 1
                step = np.expand_dims(cur_state, axis=0)
                step = np.asarray(step).astype(np.float32)
                # print(env.step(np.argmax(self.model.predict(step, verbose=0))))
                output = self.model.predict(step, verbose=0)
                action = np.argmax(output)
                next_state, reward, done, _, __ = env.step(action)
                if episode_len > 475:  # threshold for CartPole-v1
                    episodes_won += 1
                    break
                cur_state = next_state
                episode_reward += reward
            print('VANILLA: EPISODE_REWARD', episode_reward)
            
        print('VANILLA: ', episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')


class Dueling_DQN_Agent:
    def __init__(self, env, config):

        # used parameter within the agent
        self.episodes = config['EPISODES']

        self.epsilon = config['EPSILON']
        self.epsDecay = config['EPSILON_DECAY']
        self.minEps = config['MINIMUM_EPSILON']

        self.discount = config['DISCOUNT']
        self.miniBatchSize = config['MINIBATCH_SIZE']
        self.minReplayMem = config['MINIMUM_REPLAY_MEMORY']
        self.updateTQNW = config['UPDATE_TARGETNW_STEPS']
        self.learningRate = config['LEARNING_RATE']
        self.visualization = config['VISUALIZATION']

        # Replay memory to store experiences of the model with the environment
        self.replay_memory = deque(maxlen=config['REPLAY_MEMORY_SIZE'])

        # dimensions of the action and state space
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape
        self.reward_log = []
        # both q networks for online action choice and the target network
        self.model = self.create_model()

        self.targetmodel = self.model
        # counter for training steps used for updating the target network from time to time (defined in config)
        self.counterDQNTrained = 0

    def create_model(self):
        ''' DQN definition, from 2 inputs to 2 hidden layers with 24, 48 nodes with relu activation function.
        Output layer has 3 nodes with a linear activation function '''
        state_input = Input(shape=(self.observation_dim))
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128, activation='relu')(state_h1)
        # Split q-function into two components: value of state and advantage of next action (N = num_actions)
        state_value = Dense(1)(state_h2)
        advantage = Dense(self.action_dim)(state_h2)
        #
        q_value = state_value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

        output = Dense(self.action_dim, activation='linear')(q_value)
        model = Model(inputs=state_input, outputs=output)
        # loss function as Mean Squared Error with an adam optimizer with given learning rate
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learningRate))
        return model

    def load_model(self, name):
        ''' loads a model, that is, the weights of the DQN function approximator '''
        self.model.load_weights(name + ".h5")

    def save_model(self, name):
        ''' saves the weights of the DQN '''
        self.model.save_weights(name + ".h5")

    def trainDQN(self):
        self.counterDQNTrained += 1

        # minibatch handling for experience replay
        minibatch = random.sample(self.replay_memory, self.miniBatchSize)
        X_cur_states = []  # s
        X_next_states = []  # s'
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample  # s, a, r, s'
            X_cur_states.append(cur_state)
            X_next_states.append(next_state)

        X_cur_states = np.array(X_cur_states)
        X_next_states = np.array(X_next_states)

        # action values for the current_states Q(s,a). Be aware that 3 values are outputted for each state
        cur_action_values = self.model.predict(X_cur_states, verbose=0)

        # action values for the next_states taken from our target network Q hat (s,a)
        next_action_values = self.targetmodel.predict(X_next_states, verbose=0)

        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample
            if not done:
                # Q(st, at) = rt + gamma * max(Q hat(s(t+1), a(t+1)))
                cur_action_values[index][action] = reward + self.discount * np.amax(next_action_values[index])
            else:
                # Q(st, at) = rt
                cur_action_values[index][action] = reward

        # train the agent with new Q values for the states and the actions
        # the MSE is minimized with the adam optimizer with given learning rate
        self.model.fit(X_cur_states, cur_action_values, verbose=1,)

        # for each updateTQNW steps we adjust the DQN update the target network with the weights
        if self.counterDQNTrained % self.updateTQNW == 0:
            self.targetmodel.set_weights(self.model.get_weights())

    def train(self, env, name, number):
        ''' the actual training of the agent '''
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/' + current_time +"_dueling"
        summary_writer = tf.summary.create_file_writer(logdir)

        # for data gathering
        max_reward = -999999
        scores_deque = deque(maxlen=100)

        for episode in range(self.episodes):
            cur_state = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                episode_length += 1
                # set VISUALIZATION = True if want to see agent while training. But makes training a bit slower.
                # Default is showing the agent after each 50 episodes
                if self.visualization:
                    env.render()

                if (np.random.uniform(0, 1) < self.epsilon):
                    # Take random action
                    action = np.random.randint(0, self.action_dim)
                else:
                    # Take action that maximizes the total reward
                    action = np.argmax(self.model.predict(np.expand_dims(cur_state, axis=0), verbose=0)[0])

                next_state, reward, done, _ = env.step(action)

                episode_reward += reward
                # Added:
                if done:
                    if (episode_reward > max_reward):
                        self.save_model(str(episode_reward) + "_dueling_agent")

                    # elif (episode % 50 == 0):
                    #    self.save_model("Episode_"+str(episode)+"_agent")
                # Add experience to replay memory buffer
                self.replay_memory.append((cur_state, action, reward, next_state, done))
                cur_state = next_state

                # only train DQN if at least once the replay memory is full
                if (len(self.replay_memory) < self.minReplayMem):
                    continue

                self.trainDQN()

            if (self.epsilon > self.minEps and len(self.replay_memory) > self.minReplayMem):
                self.epsilon *= self.epsDecay

            # some bookkeeping.
            scores_deque.append(episode_reward)
            max_reward = max(episode_reward, max_reward)

            # tensorboard log filling
            with summary_writer.as_default():
                tf.summary.scalar('reward', episode_reward, step=episode)
                tf.summary.scalar('epsilon', self.epsilon, step=episode)
                tf.summary.scalar('Timesteps per episode', episode_length, step=episode)
                tf.summary.scalar('Avg score over 100 episode', np.mean(scores_deque), step=episode)
        self.reward_log.append(episode_reward)
        return self.reward_log

    def test(self, env, name, number):
        ''' load the weights of the DQN and perform 10 steps in the environment '''
        # create and load weights of the model
        self.load_model(name)

        # Number of episodes in which agent manages to won the game before time is over
        episodes_won = 0
        # Number of episodes for which we want to test the agent
        TOTAL_EPISODES = number

        for _ in range(TOTAL_EPISODES):
            episode_reward = 0
            cur_state = env.reset()[0]
            done = False
            episode_len = 0
            while not done:
                env.render()
                episode_len += 1

                step = np.expand_dims(cur_state, axis=0)
                step = np.asarray(step).astype(np.float32)
                output = self.model.predict(step, verbose=0)
                action = np.argmax(output)

                next_state, reward, done, _, __ = env.step(action)
                if episode_len > 475:  # threshold for CartPole-v1
                    episodes_won += 1
                    break
                cur_state = next_state
                episode_reward += reward
            print('DUELING: EPISODE_REWARD', episode_reward)

        print('DUELING: ', episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')