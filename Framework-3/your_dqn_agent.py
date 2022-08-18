import numpy as np
import random
from collections import deque
import datetime

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
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48, activation='relu')(state_h1)
        output = Dense(self.action_dim, activation='linear')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        # loss function as Mean Squared Error with an adam optimizer with given learning rate
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learningRate))
        return model

    def load_model(self, name):
        ''' loads a model, typically .h5 files are used for that'''
        # TODO: save the relevant information of the model
        
    def save_model(self, name):
        ''' saves a model '''
        # TODO: load the relevant information of the model
    
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
        cur_action_values = self.model.predict(X_cur_states)
        
        # action values for the next_states taken from our target network Q hat (s,a)
        next_action_values = self.targetmodel.predict(X_next_states)
        
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample
            if not done:
                # TODO: implement q-learning update rule
                # Q(st, at) = ...
                cur_action_values[index][action] = 0
            else:
                # TODO: implement q-learning update rule
                # Q(st, at) = ...
                cur_action_values[index][action] = 0
        
        # train the agent with new Q values for the states and the actions
        # the MSE is minimized with the adam optimizer with given learning rate
        self.model.fit(X_cur_states, cur_action_values, verbose=0)
        
        # for each updateTQNW steps we adjust the DQN update the target network with the weights
        if self.counterDQNTrained % self.updateTQNW == 0:
            # TODO: set target DQN to current DQN
            self.targetmodel = None
    
    def train(self, env):
        ''' the actual training of the agent '''
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/'+ current_time
        summary_writer = tf.summary.create_file_writer(logdir)

        # for data gathering
        max_reward = -999999
        reward_per_episode = np.array([])

        for episode in range(self.episodes):
            cur_state = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                episode_length += 1
                # set VISUALIZATION = True if want to see agent while training. But makes training a bit slower. 
                if self.visualization:
                    env.render()
                
                # epsilon handling, that is, trade of exploration vs exploitation
                if(np.random.uniform(0, 1) < self.epsilon):
                    # Take random action
                    action = np.random.randint(0, self.action_dim)
                else:
                    # Take action that maximizes the total reward
                    action = np.argmax(self.model.predict(np.expand_dims(cur_state, axis=0))[0])

                # take one step in the environment with the calculated action
                next_state, reward, done, _ = env.step(action)

                # calculate the reward of the episode as sum of rewards
                episode_reward += reward

                if done and episode_length < 200:
                    # If episode is ended we have won the game. So, give some large positive reward
                    reward = 250 + episode_reward
                    # save the model if we are getting maximum score this time
                    if(episode_reward > max_reward):
                        self.save_model(str(episode_reward)+"_agent")
                else:
                    # In other cases reward will be proportional to the distance that car has travelled 
                    # from it's previous location + velocity of the car
                    reward = 5*abs(next_state[0] - cur_state[0]) + 3*abs(cur_state[1])
                    
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
            max_reward = max(episode_reward, max_reward)
            # tensorboard log filling
            with summary_writer.as_default():
                tf.summary.scalar('reward', episode_reward, step=episode)
                tf.summary.scalar('epsilon', self.epsilon, step=episode)
                tf.summary.scalar('Timesteps per episode', episode_length, step=episode)

    def test(self, env, name):
        ''' load the weights of the DQN and perform 10 steps in the environment '''
        # create and load weights of the model
        self.load_model(name)

        # Number of episodes in which agent manages to won the game before time is over
        episodes_won = 0
        # Number of episodes for which we want to test the agnet
        TOTAL_EPISODES = 10 

        for _ in range(TOTAL_EPISODES):
            episode_reward = 0
            cur_state = env.reset()
            done = False
            episode_len = 0
            while not done:
                env.render()
                episode_len += 1
                next_state, reward, done, _ = env.step(np.argmax(self.model.predict(np.expand_dims(cur_state, axis=0))))
                if done and episode_len < 200:
                    episodes_won += 1
                cur_state = next_state
                episode_reward += reward
            print('EPISODE_REWARD', episode_reward)
            
        print(episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')