import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque
import logging
import random
from datetime import datetime
np.random.seed(1)
tf.set_random_seed(2)
from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from environment.maze import actions
from models import AbstractModel
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



class ExperienceReplay():
    """ Store game transitions (from state s to s' via action a) and record the rewards. When
        a sample is requested update the Q's.

        :param model: Keras NN model.
        :param int max_memory: Number of consecutive game transitions to store.
        :param float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
    """
    def __init__(self, max_memory):
        self.memory = list()
        self.max_memory = max_memory

    def remember(self, transition):
        """ Add a game transition at the tail of the memory list.

            :param list transition: [state, move, reward, next_state, status]
        """
        self.memory.append(transition)
        if len(self.memory) > self.max_memory:
            del self.memory[0]  # forget the oldest memories

    def get_samples(self, sample_size=10):
        """ Randomly retrieve a number of observed game states and the corresponding Q target vectors.

        :param int sample_size: Number of states to return
        :return np.array: input and target vectors
        """
        mem_size = len(self.memory)  # how many episodes are currently stored (remember a matrix it's just a list of elements which are, in turn, lists)
        sample_size = min(mem_size, sample_size)  # cannot take more samples then available in memory
        states = []
        moves = []
        rewards = []
        next_states = []
        stati = []
        # update the Q's from the sample
        for i,idx in enumerate(np.random.choice(range(mem_size), sample_size, replace=False)):
            state, move, reward, next_state, status = self.memory[idx]
            states.append(state)
            moves.append(move)
            rewards.append(reward)
            next_states.append(next_state)
            stati.append(status)
        states_augm = self.state_creator(states)
        next_states_augm = self.state_creator(next_states)
        
        return states_augm, next_states_augm, rewards, moves, stati
   
    def state_creator(self,states):
        state_augm = np.zeros((len(states), self.state_size[0]))
        for i_state in range(len(states)):
            state = states[i_state]
            cols = state[0][0]
            rows = state[0][1]
            flags = dict()
            obst = dict()
            for action in actions.keys():
                for i_dist in range(1,self.maze.shape[0] + 1):
                    cell = (cols + i_dist*action[0], rows + i_dist*action[1])
                    if flags.get(action,0) == 0 and (cell in self.walls or (cell in self.cells and self.maze[cell[::-1]] == 1)):
                        obst[action] = i_dist - 1
                        flags[action] = 1
            target_delta_col = self.exit_cell[0] - cols           
            target_delta_row = self.exit_cell[1] - rows
            state_augm[i_state] = np.array([[* tuple(obst.values()) + (target_delta_col, target_delta_row)]])
        return state_augm



class QReplayAC(AbstractModel):
    """ Prediction model which uses Q-learning and a neural network which replays past moves.

        The network learns by replaying a batch of training moves. The training algorithm ensures that
        the game is started from every possible cell. Training ends after a fixed number of games, or
        earlier if a stopping criterion is reached (here: a 100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.game = game
        self.state_size = (10,)
        self.sess = tf.Session()
        K.set_session(self.sess)        
        self.learning_rate = 0.01
        self.tau  = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, 
            [None, 8]) # where we will feed de/dC (from critic)
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #		

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self):
        state_input = Input(shape=self.state_size)
        h1 = Dense(32, activation='relu')(state_input)
        h2 = Dense(64, activation='relu')(h1)
        h3 = Dense(32, activation='relu')(h2)
        output = Dense(8, activation='relu')(h3)
        
        model = Model(input=state_input, output=output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model
    
    def create_critic_model(self):
        state_input = Input(shape=self.state_size)
        state_h1 = Dense(32, activation='relu')(state_input)
        state_h2 = Dense(64)(state_h1)
        
        action_input = Input(shape=(8,))
        action_h1    = Dense(64)(action_input)
        
        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(32, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)
        
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def _train_actor(self, states, next_states, rewards, moves, stati):
        for i in range(states.shape[0]):
            cur_state = states[i]
            cur_state = np.array([cur_state])
            predicted_action = self.actor_model.predict(cur_state)
            print(cur_state, predicted_action)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_state,
                self.critic_action_input: predicted_action
            })[0]
            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })
            
    def _train_critic(self, states, next_states, rewards, moves, stati):
        for i in range(states.shape[0]):
            cur_state = states[i]
            new_state = next_states[i]
            reward = rewards[i]
            action = moves[i]
            status = stati[i]            
            if status != "win" and status != "lose":
                new_state = np.array([new_state])
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            action_index = actions.get(action)[1]
            action_vector = np.zeros((len(actions),))
            action_vector[action_index] = 1
            cur_state = np.array([cur_state])
            action_vector = np.array([action_vector])
            reward = np.array([reward])
            self.critic_model.fit([cur_state, action_vector], reward, verbose=0)

    def train_models(self):
        batch_size = 16
        if len(self.experience.memory) < batch_size:
            return

        states, next_states, rewards, moves, stati = self.experience.get_samples(self.sample_size)
        self._train_critic(states, next_states, rewards, moves, stati)
        self._train_actor(states, next_states, rewards, moves, stati)    

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        target_actor_weights = self.target_actor_model.get_weights()
        
        for i in range(len(target_actor_weights)):
            target_actor_weights[i] = actor_model_weights[i]*0.125 + (1 - 0.125)*target_actor_weights[i]
        self.target_actor_model.set_weights(target_actor_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        target_critic_weights = self.target_critic_model.get_weights()
        
        for i in range(len(target_critic_weights)):
            target_critic_weights[i] = critic_model_weights[i]*0.125 + (1 - 0.125)*target_critic_weights[i]
        self.target_critic_model.set_weights(target_critic_weights)		

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()
    

 
    def save(self, filename):
        with open(filename + ".json", "w") as outfile:
            outfile.write(self.model.to_json())
        self.model.save_weights(filename + ".h5", overwrite=True)

    def load(self, filename):
        with open(filename + ".json", "r") as infile:
            self.model = model_from_json(infile.read())
        self.model.load_weights(filename + ".h5")

    def train(self, stop_at_convergence=False, **kwargs):
        """ Hyperparameters:

            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword int episodes: number of training games to play
            :keyword int sample_size: number of samples to replay for training
            :return int, datetime: number of training episodes, total time spent
        """
        self.gamma = kwargs.get("discount", 0.90)        
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        episodes = kwargs.get("episodes", 10000)
        self.sample_size = kwargs.get("sample_size", 32)
        max_memory = kwargs.get("max_memory",1000)

        experience = ExperienceReplay(max_memory)
        self.experience = experience
        experience.maze = self.game.maze
        experience.cells = self.game.cells
        experience.exit_cell = self.game.exit
        experience.state_size = self.state_size
        experience.walls = self.game.walls

        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()  # starting cells not yet used for training
        start_time = datetime.now()

        '''cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        action = action.reshape((1, env.action_space.shape[0]))
        new_state = new_state.reshape((1, env.observation_space.shape[0]))'''

        for episode in range(1, episodes + 1):
            start = datetime.now()
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)
            state = self.environment.reset(start_cell)
            actions_counter = 0
            loss = 0.0

            while True:
                if np.random.random() < exploration_rate:
                    '''c_state  = state[0][0]
                    r_state = state[0][1]
                    r_target, c_target = self.environment.exit
                    delta_r = r_target - r_state
                    delta_c = c_target - c_state
                    delta = np.abs(delta_r) + np.abs(delta_c)
                    delta_r_percent = np.abs(delta_r)/delta*100
                    delta_c_percent = np.abs(delta_c)/delta*100
                    move = tuple(np.sign([delta_c, delta_r]))
                    actions_list = self.environment.actions.copy()
                    actions_list.remove(move)
                    if np.sum(np.abs(move)) == len(move):  #diagonal movement
                        move_c = (move[0],0)
                        move_r = (0,move[1])
                        actions_list.remove(move_c)
                        actions_list.remove(move_r)
                        if np.abs(delta_r - delta_c) < 20:
                            action_d_pool = [move]*35
                            action_r_pool = [move_r]*25
                            action_c_pool = [move_c]*25
                            actions_pool = np.concatenate((action_d_pool, action_c_pool, action_r_pool))
                            for i in range(len(actions_list)):
                                actions_pool = np.concatenate((actions_pool, [actions_list[i]]*3))
                        else:
                            action_d_pool = [move]*20
                            action_r_pool = [move_r]*int(np.round(delta_r_percent*65))
                            action_c_pool = [move_c]*int(np.round(delta_c_percent*65))
                            actions_pool = np.concatenate((action_d_pool, action_c_pool, action_r_pool))
                            for i in range(len(actions_list)):
                                actions_pool = np.concatenate((actions_pool, [actions_list[i]]*3))
                    else:
                        action_move_pool = [move]*79
                        actions_pool = action_move_pool
                        for i in range(len(actions_list)):
                            actions_pool = np.concatenate((actions_pool, [actions_list[i]]*3))
                    action = tuple(random.choice(actions_pool))
                    #print('casual action: ', action)'''
                    action = random.choice(self.environment.actions)

                else:
                    action = self.predict(state)     
                next_state, reward, status = self.environment.step(action)

                cumulative_reward += reward
                experience.remember([state, action, reward, next_state, status])

                if status in ("win", "lose"):  # terminal state reached, stop episode
                    break

                self.train_models()

                state = next_state

                self.environment.render_q(self)

                if actions_counter % 5 == 0:
                    self.update_target()
                actions_counter += 1
            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | loss: {:.4f} | e: {:.5f}"
                         .format(episode, episodes, status, loss, exploration_rate))
            

            if episode % 500 == -1:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                w_all, win_rate = self.environment.win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # explore less as training progresses
            if episode % 2 == 0:
                self.update_target()
        self.save(self.name)  # Save trained models weights and architecture

        now = datetime.now()
        time_elapsed = now - start_time
        self.time = now.timestamp() - start_time.timestamp()
        logging.info("episodes: {:d} | time spent: {}".format(episode, time_elapsed))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time


    def q(self, state):
        """ Get q values for all actions for a certain state. """
        state = np.array([state])
        state = self.experience.state_creator(state)
        return self.model.predict(state)[0]


    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: Game state.
            :return int: Chosen action.
        """
        state = np.array([state])
        state = self.experience.state_creator(state)
        actions_prob = self.actor_model.predict(state)
        actions_index = np.nonzero(actions_prob == np.max(actions_prob))[1]   
        print(actions_prob)     
        return self.environment.actions[random.choice(actions_index)]

