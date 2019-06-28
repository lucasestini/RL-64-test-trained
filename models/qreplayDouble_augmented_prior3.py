import logging
import random
from datetime import datetime

import numpy as np

np.random.seed(1)
import tensorflow
 
tensorflow.set_random_seed(2)

from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json 

from environment.maze import actions
from models import AbstractModel
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from .utils.memory_buffer import MemoryBuffer  

 
class ExperienceReplay:
    """ Store game transitions (from state s to s' via action a) and record the rewards. When
        a sample is requested update the Q's.

        :param model: Keras NN model.
        :param int max_memory: Number of consecutive game transitions to store.
        :param float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
    """

    def __init__(self, model, target_model, max_memory=1000, discount=0.95):
        self.model = model
        self.target_model = target_model
        self.discount = discount
        self.memory = list()
        self.td = []
        self.max_memory = max_memory
        self.tau = .125
        self.with_per = True
        self.buffer = MemoryBuffer(self.max_memory, self.with_per)
    


    def remember(self, state, action, reward, next_state, status):
        """ Add a game transition at the tail of the memory list.

            :param list transition: [state, move, reward, next_state, status]
        """

        q = self.predict(state)
        q_target = self.predict_target(next_state)
        next_action_best = random.choice(np.nonzero(q == np.max(q))[0])  # get index of the action(s) with the max value
        td_error = abs(reward + self.discount * q_target[next_action_best]- q[action]) 
        self.buffer.memorize(state, action, reward, status, next_state, [td_error])


    def predict(self, state):
        """ Predict the Q vector belonging to this state.

            :param np.array state: Game state.
            :return np.array: Array with Q's per action.
        """
        return self.model.predict(state)[0]  # prediction is a [1][num_actions] array with Q's
    
    def predict_target(self,state):
        return self.target_model.predict(state)[0]

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def get_samples(self, sample_size=10):
        """ Randomly retrieve a number of observed game states and the corresponding Q target vectors.

        :param int sample_size: Number of states to return
        :return np.array: input and target vectors
        """
        num_actions = self.model.output_shape[-1]  # number of actions in output layer
        q = np.zeros((sample_size, num_actions), dtype=float)
        states, moves, rewards, stati, next_states, idx = self.buffer.sample_batch(sample_size)
        # update the Q's from the sample
        for i in range(len(states)):
            q[i] = self.predict(states[i])
            q_old = q[i][moves[i]]
            q_target = self.predict_target(next_states[i])
            if stati[i] == "win":
                q[i, moves[i]] = rewards[i]  # no discount needed if a terminal state was reached.
            else:
                next_action_best = random.choice(np.nonzero(q[i] == np.max(q[i]))[0])  # get index of the action(s) with the max value
                q[i, moves[i]] = rewards[i] + self.discount * np.max(q_target[next_action_best])
                if(self.with_per):
                    # Update PER Sum Tree
                    self.buffer.update(idx[i], abs(q_old - q[i, next_action_best]))
        return states.reshape(states.shape[0], states.shape[2]), q
    
    def state_creator(self,state):
        state_augm = np.zeros((1, self.state_size[0]))
        cols = state[0][0]
        rows = state[0][1]
        flags = dict()
        obst = dict()
        delta_x = self.exit_cell[0] - cols
        delta_y = self.exit_cell[1] - rows
        move = tuple(np.sign([delta_x, delta_y]))
        if (cols,rows) not in [self.exit_cell]:
            if np.sum(np.abs(move)) == 2:
                move_l = (move[0],0)
                move_r = (0,move[0])
            elif np.sum(np.abs(move)) == 1:
                if move[0] == 0:
                            move_l = (-1,move[1])
                            move_r = (1,move[1])
                if move[1] == 0:
                        move_l = (move[0],-1)
                        move_r = (move[0],1)                         

            ray_moves = [move_l, move, move_r]
            
            for action in ray_moves:
                for i_dist in range(1,self.maze.shape[0] + 1):
                    cell = (cols + i_dist*action[0], rows + i_dist*action[1])
                    if flags.get(action,0) == 0 and (cell in self.walls or (cell in self.cells and self.maze[cell[::-1]] == 1)):
                        obst[action] = i_dist - 1
                        flags[action] = 1
            target_delta_col = self.exit_cell[0] - cols           
            target_delta_row = self.exit_cell[1] - rows
            state_augm = np.array([[* tuple(obst.values()) + (target_delta_col, target_delta_row)]])
        else:
            state_augm = np.array([[*(0,0,0,0,0)]])
        return state_augm


class QReplayDoubleAugmPrior3(AbstractModel):
    """ Prediction model which uses Q-learning and a neural network which replays past moves.

        The network learns by replaying a batch of training moves. The training algorithm ensures that
        the game is started from every possible cell. Training ends after a fixed number of games, or
        earlier if a stopping criterion is reached (here: a 100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.game = game
        self.state_size = (3 + 2,)
 
        if kwargs.get("load", False) is False:
            self.model = Sequential()
            self.model.add(Dense(game.maze.size, input_shape=self.state_size, activation="relu"))
            self.model.add(Dense(game.maze.size, activation="relu"))
            self.model.add(Dense(len(actions)))
        else:
            self.load(self.name)

        self.model.compile(optimizer="adam", loss="mse")

        self.target_model = Sequential()
        self.target_model.add(Dense(game.maze.size, input_shape=self.state_size, activation="relu"))
        self.target_model.add(Dense(game.maze.size, activation="relu"))
        self.target_model.add(Dense(len(actions)))
        self.target_model.compile(optimizer="adam", loss="mse")

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
        max_memory = kwargs.get("max_memory", 1000)
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        episodes = kwargs.get("episodes", 10000)
        batch_size = kwargs.get("sample_size", 32)
        experience = ExperienceReplay(self.model, self.target_model, discount=discount, max_memory=max_memory)
        self.experience = experience
        experience.maze = self.game.maze
        experience.cells = self.game.cells
        experience.exit_cell = self.game.exit_cell
        experience.state_size = self.state_size
        experience.walls = self.game.walls

        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()  # starting cells not yet used for training
        start_time = datetime.now()


        for episode in range(1, episodes + 1):
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)
            state = self.environment.reset(start_cell)
            actions_counter = 0
            loss = 0.0
            self.game.old_action = (0,0)

            while True:
                if np.random.random() < exploration_rate:
                    c_state  = state[0][0]
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
                    
                    #action = random.choice(self.environment.actions)
                    action_index = [actions[action][1]]
                    
                
                else:
                    # q = experience.predict(state)
                    # action = random.choice(np.nonzero(q == np.max(q))[0])
                    action,action_index = self.predict(state)

                next_state, reward, status = self.environment.step(action)
                state_augm = experience.state_creator(state)
                next_state_augm = experience.state_creator(next_state) 
                cumulative_reward += reward
                experience.remember(state_augm, action_index, reward, next_state_augm, status)
                if status in ("win", "lose"):  # terminal state reached, stop episode
                    break
                if experience.buffer.size() > 2*batch_size:
                    inputs, targets = experience.get_samples(sample_size=batch_size)
                    self.model.fit(inputs, targets, epochs=1, batch_size=16, verbose=0)
                    if actions_counter % 5 == 0:
                        experience.target_train()
                    actions_counter += 1
                    loss += self.model.evaluate(inputs, targets, verbose=0)
                state = next_state

                self.environment.render_q(self)


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
        self.save(self.name)  # Save trained models weights and architecture

        now = datetime.now()
        time_elapsed = now - start_time
        self.time = now.timestamp() - start_time.timestamp()
        logging.info("episodes: {:d} | time spent: {}".format(episode, time_elapsed))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time


    def q(self, state):
        """ Get q values for all actions for a certain state. """
        state = self.experience.state_creator(state)
        return self.model.predict(state)[0]

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: Game state.
            :return int: Chosen action.
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions_index = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return self.environment.actions[random.choice(actions_index)],actions_index

