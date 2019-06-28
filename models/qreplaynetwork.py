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


class ExperienceReplay:
    """ Store game transitions (from state s to s' via action a) and record the rewards. When
        a sample is requested update the Q's.

        :param model: Keras NN model.
        :param int max_memory: Number of consecutive game transitions to store.
        :param float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
    """

    def __init__(self, model, max_memory=200, discount=0.95):
        self.model = model
        self.discount = discount
        self.memory = list()
        self.max_memory = max_memory

    def remember(self, transition):
        """ Add a game transition at the tail of the memory list.

            :param list transition: [state, move, reward, next_state, status]
        """
        self.memory.append(transition)
        if len(self.memory) > self.max_memory:
            del self.memory[0]  # forget the oldest memories

    def predict(self, state):
        """ Predict the Q vector belonging to this state.

            :param np.array state: Game state.
            :return np.array: Array with Q's per action.
        """
        return self.model.predict(state)[0]  # prediction is a [1][num_actions] array with Q's

    def get_samples(self, sample_size=10):
        """ Randomly retrieve a number of observed game states and the corresponding Q target vectors.

        :param int sample_size: Number of states to return
        :return np.array: input and target vectors
        """
        mem_size = len(self.memory)  # how many episodes are currently stored (remember a matrix it's just a list of elements which are, in turn, lists)
        sample_size = min(mem_size, sample_size)  # cannot take more samples then available in memory
        state_size = self.memory[0][0].size
        num_actions = self.model.output_shape[-1]  # number of actions in output layer

        states = np.zeros((sample_size, state_size), dtype=int)
        targets = np.zeros((sample_size, num_actions), dtype=float)

        # update the Q's from the sample
        for i, idx in enumerate(np.random.choice(range(mem_size), sample_size, replace=False)):
            state, move, reward, next_state, status = self.memory[idx]

            states[i] = state
            targets[i] = self.predict(state)

            if status == "win":
                targets[i, move] = reward  # no discount needed if a terminal state was reached.
            else:
                targets[i, move] = reward + self.discount * np.max(self.predict(next_state))


        return states, targets


class QReplayNetworkModel(AbstractModel):
    """ Prediction model which uses Q-learning and a neural network which replays past moves.

        The network learns by replaying a batch of training moves. The training algorithm ensures that
        the game is started from every possible cell. Training ends after a fixed number of games, or
        earlier if a stopping criterion is reached (here: a 100% win rate).

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)

        if kwargs.get("load", False) is False:
            self.model = Sequential()
            self.model.add(Dense(game.maze.size, input_shape=(2,), activation="relu"))
            self.model.add(Dense(game.maze.size, activation="relu"))
            self.model.add(Dense(len(actions)))
        else:
            self.load(self.name)

        self.model.compile(optimizer="adam", loss="mse")

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
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        episodes = kwargs.get("episodes", 10000)
        sample_size = kwargs.get("sample_size", 100)

        experience = ExperienceReplay(self.model, discount=discount)
        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()  # starting cells not yet used for training
        start_time = datetime.now()

        t_fitting_epochs = []
        t_get_samples_epochs = []

        for episode in range(1, episodes + 1):
            self.environment.old_action = (0,0)
            exploration_rate = 0.8*np.exp(-episode/(episodes/9)) + 0.001

            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)

            loss = 0.0
            actions_count = 1
            #t_fitting = []
            #t_get_samples = []
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
                    #print('casual action: ', action)

                else:
                    # q = experience.predict(state)
                    # action = random.choice(np.nonzero(q == np.max(q))[0])
                    action = self.predict(state)     
                    #print('predict action: ', action)

                next_state, reward, status = self.environment.step(action)

                cumulative_reward += reward

                experience.remember([state, action, reward, next_state, status])

                if status in ("win", "lose"):  # terminal state reached, stop episode
                    break
                #t_samples = datetime.now().timestamp()
                if actions_count % 3 == 0:
                    inputs, targets = experience.get_samples(sample_size=sample_size)
                    #t_get_samples.append(datetime.now().timestamp()-t_samples)

                    #t_fit = datetime.now().timestamp()
                    self.model.fit(inputs,
                               targets,
                               epochs=4,
                               batch_size=16,
                               verbose=0)
                    loss += self.model.evaluate(inputs, targets, verbose=0)
                #t_fitting.append(datetime.now().timestamp()-t_fit)


                state = next_state

                self.environment.render_q(self)

                actions_count += 1
            #t_fitting_epochs.append(np.sum(t_fitting))
            #t_get_samples_epochs.append(np.sum(t_get_samples))
            if status in "lose":
                for i in range(actions_count):
                    del experience.memory[-1]
            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | loss: {:.4f} | e: {:.5f}"
                         .format(episode, episodes, status, loss, exploration_rate))

            if episode % 100 == -1:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                w_all, win_rate = self.environment.win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            #exploration_rate *= exploration_decay  # explore less as training progresses

        self.save(self.name)  # Save trained models weights and architecture

        now = datetime.now()
        time_elapsed = now - start_time
        self.time = now.timestamp() - start_time.timestamp()
        logging.info("episodes: {:d} | time spent: {}".format(episode, time_elapsed))
        
        #self.t_fitting_epochs = np.sum(t_fitting)
        #self.t_get_samples_epochs = np.sum(t_get_samples)
        return cumulative_reward_history, win_history, t_fitting_epochs, t_get_samples_epochs

    def q(self, state):
        """ Get q values for all actions for a certain state. """
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
        return self.environment.actions[random.choice(actions_index)]
