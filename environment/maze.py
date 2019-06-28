import logging

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from numpy.linalg import norm
from numpy import dot
import random

CELL_EMPTY = 0  #0 indicates empty cell where the agent can move to
CELL_OCCUPIED = 1  #1 indicates cell which contains a wall and cannot be entered
CELL_CURRENT = 2 #2 indicates current cell of the agent


# all actions the agent can choose, plus a dictionary for textual representation
# MOVES are 2-elements vector [col_move, row_move]: MOVE[0]: action in column direction (-): -1 left, +1 right
#MOVE[1]: action in row direction (|): -1 up, +1 down
actions_number = 8
MOVE_LEFT = (-1,0)
MOVE_RIGHT = (1,0)
MOVE_UP = (0,-1) 
MOVE_DOWN = (0,1)
MOVE_UL = (-1,-1)
MOVE_UR = (1,-1)
MOVE_DL = (-1,1)
MOVE_DR = (1,1)

actions = {
    MOVE_LEFT: ("move left",0),
    MOVE_RIGHT: ("move right",1),
    MOVE_UP: ("move up",2),
    MOVE_DOWN: ("move down",3),
    MOVE_UL: ("move up-left",4),
    MOVE_UR: ("move up-right",5),
    MOVE_DL: ("move down-left",6),
    MOVE_DR: ("move down-right",7)
}


class Maze:
    """ A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.

        The layout of the maze and the rules how to move through it are called the environment. An agent is placed
        at start_cell. The agent chooses actions (move left/right/up/down) in order to reach the exit_cell. Every
        action results in a reward or penalty which are accumulated during the game. Every move gives a small
        penalty (-0.05), returning to a cell the agent visited earlier a bigger penalty(-0.25) and running into
        a wall a large penalty (-0.75). The reward (+2.0) is collected when the agent reaches the exit. The
        game always reaches a terminal state; the agent either wins or looses. Obviously reaching the exit means
        winning, but if the penalties the agent is collecting during play exceed a certain threshold the agent is
        assumed to wander around cluelessly and looses.

        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlibs plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """

    def __init__(self, maze,  close_reward, start_cell=(0, 0), exit_cell=None):
        """ Create a new maze with a specific start- and exit-cell.

            :param numpy.array maze: 2D Array containing empty cells (=0) and cells occupied with walls (=1).
            :param tuple start_cell: Starting cell for the agent in the maze (optional, else upper left).
            :param tuple exit_cell: Exit cell which the agent has to reach (optional, else lower right).
        """
        self.maze = maze
        self.__minimum_reward = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold
        self.close_reward = close_reward
        self.old_action = (0,0)
        self.angle_target = 0
        self.delta_r = maze.shape[1]
        self.delta_c = maze.shape[0]


        self.actions = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, MOVE_UL, MOVE_UR, MOVE_DL, MOVE_DR]

        nrows, ncols = self.maze.shape
        exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell

        self.exit = exit_cell
        self.exit_cell = exit_cell
        self.start_cell = start_cell
        self.__exit_cell = exit_cell
        self.__previous_cell = self.__current_cell = start_cell
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == CELL_EMPTY]
        ##################################################
        self.__close = set()
        for col in range(ncols):
            for row in range(nrows):
                if self.maze[row, col] == CELL_OCCUPIED:
                    if (col+1,row) in self.empty:
                        self.__close.add((col+1, row))
                    if (col-1,row) in self.empty:
                        self.__close.add((col-1, row))
                    if (col,row+1) in self.empty:
                        self.__close.add((col, row+1))
                    if (col,row-1) in self.empty:
                        self.__close.add((col, row-1))
        #print(self.__close)
        ############################################################
        self.walls = []
        for i in range(-1,ncols + 1):
            for j in [-1,nrows]:
                self.walls.append((i,j))
        for i in range(-1,nrows + 1):
            for j in [-1,ncols]:
                self.walls.append((j,i))
        self.walls = list(set(self.walls))
        ################################################################
        for cell in self.empty:
            col = cell[0]
            row = cell[1]
            flag = 0
            for action in self.actions:
                if (col + action[0], row + action[1]) in self.empty:
                    flag = 1
            if flag == 0:
                self.empty.remove(cell)
        ################################################################################           
                
       
            
        if exit_cell not in self.empty:
            raise Exception("Error: exit cell at {} is not inside maze".format(exit_cell))
        else:
            self.empty.remove(exit_cell)

        if exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(exit_cell))
        if self.maze[exit_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(exit_cell))

        self.__render = "nothing"
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell

        self.i = 0

        self.reset(start_cell)

    def reset(self, start_cell=(0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.

            :param tuple start_cell: Here the agent starts its journey through the maze (optional, else upper left).
            :return: New state after reset.
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == CELL_OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        if self.__render in ("training", "moves"):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=25)  # start is a big red square
            self.__ax1.plot(*self.__exit_cell, "gs", markersize=25)
            # exit is a big green square
            #maze_plot = np.dot(self.maze_plot,1/2) #used with biased
            maze_plot = np.subtract(np.ones(self.maze.shape), self.maze) #used with unbiased
            self.__ax1.imshow(maze_plot, cmap="gray")
            # plt.pause(0.001)  # replaced by the two lines below
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __draw(self):
        """ Draw a line from the agents previous to its current cell. """
        #print(self.__current_cell)
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell is a red dot
        # plt.pause(0.001)  # replaced by the two lines below
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()
        plt.savefig('immagini/%08d.png' % self.i) #this line save the gif(time consuming)
        self.i += 1

    def render(self, content="nothing"):
        """ Define what will be rendered during play and/or training.

            :param str content: "nothing", "training" (moves and q), "moves"
        """
        if content not in ("nothing", "training", "moves"):
            raise ValueError("unexpected content: {}".format(content))

        self.__render = content
        if content == "nothing":
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if content == "training":
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if content in ("moves", "training"):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param int action: The agent will move in this direction.
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        #logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(actions[action], reward, status))
        return state, reward, status

    def __execute(self, action):
        """ Execute action and collect the reward or penalty.

            :param int action: The agent will move in this direction.
            :return float: Reward or penalty after the action is done.
        """
        possible_actions = self.__possible_actions(self.__current_cell)
        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row = self.__current_cell
            target_versor = (self.__exit_cell[0] - col, self.__exit_cell[1] - row)
            col += action[0]
            row += action[1]

            target_versor = target_versor/norm(target_versor)
            self.angle_target = dot(action/norm(action), target_versor)
 
            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__render != "nothing":
                self.__draw()

###############################################################################################################################
            '''if self.__current_cell == self.__exit_cell:
                reward = 2.0  # maximum reward for reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = -0.25  # penalty for returning to a cell which was visited earlier
                ##############################################################
                if self.__current_cell in self.__close:
                    reward = self.close_reward
                ###########################################################
            else:
                reward = -0.1 # penalty for a move which did not result in finding the exit cell
                ###########################################################
                if self.__current_cell in self.__close:
                    reward = self.close_reward
                ##########################################################
            self.__visited.add(self.__current_cell)
            if self.old_action not in [(0,0)]:
                angle_action = np.arccos(dot(self.old_action/norm(self.old_action), action/norm(action)))
                if angle_action >= np.pi/2:
                    reward = -0.25
            if np.abs(self.old_action[0] - action[0]) == 2 or np.abs(self.old_action[1] - action[1]) == 2 or \
                np.abs(self.old_action[0] - action[0]) == 1 and np.abs(self.old_action[1] - action[1]) == 1 and self.old_action not in [(0,0)]:
                reward = -0.25'''
####################################################################################################################
            if self.__current_cell == self.__exit_cell:
                reward = 2.0  # maximum reward for reaching the exit cell
                '''elif self.__current_cell in self.__visited:
                reward = -0.25'''
            else:
                reward = -1/20*np.exp(np.arccos(self.angle_target)/np.pi) 
            if self.old_action not in [(0,0)]:
                angle_action = np.arccos(dot(self.old_action/norm(self.old_action), action/norm(action)))
                if angle_action >= np.pi/2:
                    reward -= 0.3
            '''if self.old_old_action not in [(0,0)]:
                angle_old_action = np.arccos(dot(self.old_old_action/norm(self.old_old_action), action/norm(action)))
                if angle_old_action >= np.pi/2:
                    reward -= 0.15'''
            if self.__current_cell in self.__close:
                reward += self.close_reward
########################################################################################################
        else:
            reward = -0.75  # penalty for trying to enter an occupied cell (= a wall) or moving out of the maze
        self.old_action = action
        return reward

    def __possible_actions(self, cell=None):
        """ Create a list with possible actions, avoiding the maze's edges and walls.

            :param tuple cell: Location of the agent (optional, else current cell).
            :return list: All possible actions.
        """
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell
        
        possible_actions = self.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_UP)
  
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_DOWN)       

        if col == 0 or (col > 0 and self.maze[row, col - 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_LEFT)  

        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_RIGHT)

        #############################################################################################    
        if row == 0 or col == 0 or (row > 0 and  col> 0 and self.maze[row - 1, col - 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_UL)
        
        if row == 0 or col == ncols - 1 or (row > 0 and col < ncols - 1 and self.maze[row - 1, col + 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_UR)
        
        if row == nrows - 1 or col == 0 or (row < nrows - 1 and col > 0 and self.maze[row + 1, col - 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_DL)
        
        if row == nrows - 1 or col == ncols - 1 or (row < nrows - 1 and col < ncols - 1 and self.maze[row + 1, col + 1] == CELL_OCCUPIED):
            possible_actions.remove(MOVE_DR)
        
        return possible_actions

    def __status(self):
        """ Determine the game status.

            :return str: Current game status (win/lose/playing).
        """
        if self.__current_cell == self.__exit_cell:
            return "win"

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            return "lose"

        return "playing"

    def __observe(self):
        """ Return the state of the maze - in this example the agents current location.

            :return numpy.array [1][2]: Agents current location.
        """
        return np.array([[*self.__current_cell]])

    def play(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: The prediction model to use.
            :param tuple start_cell: Agents initial cell (optional, else upper left).
            :return str: "win" or "lose"
        """
        self.reset(start_cell)
        actions_counter = 0
        state = self.__observe()
        trajectory = [(state[0][0],state[0][1])]
        t_start = datetime.now()
        if model.name == "NN double augmented prior x,y state +  action":
            old_action = (0,0)
            while True:
                action,action_index = model.predict(state, old_action)
                state, reward, status = self.step(action)
                old_action = action
                if status in ("win", "lose"):
                    return status  
        else:    
            while True:
                action,action_index = model.predict(state=state)
                state, reward, status = self.step(action)
                trajectory.append((state[0][0],state[0][1]))
        
                if status in ("win", "lose") or actions_counter > 35:
                    t_finish = datetime.now()
                    time_elapsed = t_finish.timestamp() - t_start.timestamp()
                    return status, trajectory, time_elapsed
                
                actions_counter += 1

    def win_all(self, model):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = "nothing"  # avoid rendering anything during execution of win_all()

   
        win = 0
        lose = 0
   


        #for cell in [(0,0)]:
        for cell in self.empty:
            status = self.play(model,cell)
           
            
            if status == "win":
                win += 1
            else:
                lose += 1
            

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

        self.__render = previous

        result = True if lose == 0 else False
        return result, win / (win + lose)

    def render_q(self, model):
        """ Render the recommended action for each cell. """
        if self.__render != "training":
            return

        nrows, ncols = self.maze.shape

        self.__ax2.clear()
        self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
        self.__ax2.set_xticklabels([])
        self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
        self.__ax2.set_yticklabels([])
        self.__ax2.grid(True)
        self.__ax2.plot(*self.__exit_cell, "gs", markersize=25)  # exit is a big green square

        for cell in self.empty:
            state = cell
            q = model.q(state) if model is not None else [0, 0, 0, 0]
            a = np.nonzero(q == np.max(q))[0]

            for action in a:
                dx = 0
                dy = 0
                if action == 0:  # left
                    dx = -0.2
                if action == 1:  # right
                    dx = +0.2
                if action == 2:  # up
                    dy = -0.2
                if action == 3:  # down
                    dy = 0.2
                if action == 4:  # up-left
                    dx = -0.2
                    dy = -0.2
                if action == 5:  # up-right
                    dx = +0.2
                    dy = -0.2
                if action == 6:  # down-left
                    dx = -0.2
                    dy = +0.2
                if action == 7:  #d down-right
                    dx = +0.2
                    dy = +0.2

                self.__ax2.arrow(*cell, dx, dy, head_width=0.2, head_length=0.1)

        self.__ax2.imshow(self.maze, cmap="binary")
        self.__ax2.get_figure().canvas.draw()
        # plt.pause(0.001)

    def play_final(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: The prediction model to use.
            :param tuple start_cell: Agents initial cell (optional, else upper left).
            :return str: "win" or "lose"
        """
        self.reset(start_cell)

        state = self.__observe()
        action_counter = 0

        ############################
        close_counter = 0
        ###########################
        while True:
            action, action_index = model.predict(state=state)
            state, reward, status = self.step(action)
            action_counter += 1
            #############################
            state_tuple = (state[0][0], state[0][1]) #state is given by __observe in [[np.array]] form (why?)
            if state_tuple in self.__close:
                close_counter += 1
            #################################
            if status in ("win", "lose"):
                return status, action_counter, close_counter

    def win_all_final(self, model):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = "nothing"  # avoid rendering anything during execution of win_all()

   
        win = 0
        lose = 0
        actions_counter_list = []
        close_counter_list = []


        status, actions_counter, close_counter = self.play_final(model,self.start_cell)

        if status == "win":
            close_counter_list.append(close_counter)                
            actions_counter_list.append(actions_counter) 
        
        if status == "win":
            win += 1
        else:
            lose += 1
        

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))


        result = True if lose == 0 else False
        return np.mean(actions_counter_list), np.sum(close_counter_list), model.time, lose
 
    
