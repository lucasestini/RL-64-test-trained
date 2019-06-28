import matplotlib.pyplot as plt
from environment import Maze
from models import *
from plotMap import plotmap
import pickle
from play import play

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S")




patient_number = '0866'
map_index = str(1)
map_name = 'mappe_test_152/map_'+patient_number+'_'+map_index
maze = np.load(map_name+'.npy')
plotmap(map_name+'.npy')
load = False




if __name__ == '__main__':
        
    start_cell = tuple(int(x) for x in input('start cell: ').split())
    exit_cell = tuple(int(x) for x in input('exit cell: ').split())
    game = Maze(maze, start_cell=start_cell, exit_cell=exit_cell, close_reward = -0.5)

    if 0:  # train using a neural network with experience replay (also saves the resulting model)
        model = QReplayDoubleAugmPrior4(game, name = "NN double augm prior 4 rays")
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.60, episodes=700, max_memory=maze.size * 4)

    if 1:  # train using a neural network with experience replay (also saves the resulting model)
        game.render("moves")# to see the traning--> time consuming
        model_name = "NN double augm prior 8 rays +  delta location "+ patient_number +'_'+map_index
        model = QReplayDoubleAugmPrior8(game, name = model_name, load = False)  
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.90, episodes=2200, max_memory=maze.size * 4) 

    if 0:  # train using a neural network with experience replay (also saves the resulting model)
        model = QReplayDoubleAugmPrior3(game, name = "NN double augm prior 3 rays +  delta location")  
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.60, episodes=300, max_memory=maze.size * 4)

    if 0:  # train using a neural network with experience replay (also saves the resulting model)
        model = QReplayDoubleAugmPrior5(game, name = "NN double augm prior 5 rays (including 2 semidiagonal) +  delta location")  
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.60, episodes=300, max_memory=maze.size * 4) 

    if 0:  # train using a neural network with experience replay (also saves the resulting model)
        model = QReplayDoubleAugmPrior5fan(game, name = "NN double augm prior 5 rays (fan) +  delta location")  
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.60, episodes=300, max_memory=maze.size * 4, load = load) 

    if 0:  # train using a neural network with experience replay (also saves the resulting model)
        model = QReplayDoublePrior(game, name = "NN double prior x,y state "+ patient_number +'_'+map_number, load = load)  
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.60, episodes=800, max_memory=maze.size * 4) 
    if 0:  # train using a neural network with experience replay (also saves the resulting model)
        model = QReplayDoubleActionPrior(game, name = "NN double augmented prior x,y state +  action")  
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.60, episodes=500, max_memory=maze.size * 4) 
    try:
        h  # force a NameError exception if h does not exist (and thus don't try to show win rate and cumulative reward)
        fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
        fig.canvas.set_window_title(model.name)
        ax1.plot(*zip(*w))
        ax1.set_xlabel("episode")
        ax1.set_ylabel("win rate")
        ax2.plot(h)
        ax2.set_xlabel("episode")
        ax2.set_ylabel("cumulative reward")
        plt.show()
    except NameError:
        pass
    plt.grid(True)
    plt.imshow(maze, cmap="binary")
    plt.show()
    game.render("moves")
    game.play(model, start_cell = start_cell)

    #load = False
    actions_counter, close_counter, time, lost = game.win_all_final(model)


    plt.savefig('risultato.png')

    logging.info('Mean length of path {}, time: {}, with modality {} | lost {}'.format(actions_counter, time, model.name, lost))

    logging.info('# of close-to-obstacles states visited in all games: {} with modality {}'.format(close_counter, model.name))


    plt.show()  # must be placed here else the image disappears immediately at the end of the program

