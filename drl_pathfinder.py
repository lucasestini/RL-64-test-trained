import numpy as np
from plotMap import plotmap
from keras.models import model_from_json
from environment import Maze
from models import *
import pickle




def play(map_index, start_cell, exit_cell):
    map_npy = 'mappe_test_152/map_'+map_index+'.npy'
    #plt.grid(True)
    maze = np.load(map_npy)
    #exit_cell = (35,30)
    model_name = 'NN double augm prior 8 rays +  delta location '+map_index
    #while True:

    #plt.imshow(maze, cmap="binary")
    #plt.plot(exit_cell[0], exit_cell[1], "gs", markersize=5)  # exit is a big green square
    #plt.title(map_npy)
    #plt.show()
    #start_cell = tuple(int(x) for x in input('start cell: ').split())
    game = Maze(maze, start_cell=start_cell, exit_cell=exit_cell, close_reward = -0.5)
    model = QReplayDoubleAugmPrior8(game, name = model_name, load = True)
    status, trajectory, time_elapsed = game.play(model, start_cell = start_cell)
    #game.render("moves")
    game.play(model, start_cell = start_cell)
    #print('*******************************************')
    #print('status = {}'.format(status))
    #print('trajectory = {}'.format(trajectory))
    #print('time elapsed = {} seconds'.format(time_elapsed))
    #repeat = input('Type True to repeat: ')
    #if repeat != "True":
    return trajectory, time_elapsed




map_index = '3352_1'  #inserire nome mappa eg. 0866_1
entrylist = [(22,30), (23,28), (22,26), (23,23), (25,25), (24,22), (25,22), (37,17), (30,22), (31,21)] #inserire la lista dei punti di ingresso
exitpoint = (35,29)
count = 0
time = []

for entry in entrylist:
    print('-----------------')
    print('Path drl ' + str(count) + ': entry = ' + str(entry))
    path, el_time = play(map_index, entry, exitpoint)
    print(el_time)
    time.append(el_time)
    for index, pos in enumerate(path):
        path[index] = (pos[1], pos[0])
    print(path)
    path = np.array(path)
    pathfile_name = 'path/drl/' + map_index + '_path_' + str(count) + '.npy'
    np.save(pathfile_name, path)
    print(' Salvato file path')
    count = count + 1

time = np.array(time)
timefile_name = 'path/drl/' + map_index + '_time.npy'
np.save(timefile_name, time)
print('Salvato file tempi')

