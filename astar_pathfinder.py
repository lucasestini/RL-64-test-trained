import astar_algorithm
import numpy as np
import datetime
import os


map = '9647_2'  #inserire nome mappa eg. 0866_1
entrylist = [(38,1),  (18,3),  (31,0),  (38,5),  (45,9),  (18,5),  (45,11), (24,14), (34,1),  (29,16)] #inserire la lista dei punti di ingresso
exitpoint = (27,10)
count = 0
time = []

map_npy = 'mappe_test_152/' + 'map_' + map + '.npy'
maze = np.load(map_npy)

for entry in entrylist:
    print('-------------------')
    print('Path a* ' + str(count) + ': entry = ' + str(entry))
    t_start = datetime.datetime.now()
    path = astar_algorithm.astar(maze, (entry[1], entry[0]), (exitpoint[1], exitpoint[0]))
    t_end = datetime.datetime.now()
    elapsed_time = t_end.timestamp() - t_start.timestamp()
    print(elapsed_time)
    time.append(elapsed_time)
    print(path)
    path = np.array(path)
    pathfile_name = 'path/astar/' + map + '_path_' + str(count) + '.npy'
    np.save(pathfile_name, path)
    print(' Salvato file path')
    count = count+1


time = np.array(time)
timefile_name = 'path/astar/' + map + '_time.npy'
np.save(timefile_name, time)
print('Salvato file tempi')
