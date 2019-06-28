import numpy as np
import matplotlib.pyplot as plt
def plotmap():
    for i in range(0,250,8):
        print(i)
        maze = np.load('mappe_64_trans/map_' + str(i) + '.npy')
        #maze = np.load('mappe_64_front/map_' + str(i) + '.npy')
        puntox = 3
        puntoy = 3

        puntox2 = 7
        puntoy2 = 3
        plt.grid(True)
        plt.plot(puntox, puntoy, "rs", markersize=5)  # start is a big red square
        plt.plot(puntox2, puntoy2, "gs", markersize=5)  # exit is a big green square
        title = 'map '+str(i)
        plt.title(title)
        plt.imshow(maze, cmap="binary")
        plt.show()
        save = input('type True to save')
        if save == "True":
                y_half = int(input('type y cut '))
                maze_up = maze[0:y_half,:]
                maze_down = maze[y_half:,:]
                np.save('mappe_test/map_0866_'+str(i)+'_1',maze_up)
                np.save('mappe_test/map_0866_'+str(i)+'_2',maze_down)




if __name__ == "__main__": 
    plotmap()