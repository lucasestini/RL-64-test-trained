import numpy as np
import matplotlib.pyplot as plt
def plotmap(map_name):
        matrice = np.load(map_name)
        puntox = 3
        puntoy = 3

        puntox2 = 7
        puntoy2 = 3
        plt.grid(True)
        #plt.plot(puntox, puntoy, "rs", markersize=5)  # start is a big red square
        #plt.plot(puntox2, puntoy2, "gs", markersize=5)  # exit is a big green square
        plt.title(map_name)
        plt.imshow(matrice, cmap="binary")
        plt.show()

if __name__ == "__main__": 
        mapname = input('insert map name: ')
        plotmap(mapname)