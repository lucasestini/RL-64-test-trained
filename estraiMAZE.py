import nibabel as ni
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import *

img = ni.load('/Users/alicesegato/Desktop/DistanceMap_Obstacles.nii')

data = img.get_data()

print('misura 1', int(img.shape[0]))
print('misura 2', int(img.shape[1]))
print('misura 3', int(img.shape[2]))

final_dim = 8 #64
initial_dim = int(img.shape[0])

# salvo un file per ogni k della distance map
for k in range(0, int(img.shape[2]), 8):
    mat = np.zeros((initial_dim, initial_dim), dtype=np.int32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if data[i][j][k] > 10:
                nulla = 1
            else:
                mat[i, j] = 1  # metto ostacolo
                if (i == 5 and j == 5) or (i == 39 and j == 26):
                    print
                    'oddio, {}'.format(k)

    mat = zoom(mat, float(final_dim) / float(initial_dim))
    print('Dimensione 64x64 giusto?', mat.shape)
    np.save('mappe/map_{}.npy'.format(k), mat)