import numpy as np
import random
import tifffile

zipFile = np.load('./ImageStackTest.npz')
output = zipFile['labelledOut']

tifffile.imwrite('ImageStackTestFinal.tif', output)