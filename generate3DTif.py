# generates 3D tif with no colour
import numpy as np
import tifffile

zipFile = np.load('./ImageStackTest.npz')
output = zipFile['labelledOut']

tifffile.imwrite('ImageStackTestFinal.tif', output)