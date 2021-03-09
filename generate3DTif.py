import numpy as np
import random
import tifffile

zipFile = np.load('./ImageStackTest.npz')
output = zipFile['blurThresh']

tifffile.imwrite('ImageStackTestBlurThresh.tif', output)