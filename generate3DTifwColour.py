import numpy as np
import random
from scipy.ndimage.measurements import label
import tifffile

zipFile = np.load('./ImageStackFULL.npz')
labelledOutputs = zipFile['labelledOut'][235:, 0:1200]
print(labelledOutputs.shape)

maxN = np.amax(labelledOutputs)
print(maxN)
print(labelledOutputs.dtype)

labelledOutputs = np.asarray(labelledOutputs, dtype='uint16')
maxN = np.amax(labelledOutputs)
print(maxN)

rgbOutputs = np.zeros(shape = (labelledOutputs.shape[0], labelledOutputs.shape[1], labelledOutputs.shape[2], 3), dtype=np.uint8)

def randomColour():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    rgb = np.asarray([r, g, b])
    return rgb

colourList = []
for i in range(maxN):
    colour = randomColour()
    colourList.append(colour) 

for i in range(1, maxN+1):
    print("replacing component", i)
    pos = np.where(labelledOutputs == i)
    print(colourList[i-1])
    print(colourList[i-1].dtype)
    for j in range(len(pos[0])):
        rgbOutputs[pos[0][j], pos[1][j], pos[2][j], 0] = colourList[i-1][0]
        rgbOutputs[pos[0][j], pos[1][j], pos[2][j], 1] = colourList[i-1][1]
        rgbOutputs[pos[0][j], pos[1][j], pos[2][j], 2] = colourList[i-1][2]
        

tifffile.imwrite('ImageStackNOTFULLFORMEASUREMENTS.tif', rgbOutputs, photometric='rgb')