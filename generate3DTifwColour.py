import numpy as np
import random
import tifffile

zipFile = np.load('./ImageStackBinaryComponentsBigFused.npz')
labelledOutputs = zipFile['labelledOut'][150:200]

maxN = np.amax(labelledOutputs)
print(maxN)
print(labelledOutputs.dtype)

labelledOutputs = np.asarray(labelledOutputs, dtype='uint8')
rgbOutputs = np.zeros(shape = (labelledOutputs.shape[0], labelledOutputs.shape[1], labelledOutputs.shape[2]), dtype=np.ndarray)

def randomColour():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    rgb = np.asarray([r, g, b])
    return rgb

colourList = []
for i in range(maxN):
    print("generating colour", i)
    colour = randomColour()
    colourList.append(colour) 

for i in range(1, maxN+1):
    print("replacing component", i)
    pos = np.where(labelledOutputs == i)
    print(colourList[i-1])
    print(colourList[i-1].dtype)
    for j in range(len(pos[0])):
        rgbOutputs[pos[0][j], pos[1][j], pos[2][j]] = colourList[i-1]

tifffile.imwrite('ImageStackBinaryComponentsBigFusedRGB.tif', rgbOutputs, photometric='rgb')