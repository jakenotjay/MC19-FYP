# POC using cc3d to find air gaps - not great tbh
import numpy as np
import cc3d
import random
import tifffile

# load file
zipFile = np.load('./outputs/npz/FinalFusedThresh5.npz')
binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool')

print('The x axis is', binaryOutputs.shape[1], 'by', binaryOutputs.shape[2], 'on the y axis')
print('if this is wrong change the variable areAxesFlipped to True')

# if the xy axes flipped (often loading the tif they are) then we can flip them back the right way round
areAxesFlipped = True
if(areAxesFlipped):
    binaryOutputs = np.swapaxes(binaryOutputs, 1, 2)
    print('binary outputs now has the x axis as', binaryOutputs.shape[1], 'by', binaryOutputs.shape[2], 'on the y axis')

xShape = binaryOutputs.shape[1]
yShape = binaryOutputs.shape[2]
zShape = binaryOutputs.shape[0]

# inverse image i.e 1 to 0, 0 to 1
inverseImage = np.array(np.invert(binaryOutputs), dtype='uint8')
print('inverse image has shape', inverseImage.shape[0], inverseImage.shape[1], inverseImage.shape[2])

inverseImage = inverseImage[9:, 0:, 170:650]

# finding all connected components in 3D for comparison before blur/thresh
# only 4,8 (2D) and 26, 18, and 6 (3D) are allowed 
connectivity = 6
labelsOut, N =cc3d.connected_components(inverseImage, return_N=True, connectivity=connectivity)

print('N is ', N)

rgbOutputs = np.zeros(shape = (labelsOut.shape[0], labelsOut.shape[1], labelsOut.shape[2], 3), dtype=np.uint8)

# generates three random values between 0-255 to generate a colour value
def randomColour():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    rgb = np.asarray([r, g, b])
    return rgb

# generates colour list
colourList = []
for i in range(N):
    colour = randomColour()
    colourList.append(colour) 

# loops through all components and replaces them with a corresponding rgb value from colourList array
for i in range(1, N+1):
    print("replacing component", i)
    pos = np.where(labelsOut == i)
    print(colourList[i-1])
    print(colourList[i-1].dtype)
    for j in range(len(pos[0])):
        rgbOutputs[pos[0][j], pos[1][j], pos[2][j], 0] = colourList[i-1][0]
        rgbOutputs[pos[0][j], pos[1][j], pos[2][j], 1] = colourList[i-1][1]
        rgbOutputs[pos[0][j], pos[1][j], pos[2][j], 2] = colourList[i-1][2]

outputFile = np.zeros(shape=(zShape, xShape, yShape, 3), dtype=np.uint8)
outputFile[9:, 0:, 170:650] = rgbOutputs
        
# write image as a tiff file
tifffile.imwrite('./outputs/tif/AirGaps2.tif', outputFile, photometric='rgb')