## processDataTo3D
import numpy as np
import cv2  # computer vision
from skimage import io  # scikit-image
import cc3d
import tifffile
from scipy.ndimage import gaussian_filter

# load image as pixel array
# CHANGE THIS BEFORE RUNNING
imageFilename = './../dissertation/resources/data/bigFused-2/bigFused-2.tif'

imageStack = io.imread(imageFilename)
imageStack = imageStack
print(imageStack.shape[0], ' slices in z stack each ',
      imageStack.shape[1], ' by ', imageStack.shape[2], ' pixels ')

newImageStack = np.zeros(shape = (imageStack.shape[0], imageStack.shape[1], imageStack.shape[2]), dtype=np.uint8)

print(newImageStack.shape)

# pixel size in micrometres
# CHANGE THIS BEFORE RUNNING
pixelSize = 0.28
print('each pixel ', pixelSize, ' micrometres a side')

print('Max value in image is ', np.amax(imageStack))
print('Min value in image is ', np.amin(imageStack))

# finding all connected components in 3D for comparison before blur/thresh
# only 4,8 (2D) and 26, 18, and 6 (3D) are allowed 
connectivity = 26
labelsOut, N = cc3d.connected_components(imageStack, return_N=True, connectivity=connectivity)

print("Before any processing there are ", N, " components")

print(imageStack.shape)
blur = gaussian_filter(imageStack, sigma=1)
print(blur.shape)

# loop over every slice in z-axis
for imageNo in range(blur.shape[0]):
    print('Reading in slice ', imageNo, ' from tiff stack')

    # reading in slice and converting to numpy array
    imageSlice = blur[imageNo]

    # use simple binary threshold to filter out noise
    thresholdValue = 50
    N, outputVals = cv2.threshold(imageSlice, thresholdValue, 1, cv2.THRESH_BINARY)
    
    newImageStack[imageNo] = outputVals

# saving blurred and thresholded for comparison to original, before starting analysis
print('Saving files')
pts = np.where(newImageStack == 1)
newImageStack[pts] = 255
tifffile.imwrite('ImageStackBlurredThreshFULL.tif', newImageStack)
tifffile.imwrite('ImageStackOriginalFULL.tif', imageStack)

# finding all connected components in 3D
# only 4,8 (2D) and 26, 18, and 6 (3D) are allowed 
connectivity = 26
labelsOut, N = cc3d.connected_components(newImageStack, return_N=True, connectivity=connectivity)
binaryOut = np.zeros(shape = (imageStack.shape[0], imageStack.shape[1], imageStack.shape[2]), dtype=np.uint8)

print("initially found ", N, " components")
print("Now filtering based on size")

minFibreSize = 25
nFibres = 0
# creates binary and labelled imagery if of correct size
for i in range(1, N+1):
    print("on component ", i)
    pts = np.where(labelsOut == i)
    if(len(pts[0]) < minFibreSize):
        print('removing possible fibre of size ', len(pts[0]), ' pixels')
        binaryOut[pts] = 0
        labelsOut[pts] = 0
    else:
        print('keeping fibre of size', len(pts[0]), ' pixels')
        binaryOut[pts] = 1
        nFibres += 1
        labelsOut[pts] = nFibres

print("There are now ", nFibres, " components")

filename='ImageStackFULL.npz'
np.savez_compressed(filename, original=imageStack, blurThresh=newImageStack, labelledOut=labelsOut, binaryOut=binaryOut)

