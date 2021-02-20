# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import cv2  # computer vision
from skimage import io  # scikit-image

# load image as pixel array
imageFilename = './../dissertation/resources/For_Richard_New/cropped-1_4umperpixel.tif'

imageStack = io.imread(imageFilename)
print(imageStack.shape[0], ' slices in z stack each ',
      imageStack.shape[1], ' by ', imageStack.shape[2], ' pixels ')

newImageStack = np.zeros(shape = (imageStack.shape[0], imageStack.shape[1], imageStack.shape[2]))

print(newImageStack.shape)

# pixel size in micrometres
pixelSize = 1.4
print('each pixel ', pixelSize, ' micrometres a side')

# TODO Loop over all 184 slices and create 3D binary array
# up to 184 images
for imageNo in range(imageStack.shape[0]):
    print('read in image number ', imageNo, ' from tiff stack')
    print(imageStack[imageNo].shape)

    # get slice of image
    imageSlice = imageStack[imageNo]

    # convert image to numpy array
    imageSliceArray = np.asarray(imageSlice, dtype=int)

    # only count pixels with values > 1
    thresholdValue = 1
    print('initially counting all pixels with intensity greater than ',
        thresholdValue, ' as in fibres')
    ret, binaryImage = cv2.threshold(
        imageSlice, thresholdValue, 1, cv2.THRESH_BINARY)
    print(binaryImage.dtype)
    print(binaryImage.shape)

    # find all connected components of that binary image
    # i.e. number of sets with connected pixels
    retVal, labels = cv2.connectedComponents(binaryImage)
    print(retVal-1, ' sets of above threshold pixels')

    numFibres = retVal-1
    numFibrePixels = 0
    minFibreSize = 9

    # removing clusters of connected pixels of a size less than minFibreSize
    print('removing connected clusters of pixels with less than ',
        minFibreSize, ' pixels')
    for i in range(1, retVal):
        pts = np.where(labels == i)
        print(i, len(pts[0]))
        if len(pts[0]) < minFibreSize:
            labels[pts] = 0
            numFibres = numFibres-1
            print('removing tiny possible fibre of ', len(pts[0]), ' pixels')
        else:
            #        print('keeping fibre')
            labels[pts] = 1
            numFibrePixels = numFibrePixels+len(pts[0])
    
    print('total number of fibre pixels ', numFibrePixels)

    array1d = imageSliceArray.flatten()
    fracFibres = numFibrePixels / len(array1d)
    print('fraction of area occupied by pixels ', round(fracFibres, 4))

    meanFibreArea = (numFibrePixels / numFibres) * pixelSize**2
    print('mean area of fibre ', meanFibreArea, ' um^2')

    meanFibreDiameter = np.sqrt(meanFibreArea)
    print('mean diameter of fibre ', meanFibreDiameter, ' um')

    estChannelWidth = meanFibreDiameter / np.sqrt(fracFibres)
    print('estimated channel width  ', estChannelWidth, ' um')

    counter = np.count_nonzero(labels > 0)
    print(counter, ' fibre pixels')
    newImageStack[imageNo] = labels

filename = '3DImageStackBinary'
np.save(filename, newImageStack)