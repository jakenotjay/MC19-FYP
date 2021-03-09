import numpy as np
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

# load image as pixel array
image = np.load('./3DImageStackBinary.npy')

# pixel size in micrometres
pixelSize = 1.4
print('Each pixel has side length', pixelSize, ' micrometres a side')

# generate stats function
def genStats(imageSlice):
    imageSlice = imageSlice.astype(np.uint8)

    retVal, labels = cv2.connectedComponents(imageSlice)

    numFibres = retVal - 1
    numFibrePixels = 0
    flatArray = imageSlice.flatten()

    for i in range(1, retVal):
        pts = np.where(labels == i)
        labels[pts] = 1
        numFibrePixels = numFibrePixels + len(pts[0])

    fracFibres = numFibrePixels / len(flatArray)

    meanFibreArea = (numFibrePixels / numFibres) * pixelSize **2

    meanFibreDiameter = np.sqrt(meanFibreArea)

    estChannelWidth = meanFibreDiameter / np.sqrt(fracFibres)

    return fracFibres, meanFibreArea, meanFibreDiameter, estChannelWidth


# z plane
nSlicesZ = image.shape[0]
print(nSlicesZ)

fracFibresSlicesZ = np.zeros(nSlicesZ)
meanFibreAreaSlicesZ = np.zeros(nSlicesZ)
meanFibreDiameterSlicesZ = np.zeros(nSlicesZ)
estChannelWidthSlicesZ = np.zeros(nSlicesZ)

for imageNo in range(nSlicesZ):
    imageSlice = image[imageNo]
    fracFibres, meanFibreArea, meanFibreDiameter, estChannelWidth = genStats(imageSlice)

    fracFibresSlicesZ[imageNo] = fracFibres
    meanFibreAreaSlicesZ[imageNo] = meanFibreArea
    meanFibreDiameterSlicesZ[imageNo] = meanFibreDiameter
    estChannelWidthSlicesZ[imageNo] = estChannelWidth



# x plane
nSlicesX = image.shape[1]
print(nSlicesX)

fracFibresSlicesX = np.zeros(nSlicesX)
meanFibreAreaSlicesX = np.zeros(nSlicesX)
meanFibreDiameterSlicesX = np.zeros(nSlicesX)
estChannelWidthSlicesX = np.zeros(nSlicesX)

for imageNo in range(nSlicesX):
    imageSlice = image[0:, imageNo, 0:]
    fracFibres, meanFibreArea, meanFibreDiameter, estChannelWidth = genStats(imageSlice)

    fracFibresSlicesX[imageNo] = fracFibres
    meanFibreAreaSlicesX[imageNo] = meanFibreArea
    meanFibreDiameterSlicesX[imageNo] = meanFibreDiameter
    estChannelWidthSlicesX[imageNo] = estChannelWidth

# y plane
nSlicesY = image.shape[2]
print(nSlicesY)

fracFibresSlicesY = np.zeros(nSlicesY)
meanFibreAreaSlicesY = np.zeros(nSlicesY)
meanFibreDiameterSlicesY = np.zeros(nSlicesY)
estChannelWidthSlicesY = np.zeros(nSlicesY)

for imageNo in range(nSlicesY):
    imageSlice = image[0:, imageNo, 0:]
    fracFibres, meanFibreArea, meanFibreDiameter, estChannelWidth = genStats(imageSlice)

    fracFibresSlicesY[imageNo] = fracFibres
    meanFibreAreaSlicesY[imageNo] = meanFibreArea
    meanFibreDiameterSlicesY[imageNo] = meanFibreDiameter
    estChannelWidthSlicesY[imageNo] = estChannelWidth

# TODO : generate bins and number of items in each bin to plot inside histogram

dataZ = {
    'FibreFraction': fracFibresSlicesZ, 
    'MeanFibreArea': meanFibreAreaSlicesZ,
    'MeanFibreDiameter': meanFibreDiameterSlicesZ,
    'EstimatedChannelWidth': estChannelWidthSlicesZ
    }

dataX = {
    'FibreFraction': fracFibresSlicesX, 
    'MeanFibreArea': meanFibreAreaSlicesX,
    'MeanFibreDiameter': meanFibreDiameterSlicesX,
    'EstimatedChannelWidth': estChannelWidthSlicesX
    }

dataY = {
    'FibreFraction': fracFibresSlicesY, 
    'MeanFibreArea': meanFibreAreaSlicesY,
    'MeanFibreDiameter': meanFibreDiameterSlicesY,
    'EstimatedChannelWidth': estChannelWidthSlicesY
    }

dataFrameZ = pd.DataFrame(data = dataZ)
dataFrameX = pd.DataFrame(data = dataX)
dataFrameY = pd.DataFrame(data = dataY)

fig = make_subplots(rows=2, cols=2)

fig.add_traces(
    go.Histogram(
        x=dataFrameY['FibreFraction'],
        name='Fibre Fraction'
        ),
    rows=1, cols=1
)

fig.add_traces(
    go.Histogram(
        x=dataFrameY['MeanFibreArea'],
        name='Mean Fibre Area'),
    rows=1, cols=2
)

fig.add_traces(
    go.Histogram(
        x=dataFrameY['MeanFibreDiameter'],
        name='Mean Fibre Diameter'
    ),
    rows=2, cols=1
)

fig.add_traces(
    go.Histogram(
        x=dataFrameY['EstimatedChannelWidth'],
        name='Estimated Channel Width'
        ),
    rows=2, cols=2
)

fig.show()