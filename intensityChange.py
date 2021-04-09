import numpy as np
from skimage import io
import plotly.express as px

imageFilename = './resources/data/Fused3103.tif'
imageStack = io.imread(imageFilename)

print(imageStack.shape[0], ' slices in z stack each ',
      imageStack.shape[1], ' by ', imageStack.shape[2], ' pixels ')

print('image stack has datatype:', imageStack.dtype)

meanIntensities = []
maxMeanIntensities = []
sliceNumber = []

for i in range(imageStack.shape[0]):
    slice = imageStack[i]
    flatSlice = slice.flatten()
    print('flat slice length', len(flatSlice))

    filteredFlatSlice = flatSlice[flatSlice > 0]
    print('filtered flat slice length', len(filteredFlatSlice))

    sliceMeanPixelIntensity = np.mean(filteredFlatSlice)

    nMax = 50000 # number of max intensities to get
    sliceMaxIntensitiesIndices = np.argpartition(filteredFlatSlice, -nMax)[-nMax:]
    sliceMaxIntensitiesArray = filteredFlatSlice[sliceMaxIntensitiesIndices]
    sliceMaxIntensitiesMean = np.mean(sliceMaxIntensitiesArray)

    meanIntensities.append(sliceMeanPixelIntensity)
    maxMeanIntensities.append(sliceMaxIntensitiesMean)
    sliceNumber.append(i+1)

statsDataFrame = {
    'meanIntensities': meanIntensities,
    'maxMeanIntensities': maxMeanIntensities,
    'sliceNumber': sliceNumber
}

fig = px.line(
    statsDataFrame, 
    x="sliceNumber", 
    y="meanIntensities", 
    title="Mean intensity as a function of slice for 'Fused 3103' image"
    )
fig.show()

maxMeanIntensitiesTitle = "Mean intensity of the top " + str(nMax) + " pixels as a function of slice for 'Fused 3103' image"

fig = px.line(
    statsDataFrame, 
    x="sliceNumber", 
    y="maxMeanIntensities", 
    title=maxMeanIntensitiesTitle
)
fig.show()


