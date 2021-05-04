## Creates stats before and after processing of an image to get differences for analysis
## can be used to verify that processing is working correctly
import numpy as np
import cv2
from numpy.core.fromnumeric import nonzero
import pandas as pd
import plotly.graph_objects as go

def noiseAnalysis(image):
    nSlices = image.shape[0]
    nNoiseSlice = []
    noiseFracSlice = []
    nonZeroPixelFracSlice = []
    sliceN = []
    meanNoiseIntensity = []
    nonZeroPixelsSlice = []

    for slice in range(nSlices):
        sliceN.append(slice+1)
        print('on slice', slice)
        sliceImage = image[slice]
        nComponents, componentsArray = cv2.connectedComponents(sliceImage)
        print('there are', nComponents, 'components')
        nNoise = 0
        noisePixelIntensity = []

        # calculates number of noise pixels and intensity of that noise
        for component in range(1, nComponents+1):
            # print('considering component', component)
            pts = np.where(componentsArray == component)
            if(len(pts[0]) == 1):
                nNoise +=1
                noisePixelIntensity.append(sliceImage[pts])

        if(nNoise > 0):
            print('max noise intensity', np.max(noisePixelIntensity))
            print('min noise intensity', np.min(noisePixelIntensity))
            meanNoiseIntensity.append(np.mean(noisePixelIntensity))
        else:
            meanNoiseIntensity.append(0)
        
        print('number of noise particle', nNoise)

        ## pixelFractionCalculations
        # calculate number of pixels with intensity > 0
        nNonZeroPixels = np.count_nonzero(sliceImage)
        nonZeroPixelsSlice.append(nNonZeroPixels)
        totalPixels = sliceImage.shape[0] * sliceImage.shape[1]
        fracNonZeroPixels = nNonZeroPixels / totalPixels
        # calculate that as a fraction of total number of pixels x * y
        # use the number of pixels with intensity to calculate noiseFracSlice

        nonZeroPixelFracSlice.append(fracNonZeroPixels)
        nNoiseSlice.append(nNoise)
        fracNoisePixelsToNonZeroPixels = nNoise / nNonZeroPixels
        noiseFracSlice.append(fracNoisePixelsToNonZeroPixels)
        
    noiseData = {
        'sliceN': sliceN,
        'nNoisePixels': nNoiseSlice,
        'noiseFracSlice': noiseFracSlice,
        'meanNoiseIntensity': meanNoiseIntensity,
        'nonZeroPixelFracSlice': nonZeroPixelFracSlice,
        'nonZeroPixelsSlice': nonZeroPixelsSlice
    }

    df = pd.DataFrame(data=noiseData)
    print(df)

    return df

def visualiseNoiseAnalysis(originalDF, outputDF):
    noiseFig = go.Figure()

    noiseFig.add_trace(go.Scatter(
        x=originalDF['sliceN'], y=originalDF['nNoisePixels'], 
        mode='lines',
        name='N of Noise Pixels (Isolated Pixels) of original image'
    ))

    noiseFig.add_trace(go.Scatter(
        x=outputDF['sliceN'], y=outputDF['nNoisePixels'], 
        mode='lines',
        name='N of Noise Pixels (Isolated Pixels) of processed image'
    ))

    noiseFig.update_layout(
        title='The number of noise pixels (isolated pixels) as a function of slice',
        xaxis_title='Slice Number',
        yaxis_title='Number of Noise pixels',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    noiseFig.show()

    noiseFracFig = go.Figure()
    noiseFracFig.add_trace(go.Scatter(
        x=originalDF['sliceN'], y=originalDF['noiseFracSlice'], 
        mode='lines',
        name='Fraction of noise pixels/total non zero pixels of the original image'
        ))
    noiseFracFig.add_trace(go.Scatter(
        x=outputDF['sliceN'], y=outputDF['noiseFracSlice'], 
        mode='lines',
        name='Fraction of noise pixels/total non zero pixels of the output image'
        ))

    noiseFracFig.update_layout(
        title='The fraction of noise pixels/total non zero pixels as a function of slice',
        xaxis_title='Slice Number',
        yaxis_title='Fraction of noise pixels/total non zero pixels',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    noiseFracFig.show()

    meanNoiseIntensityFig = go.Figure()

    meanNoiseIntensityFig.add_trace(go.Scatter(
        x=originalDF['sliceN'], y=originalDF['meanNoiseIntensity'], 
        mode='lines',
        name='Mean intensity of noise pixels of original image'
        ))
    meanNoiseIntensityFig.add_trace(go.Scatter(
        x=outputDF['sliceN'], y=outputDF['meanNoiseIntensity'], 
        mode='lines',
        name='Mean intensity of noise pixels of output image'
        ))
    meanNoiseIntensityFig.update_layout(
        title='Mean intensity of noise pixels (isolated pixels)',
        xaxis_title='Slice Number',
        yaxis_title='Mean Intensity (0-255)',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    meanNoiseIntensityFig.show()

    nonZeroPixelFracSliceFig = go.Figure()
    nonZeroPixelFracSliceFig.add_trace(go.Scatter(
        x=originalDF['sliceN'], y=originalDF['nonZeroPixelFracSlice'], 
        mode='lines',
        name='Non zero pixel fraction of the original image'
        ))
    nonZeroPixelFracSliceFig.add_trace(go.Scatter(
        x=outputDF['sliceN'], y=outputDF['nonZeroPixelFracSlice'], 
        mode='lines',
        name='Non zero pixel fraction of the output image'
        ))
    nonZeroPixelFracSliceFig.update_layout(
        title='The fraction of non-zero pixels to image resolution',
        xaxis_title='Slice Number',
        yaxis_title='Non zero pixel fraction',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    nonZeroPixelFracSliceFig.show()

    nonZeroPixelsSliceFig = go.Figure()
    nonZeroPixelsSliceFig.add_trace(go.Scatter(
        x=originalDF['sliceN'], y=originalDF['nonZeroPixelsSlice'], 
        mode='lines',
        name='Number of non zero pixels for the original image'
        ))
    nonZeroPixelsSliceFig.add_trace(go.Scatter(
        x=outputDF['sliceN'], y=outputDF['nonZeroPixelsSlice'], 
        mode='lines',
        name='Number of non zero pixels for the output image'
        ))
    nonZeroPixelsSliceFig.update_layout(
        title='Number of non zero pixels as a function fo slice number',
        xaxis_title='Slice Number',
        yaxis_title='Number of non zero pixels',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    nonZeroPixelsSliceFig.show()

zipFile = np.load('./outputs/npz/FinalFusedThresh5.npz')
binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='uint8')
original = np.asarray(zipFile['original'], dtype='uint8')
originalBinary = np.zeros(shape=(original.shape[0], original.shape[1], original.shape[2]), dtype='uint8')

for imageNo in range(original.shape[0]):
    N, originalBinary[imageNo] = cv2.threshold(original[imageNo], 0, 1, cv2.THRESH_BINARY)

originalBinaryPts = np.where(originalBinary == 1)
binaryOutputsPts = np.where(binaryOutputs == 1)

differenceInPixels = len(originalBinaryPts[0]) - len(binaryOutputsPts[0])
percentageOfPixelsLost = differenceInPixels/len(originalBinaryPts[0]) * 100

print('The processing has removed', differenceInPixels, 'pixels')
print('This is a percentage of', percentageOfPixelsLost, '%')
originalNoiseDF = noiseAnalysis(originalBinary)
outputNoiseDF = noiseAnalysis(binaryOutputs)

originalNoiseDF.to_pickle('./outputs/comparisons/originalNoiseDF.pkl')
outputNoiseDF.to_pickle('./outputs/comparisons/outputNoiseDF.pkl')

visualiseNoiseAnalysis(originalNoiseDF, outputNoiseDF)
