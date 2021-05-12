import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
# import tifffile

# choose some squares
# loop through entire image counting number of zeros,
# count number of obstructions,
# use distance transform on inverse image to calculate radius,
# use that radius to calculate a channel width
# count number of continuous air gaps - i.e. gaps with no obstructions on the same pixel

# zipFile = np.load('./outputs/npz/FinalFusedThresh5.npz')
# binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool')

# pixel size in micrometres
# CHANGE THIS
pixelSize = 1.82

# print('The x axis is', binaryOutputs.shape[1], 'by', binaryOutputs.shape[2], 'on the y axis')
# print('if this is wrong change the variable areAxesFlipped to True')


def findContinuousGaps(gap):
    gap = gap.astype('uint8')
    nSlices = gap.shape[0]
    xRange = gap.shape[1]
    yRange = gap.shape[2]
    print('z range is', nSlices)
    print('x range is', xRange)
    print('y range is', yRange)
    print('number of pixels', xRange * yRange)
    gapSize = xRange * yRange

    meanGapValues = np.zeros(shape=(xRange, yRange), dtype='uint8')
    continuousCount = 0
    fibrePixels = 0

    for i in range(xRange):
        for j in range(yRange):
            airPixelCount = 0
            for k in range(nSlices):
                if(gap[k, i, j] == 0.0):
                    airPixelCount += 1

            fibrePixels = nSlices - airPixelCount
            meanGapValues[i, j]  = fibrePixels/nSlices
            # print(meanGapValues[i, j])
            if (airPixelCount == nSlices):
                continuousCount +=1

    # print(meanGapValues)

    return meanGapValues, continuousCount, fibrePixels
                
def processGap(gap):
    inverseGap = np.array(np.invert(gap), dtype='uint8')
    maxDistance = np.sqrt(inverseGap.shape[1]**2 + inverseGap.shape[2]**2)
    nSlices = gap.shape[0]
    gapSize = gap.shape[1] * gap.shape[2]

    airCount = np.zeros(nSlices)
    fibreCount = np.zeros(nSlices)
    gapRadius = np.zeros(nSlices)
    sliceN = np.zeros(nSlices)

    for i in range(nSlices):
        # print('on slice', i)
        sliceN[i] = i+1
        slice = gap[i]
        inverseSlice = inverseGap[i]
        pts = np.where(slice == 0)
        airCount[i] = len(pts[0])
        fibreCount[i] = gapSize - airCount[i]

        distances = cv2.distanceTransform(inverseSlice, cv2.DIST_L2, cv2.DIST_MASK_5)

        minFilterArray = distances > 0 
        distances = distances[minFilterArray]
        maxFilterArray = distances < maxDistance
        distances = distances[maxFilterArray]

        maxDistance = np.max(distances) * pixelSize
        gapRadius[i] = maxDistance

    meanChannelWidth = np.mean(gapRadius) * 2

    return airCount, fibreCount, gapRadius, sliceN, meanChannelWidth

def generateGraphs(sliceN, airCount, fibreCount, gapRadius, meanGapValues):
    airCountFig = go.Figure(data=go.Scatter(
    x=sliceN, y=airCount, 
    mode='lines',
    name='Number of air count pixels'
    ))
    airCountFig.update_layout(
        title='Number of air count pixels as a function of slice',
        xaxis_title='Slice Number',
        yaxis_title='Number of air pixels',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    airCountFig.show()

    fibreCountFig = go.Figure(data=go.Scatter(
        x=sliceN, y=fibreCount, 
        mode='lines',
        name='Number of fibre count pixels'
        ))
    fibreCountFig.update_layout(
        title='Number of fibre count pixels as a function of slice',
        xaxis_title='Slice Number',
        yaxis_title='Number of fibre pixels',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    fibreCountFig.show()

    gapRadiusFig = go.Figure(data=go.Scatter(
        x=sliceN, y=gapRadius, 
        mode='lines',
        name='Gap size radius'
    ))
    gapRadiusFig.update_layout(
        title='Radius of gap as a function of slice',
        xaxis_title='Slice Number',
        yaxis_title='Gap radius (um)',
        legend_title='',
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    gapRadiusFig.show()

    # meanGapValuesFig = go.Figure()
    # meanGapValuesFig.add_trace(go.Heatmap(
    #     z=meanGapValues,
    #     name='Heatmap of average gap size',
    #     colorbar=dict(
    #         title='average value'
    #     )
    # ))
    # meanGapValuesFig['layout']['yaxis']['autorange'] = 'reversed'
    # meanGapValuesFig.show()

# print('cube 1')
# airCount1, fibreCount1, gapRadius1, sliceN1, meanChannelWidth1 = processGap(cube1)
# meanGapValues1, continuousCount1, fibrePixels1 = findContinuousGaps(cube1)
# generateGraphs(sliceN1, airCount1, fibreCount1, gapRadius1, meanGapValues1)

# print('cube 2')
# airCount2, fibreCount2, gapRadius2, sliceN2, meanChannelWidth2 = processGap(cube2)
# meanGapValues2, continuousCount2, fibrePixels2 = findContinuousGaps(cube2)
# generateGraphs(sliceN2, airCount2, fibreCount2, gapRadius2, meanGapValues2)

# print('cube 3')
# airCount3, fibreCount3, gapRadius3, sliceN3, meanChannelWidth3 = processGap(cube3)
# meanGapValues3, continuousCount3, fibrePixels3 = findContinuousGaps(cube3)
# generateGraphs(sliceN3, airCount3, fibreCount3, gapRadius3, meanGapValues3)

# print('cube 4')
# airCount4, fibreCount4, gapRadius4, sliceN4, meanChannelWidth4 = processGap(cube4)
# meanGapValues4, continuousCount4, fibrePixels4 = findContinuousGaps(cube4)
# generateGraphs(sliceN4, airCount4, fibreCount4, gapRadius4, meanGapValues4)

# print('cube 5')
# airCount5, fibreCount5, gapRadius5, sliceN5, meanChannelWidth5 = processGap(cube5)
# meanGapValues5, continuousCount5, fibrePixels5 = findContinuousGaps(cube5)
# generateGraphs(sliceN5, airCount5, fibreCount5, gapRadius5, meanGapValues5)

# print('channel width has been calculated to be')
# print('for cube 1, the channel width is', meanChannelWidth1)
# print('for cube 2, the channel width is', meanChannelWidth2)
# print('for cube 3, the channel width is', meanChannelWidth3)
# print('for cube 4, the channel width is', meanChannelWidth4)
# print('for cube 5, the channel width is', meanChannelWidth5)

# print('count of continuous air pixels is', continuousCount1)
# print('count of continuous air pixels is', continuousCount2)
# print('count of continuous air pixels is', continuousCount3)
# print('count of continuous air pixels is', continuousCount4)
# print('count of continuous air pixels is', continuousCount5)

# print('count of fibre pixels is')

fileList = ['FinalFusedThresh5Final.npz', 
            'FinalFusedThresh10Final.npz',
            'FinalFusedThresh15Final.npz',
            'FinalFusedThresh25Final.npz']

gapRadii1 = []
gapRadii2 = []
gapRadii3 = []
gapRadii4 = []
gapRadii5 = []

for i in range(len(fileList)):
    zipFile = np.load('./outputs/npz/final/' + fileList[i])
    binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool') [10:30]
    
    # if the xy axes flipped (often loading the tif they are) then we can flip them back the right way round
    areAxesFlipped = True
    if(areAxesFlipped):
        binaryOutputs = np.swapaxes(binaryOutputs, 1, 2)
        print('binary outputs now has the x axis as', binaryOutputs.shape[1], 'by', binaryOutputs.shape[2], 'on the y axis')

    cube1 = binaryOutputs[0:, 150:275, 277:400]
    cube2 = binaryOutputs[0:, 2:133, 280:400]
    cube3 = binaryOutputs[0:, 335:460, 270:410]
    cube4 = binaryOutputs[0:, 480:605, 415:540]
    cube5 = binaryOutputs[0:, 485:610, 275:400]
    
    airCount, fibreCount, gapRadius, sliceN, meanChannelWidth = processGap(cube1)
    gapRadii1.append(gapRadius)
    airCount, fibreCount, gapRadius, sliceN, meanChannelWidth = processGap(cube2)
    gapRadii2.append(gapRadius)
    airCount, fibreCount, gapRadius, sliceN, meanChannelWidth = processGap(cube3)
    gapRadii3.append(gapRadius)
    airCount, fibreCount, gapRadius, sliceN, meanChannelWidth = processGap(cube4)
    gapRadii4.append(gapRadius)
    airCount, fibreCount, gapRadius, sliceN, meanChannelWidth = processGap(cube5)
    gapRadii5.append(gapRadius)

gapRadii1 = np.asarray(gapRadii1)
gapRadii2 = np.asarray(gapRadii2)
gapRadii3 = np.asarray(gapRadii3)
gapRadii4 = np.asarray(gapRadii4)
gapRadii5 = np.asarray(gapRadii5)
print('shape of gap radii 1', gapRadii1.shape)

gapRadii1 = {
    'Threshold 5': gapRadii1[0],
    'Threshold 10': gapRadii1[1],
    'Threshold 15': gapRadii1[2],
    'Threshold 25': gapRadii1[3]
}
gapRadii2 = {
    'Threshold 5': gapRadii2[0],
    'Threshold 10': gapRadii2[1],
    'Threshold 15': gapRadii2[2],
    'Threshold 25': gapRadii2[3]
}
gapRadii3 = {
    'Threshold 5': gapRadii3[0],
    'Threshold 10': gapRadii3[1],
    'Threshold 15': gapRadii3[2],
    'Threshold 25': gapRadii3[3]
} 
gapRadii4 = {
    'Threshold 5': gapRadii4[0],
    'Threshold 10': gapRadii4[1],
    'Threshold 15': gapRadii4[2],
    'Threshold 25': gapRadii4[3]
} 
gapRadii5 = {
    'Threshold 5': gapRadii5[0],
    'Threshold 10': gapRadii5[1],
    'Threshold 15': gapRadii5[2],
    'Threshold 25': gapRadii5[3]
}

gapRadii1 = pd.DataFrame(gapRadii1)
gapRadii2 = pd.DataFrame(gapRadii2)
gapRadii3 = pd.DataFrame(gapRadii3)
gapRadii4 = pd.DataFrame(gapRadii4)
gapRadii5 = pd.DataFrame(gapRadii5)

gaps = [gapRadii1, gapRadii2, gapRadii3, gapRadii4, gapRadii5]

for i in range(len(gaps)):
    gapDF = gaps[i]
    threshold5Mean = np.mean(gapDF['Threshold 5'])
    threshold5Uncertainty = np.std(gapDF['Threshold 5'])
    # threshold5Uncertainty = (np.max(gapDF['Threshold 5']) - np.min(gapDF['Threshold 5'])) / 2
    threshold10Mean = np.mean(gapDF['Threshold 10'])
    threshold10Uncertainty = np.std(gapDF['Threshold 10'])
    # threshold10Uncertainty = (np.max(gapDF['Threshold 10']) - np.min(gapDF['Threshold 10'])) / 2
    threshold15Mean = np.mean(gapDF['Threshold 15'])
    threshold15Uncertainty = np.std(gapDF['Threshold 15'])
    # threshold15Uncertainty = (np.max(gapDF['Threshold 15']) - np.min(gapDF['Threshold 15'])) / 2
    threshold25Mean = np.mean(gapDF['Threshold 25'])
    threshold25Uncertainty = np.std(gapDF['Threshold 25'])
    # threshold25Uncertainty = (np.max(gapDF['Threshold 25']) - np.min(gapDF['Threshold 25'])) / 2
    
    gapMeanValue = (threshold5Mean + threshold10Mean + threshold15Mean + threshold25Mean) / 4
    gapUncertaintyValue = (threshold5Uncertainty + threshold10Uncertainty + threshold15Uncertainty + threshold25Uncertainty) / 4
    print('For gap', i, 'it has a mean radius of', gapMeanValue, 'with an uncertainty of', gapUncertaintyValue)
    