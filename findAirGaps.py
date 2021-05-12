import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
# import tifffile
import pandas as pd

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
    maxPossibleDistance = np.sqrt(inverseGap.shape[1]**2 + inverseGap.shape[2]**2)
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
        maxFilterArray = distances < maxPossibleDistance
        distances = distances[maxFilterArray]

        if(len(distances) > 0):
            maxDistance = np.max(distances) * pixelSize
            gapRadius[i] = maxDistance

    meanChannelWidth = np.mean(gapRadius) * 2

    return airCount, fibreCount, gapRadius, sliceN, meanChannelWidth

def generateGraphs(sliceN, airCount, fibreCount, gapRadius, meanGapValues):
    # airCountFig = go.Figure(data=go.Scatter(
    # x=sliceN, y=airCount, 
    # mode='lines',
    # name='Number of air count pixels'
    # ))
    # airCountFig.update_layout(
    #     title='Number of air count pixels as a function of slice',
    #     xaxis_title='Slice Number',
    #     yaxis_title='Number of air pixels',
    #     legend_title='',
    #     font=dict(
    #         # family="Courier New, monospace",
    #         size=18,
    #         # color="RebeccaPurple"
    #     )
    # )
    # airCountFig.show()

    # fibreCountFig = go.Figure(data=go.Scatter(
    #     x=sliceN, y=fibreCount, 
    #     mode='lines',
    #     name='Number of fibre count pixels'
    #     ))
    # fibreCountFig.update_layout(
    #     title='Number of fibre count pixels as a function of slice',
    #     xaxis_title='Slice Number',
    #     yaxis_title='Number of fibre pixels',
    #     legend_title='',
    #     font=dict(
    #         # family="Courier New, monospace",
    #         size=18,
    #         # color="RebeccaPurple"
    #     )
    # )
    # fibreCountFig.show()

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

def generateComparisonGraphs(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['sliceN'], y=df['thresh 1'], mode='lines', name='Threshold 1', line_shape='spline'))
    fig.add_trace(go.Scatter(x=df['sliceN'], y=df['thresh 2'], mode='lines', name='Threshold 2', line_shape='spline'))
    fig.add_trace(go.Scatter(x=df['sliceN'], y=df['thresh 5'], mode='lines', name='Threshold 5', line_shape='spline'))
    fig.add_trace(go.Scatter(x=df['sliceN'], y=df['thresh 10'], mode='lines', name='Threshold 10', line_shape='spline'))
    fig.add_trace(go.Scatter(x=df['sliceN'], y=df['thresh 15'], mode='lines', name='Threshold 15', line_shape='spline'))
    fig.add_trace(go.Scatter(x=df['sliceN'], y=df['thresh 25'], mode='lines', name='Threshold 25', line_shape='spline'))
    fig.add_trace(go.Scatter(x=df['sliceN'], y=df['thresh 50'], mode='lines', name='Threshold 50', line_shape='spline'))
    
    fig.update_yaxes(title_text="Gap Radii (um)")
    fig.update_xaxes(title_text="Slice Number (depth)")

    fig.update_layout(
        # showlegend=False,
        # width=500,
        font=dict(
            size=16
    ))
    
    fig.show()
    
# choose some squares
# loop through entire image counting number of zeros,
# count number of obstructions,
# use distance transform on inverse image to calculate radius,
# use that radius to calculate a channel width
# count number of continuous air gaps - i.e. gaps with no obstructions on the same pixel

zipFileList = [
    'FinalFusedThresh10Final.npz'
    ]

sliceN = []
gapRadii = []

for i in range(len(zipFileList)):
    print('looking at zip file', i+1)
    zipFile = np.load('./outputs/npz/final/'+zipFileList[i])
    binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool')

    # pixel size in micrometres
    # CHANGE THIS
    pixelSize = 1.82

    print('The x axis is', binaryOutputs.shape[1], 'by', binaryOutputs.shape[2], 'on the y axis')
    print('if this is wrong change the variable areAxesFlipped to True')

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

    print('cube 1')
    airCount1, fibreCount1, gapRadius1, sliceN1, meanChannelWidth1 = processGap(cube1)
    meanGapValues1, continuousCount1, fibrePixels1 = findContinuousGaps(cube1)
    generateGraphs(sliceN1, airCount1, fibreCount1, gapRadius1, meanGapValues1)

    if(i == 0):
        sliceN = sliceN1
    
    gapRadii.append(gapRadius1)

    print('cube 2')
    airCount2, fibreCount2, gapRadius2, sliceN2, meanChannelWidth2 = processGap(cube2)
    meanGapValues2, continuousCount2, fibrePixels2 = findContinuousGaps(cube2)
    generateGraphs(sliceN2, airCount2, fibreCount2, gapRadius2, meanGapValues2)

    print('cube 3')
    airCount3, fibreCount3, gapRadius3, sliceN3, meanChannelWidth3 = processGap(cube3)
    meanGapValues3, continuousCount3, fibrePixels3 = findContinuousGaps(cube3)
    generateGraphs(sliceN3, airCount3, fibreCount3, gapRadius3, meanGapValues3)

    print('cube 4')
    airCount4, fibreCount4, gapRadius4, sliceN4, meanChannelWidth4 = processGap(cube4)
    meanGapValues4, continuousCount4, fibrePixels4 = findContinuousGaps(cube4)
    generateGraphs(sliceN4, airCount4, fibreCount4, gapRadius4, meanGapValues4)

    print('cube 5')
    airCount5, fibreCount5, gapRadius5, sliceN5, meanChannelWidth5 = processGap(cube5)
    meanGapValues5, continuousCount5, fibrePixels5 = findContinuousGaps(cube5)
    generateGraphs(sliceN5, airCount5, fibreCount5, gapRadius5, meanGapValues5)

    print('channel width has been calculated to be')
    print('for cube 1, the channel width is', meanChannelWidth1)
    print('for cube 2, the channel width is', meanChannelWidth2)
    print('for cube 3, the channel width is', meanChannelWidth3)
    print('for cube 4, the channel width is', meanChannelWidth4)
    print('for cube 5, the channel width is', meanChannelWidth5)

    print('count of continuous air pixels is', continuousCount1)
    print('count of continuous air pixels is', continuousCount2)
    print('count of continuous air pixels is', continuousCount3)
    print('count of continuous air pixels is', continuousCount4)
    print('count of continuous air pixels is', continuousCount5)

data = {
    'sliceN': sliceN,
    'thresh 1': gapRadii[0],
    'thresh 2': gapRadii[1],
    'thresh 5': gapRadii[2],
    'thresh 10': gapRadii[3],
    'thresh 15': gapRadii[4],
    'thresh 25': gapRadii[5],
    'thresh 50': gapRadii[6],
}

# df = pd.DataFrame(data)
# generateComparisonGraphs(df)
