# calculates channel width estimates based on fibre distances
# calculates local flow speed based on fibre distances
import numpy as np
import cv2
from numpy.core.fromnumeric import swapaxes
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

zipFile = np.load('./ImageStackFULL.npz')
binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool')

print('The x axis is', binaryOutputs.shape[1], 'by', binaryOutputs.shape[2], 'on the y axis')
print('if this is wrong change the variable areAxesFlipped to True')

areAxesFlipped = True
if(areAxesFlipped):
    binaryOutputs = np.swapaxes(binaryOutputs, 1, 2)
    print('binary outputs now has the x axis as', binaryOutputs.shape[1], 'by', binaryOutputs.shape[2], 'on the y axis')

# inverse image i.e 1 to 0, 0 to 1
inverseImage = np.array(np.invert(binaryOutputs), dtype='uint8')
print('inverse image has shape', inverseImage.shape[0], inverseImage.shape[1], inverseImage.shape[2])


nSlices = inverseImage.shape[0]

# average flow speeds in cms-1, calculated assuming perfect face seal
# and measured over 1 minute
restFlowSpeed = 0.5 * (10**-2) 
mildFlowSpeed = 1.8 * (10**-2)
moderateFlowSpeed = 2.7 * (10**-2) # able to sustain for 8 hoursof work
maximalFlowSpeed = 7.5 * (10**-2) # upper limit of what can be sustained for competitive sports

# change this variable to one of the constants defined above to change results of calculations
flowSpeed = mildFlowSpeed

# calculate max possible distance sqr(x^2 + y^2)
maxDistance = np.sqrt(inverseImage.shape[1]**2 + inverseImage.shape[2]**2)
print('max possible distance', maxDistance, 'pixels')

# sliceNumber, mean, mean^2, stdDev, depth, fibreFrac
statsArray = np.zeros([6, nSlices])

# pixel size in micrometres
pixelSize = 0.28

# finds distance to nearest fibre and generates statistics
def generateDistanceStats(slice, cropXStart, cropXEnd, cropYStart, cropYEnd, filterMin=True, filterMax=True):
    # for all zeros (air pixels) find nearest 1 pixel (fibres)
    # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga25c259e7e2fa2ac70de4606ea800f12f
    distances = cv2.distanceTransform(slice, cv2.DIST_L2, cv2.DIST_MASK_5)
    print('finding distances in periodic shape with shape', distances.shape)
    distances = distances[cropXStart:cropXEnd, cropYStart:cropYEnd]
    print('now cropping to original size, with shape', distances.shape)
    
    # get rid of fibres (0) and those that are impossibly large
    if(filterMin):
        minFilterArray = distances > 0
        distances = distances[minFilterArray]
    if(filterMax):
        maxFilterArray = distances < maxDistance
        distances = distances[maxFilterArray]

    flatDistances = distances.flatten()

    # find distance, mean, mean squared and std
    distances = distances * pixelSize
    flatDistances = flatDistances * pixelSize
    mean = np.mean(flatDistances)
    meanSquared = mean ** 2
    std = np.std(flatDistances)

    return flatDistances, distances, mean, meanSquared, std

# finds local flow speeds based on fibre fraction
def generateLocalFlowSpeeds(sliceNumber):
    slice = inverseImage[sliceNumber]
    periodicSlice, cropXStart, cropXEnd, cropYStart, cropYEnd = generatePeriodicSlice(slice)

    flatDistances, distances, mean, meanSquared, std = generateDistanceStats(periodicSlice, cropXStart, cropXEnd, cropYStart, cropYEnd, False, False)
    # distances are in um so we need to convert to metres
    distances = distances * (10**-6)
    fibreFrac = calcFibreFrac(slice)

    averageVelocity = calcAvgVelocityInLayer(flowSpeed, fibreFrac)
    print('fibre fraction is:', fibreFrac, 'average velocity is:', averageVelocity)

    # 2d array of square distances
    distancesSqr = distances ** 2
    averageOfDistanceSquared = np.mean(distancesSqr)

    # calulates u_vox
    u_vox = averageVelocity * (distancesSqr / averageOfDistanceSquared)
    meanU = np.mean(u_vox)
    print('average of u_vox is', meanU, 'input average velocity was', averageVelocity)
    stdU = np.std(u_vox)

    statsData = {
        'distances': distances,
        'u_vox': u_vox
    }

    return statsData, meanU, stdU

# calculates average flow velocity depending on fraction of fibres in image
def calcAvgVelocityInLayer(avgSpeed, fracFibre): 
    denominator = 1 - fracFibre
    return avgSpeed/denominator

# calculates fibre fraction - fraction of image made up of fibre pixels
def calcFibreFrac(slice):
    flatSlice = slice.flatten()
    pts = np.where(slice == 0)  
    nFibrePixels = len(pts[0])
    fracFibres = nFibrePixels/len(flatSlice)
    return fracFibres

def generatePeriodicSlice(slice):
    print('creating periodic slice')
    print('original slice has shape', slice.shape)

    # length of slice in x and y
    xLength = slice.shape[0]
    yLength = slice.shape[1]

    # define new periodic slice which will be double in size in both axes
    periodicSlice = np.zeros([2*xLength, 2*yLength], dtype='uint8')

    # split into lengths of four equal quarters of slice
    x1 = int(np.ceil(xLength/2))
    x2 = int(np.floor(xLength/2) + x1)
    y1 = int(np.ceil(yLength/2))
    y2 = int(np.floor(yLength/2) + y1)

    # define arrays which are the 4 quarters of the slice
    # slices in python are like so 0:d_i translates to 0 <= n_i < d_i
    bottomLeft = slice[0:x1, 0:y1]
    bottomRight = slice[x1:x2, 0:y1]
    topLeft = slice[0:x1, y1:y2]
    topRight = slice[x1:x2, y1:y2]

    # create coordinate set for a 4x4 grid made up of the four quarters
    X1 = x2-x1
    X2 = X1 + x1
    X3 = X2 + (x2-x1)
    X4 = X3 + x1
    Y1 = y2-y1
    Y2 = Y1 + y1
    Y3 = Y2 + (y2-y1)
    Y4 = Y3 + y1

    # create periodic slice made up from the quarters

    periodicSlice[0:X1, 0:Y1] = topRight
    periodicSlice[X1:X2, 0:Y1] = topLeft
    periodicSlice[X2:X3, 0:Y1] = topRight
    periodicSlice[X3:X4, 0:Y1] = topLeft

    periodicSlice[0:X1, Y1:Y2] = bottomRight
    periodicSlice[X1:X2, Y1:Y2] = bottomLeft
    periodicSlice[X2:X3, Y1:Y2] = bottomRight
    periodicSlice[X3:X4, Y1:Y2] = bottomLeft

    periodicSlice[0:X1, Y2:Y3] = topRight
    periodicSlice[X1:X2, Y2:Y3] = topLeft
    periodicSlice[X2:X3, Y2:Y3] = topRight
    periodicSlice[X3:X4, Y2:Y3] = topLeft

    periodicSlice[0:X1, Y3:Y4] = bottomRight
    periodicSlice[X1:X2, Y3:Y4] = bottomLeft
    periodicSlice[X2:X3, Y3:Y4] = bottomRight
    periodicSlice[X3:X4, Y3:Y4] = bottomLeft

    fig = go.Figure()
    pos = np.where(periodicSlice == 0)
    fig.add_trace(go.Scatter(x=pos[0], y=pos[1], mode='markers'))
    fig.add_trace(go.Scatter(x=[0, X4], y=[Y1,Y1], line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[0,X4], y=[Y3, Y3], line=dict(dash='dash'))) 
    fig.add_trace(go.Scatter(x=[X1, X1], y=[0, Y4], line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[X3, X3], y=[0, Y4], line=dict(dash='dash')))     
    fig['layout']['yaxis']['autorange']= "reversed"
    fig.show()

    # pos = np.where(slice == 0)
    # fig = px.scatter(x=pos[1], y=pos[0])
    # fig['layout']['yaxis']['autorange']= "reversed"
    # fig.show()

    # originalSlice = periodicSlice[X1:X3, Y1:Y3]
    # print('original slice shape is', originalSlice.shape, 'which should be the same as', slice.shape)
    # pos = np.where(originalSlice == 0)
    # fig = px.scatter(x=pos[1], y=pos[0])
    # fig['layout']['yaxis']['autorange']= "reversed"
    # fig.show()

    return periodicSlice, X1, X3, Y1, Y3


# UNCOMMENT TO CALCULATE CHANNEL WIDTH AT EVERY SLICE AND PLOT
# looking at channel width at every slice
# for i in range(nSlices):
#     print(i, 'of', nSlices-1)
#     slice = inverseImage[i]
#     flatSlice = slice.flatten()

#     flatDistances, distances, mean, meanSquared, std = generateDistanceStats(slice)
#     # u_vox, meanU, stdU = generateLocalFlowSpeeds(flatDistances)

#     fracFibres = calcFibreFrac(slice)

#     statsArray[0, i] = i
#     statsArray[1, i] = mean
#     statsArray[2, i] = meanSquared
#     statsArray[3, i] = std
#     statsArray[4, i] = i * pixelSize
#     statsArray[5, i] = fracFibres

# statsData = {
#     'sliceNumber': statsArray[0],
#     'mean': statsArray[1],
#     'meanSqr': statsArray[2],
#     'std': statsArray[3],
#     'depth': statsArray[4],
#     'fibreFrac': statsArray[5]
# }

# channelWidthDF = pd.DataFrame(data = statsData)
# fig = px.line(
#         channelWidthDF, 
#         x='depth', 
#         y='mean', 
#         title='Mean channel width of fibres as a function of depth (z)',
#         labels={
#                      "depth": "Mask Depth, Z-height, (um)",
#                      "mean": "Mean Channel Width (um)"
#         },
#     )
# fig.show()

# figFracVsChannel = make_subplots(specs=[[{"secondary_y": True}]])
# figFracVsChannel.add_trace(
#     go.Scatter(x=statsData['depth'], y=statsData['mean'], name='Mean channel width of  fibres as a function of depth (z)'),
#     secondary_y=False
# )

# figFracVsChannel.add_trace(
#     go.Scatter(x=statsData['depth'], y=statsData['fibreFrac'], name='Fraction of image taken up by fibres as a function of depth (z)'),
#     secondary_y=True
# )

# figFracVsChannel.update_layout(
#     title_text="Comparison of fibre fraction to channel width"
# )

# figFracVsChannel.update_xaxes(title_text="Mask Depth, Z-height, (um)")

# figFracVsChannel.update_yaxes(title_text="Mean Channel Width (um)", secondary_y=False)
# figFracVsChannel.update_yaxes(title_text="Fibre fraction", secondary_y=True)

# figFracVsChannel.show()


# now considering slices at 300 and 500 i.e. spots where we've noticed different things

stats300Data, meanU300, stdU300 = generateLocalFlowSpeeds(299)
#stats500Data, meanU500, stdU500 = generateLocalFlowSpeeds(499)
print('property, min, max, mean, std')
print('u_vox300', np.min(stats300Data['u_vox']), np.max(stats300Data['u_vox']), np.mean(stats300Data['u_vox']), np.std(stats300Data['u_vox']))
#print('u_vox500', np.min(stats500Data['u_vox']), np.max(stats500Data['u_vox']), np.mean(stats500Data['u_vox']), np.std(stats500Data['u_vox']))


# UNCOMMENT FOR HISTOGRAMS OF COMPARISON BETWEEN SLICE 300 and SLICE 500
# figSliceComparison = go.Figure()
# figSliceComparison.add_trace(go.Histogram(
#                     x=stats300Data['distances'],
#                     name='Slice 300 (84um) nearest fibre distance',
#                     opacity=0.8,
#                     nbinsx=20
#                 ))

# figSliceComparison.add_trace(go.Histogram(
#                     x=stats500Data['distances'],
#                     name='Slice 500 (140um) nearest fibre distance',
#                     opacity=0.8,
#                     nbinsx=20
#                 ))

# figSliceComparison.update_layout(
#                     barmode='overlay',
#                     title_text='Histogram Comparison of slice 300 and 500 distances to the nearest fibre', # plot title
#                     xaxis_title_text='Distance (um)',
#                     yaxis_title_text='Count',
#                     # xaxis=dict(
#                     #     tickmode='linear',
#                     #     tick0=0.0,
#                     #     dtick=10.0
#                     # )
#                     )

# figSliceComparison.show()

# figSliceComparisonU = go.Figure()
# figSliceComparisonU.add_trace(go.Histogram(
#                     x=stats300Data['u_vox'],
#                     name='Slice 300 (84um) local flow speeds (lower fibre density)',
#                     opacity=0.8,
#                     nbinsx=50
#                 ))

# figSliceComparisonU.add_trace(go.Histogram(
#                     x=stats500Data['u_vox'],
#                     name='Slice 500 (140um) local flow speeds (higher fibre density)',
#                     opacity=0.8,
#                     nbinsx=50
#                 ))

# figSliceComparisonU.update_layout(
#                     barmode='overlay',
#                     title_text='Histogram Comparison of slice 300 and 500 Flow speeds', # plot title
#                     xaxis_title_text='Local flow speeds (units)',
#                     yaxis_title_text='Count',
#                     # xaxis=dict(
#                     #     tickmode='linear',
#                     #     tick0=0.0,
#                     #     dtick=10.0
#                     # )
#                     )

# figSliceComparisonU.show()

# function to plot scatter and heatmap comparisons - used to plot fibres and the local flow around those fibres
def plotScatterHeatmapComparison(slice, heatmapData, sliceName, heatmapName, colorbarTitle):
    pos = np.where(slice == 1)
    print('slice properties,', slice.shape)
    print('len of pos[0]', len(pos[0]))
    print('heatmap properties', heatmapData.shape)

    if(areAxesFlipped):
        heatmapData=swapaxes(heatmapData, 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pos[0],
        y=pos[1],
        mode='markers',
        marker=dict(
            size=1,
        ),
        name=sliceName,
    ))

    fig.add_trace(go.Heatmap(
        z=heatmapData,
        name=heatmapName,
        colorbar=dict(
            title=colorbarTitle
        )
    ))

    fig.update_layout(
        autosize=False,
        width=600,
        height=3000
    )
    fig['layout']['yaxis']['autorange']= "reversed"

    fig.show()

plotScatterHeatmapComparison(binaryOutputs[299], stats300Data['u_vox'], 'Fibres', 'Local flow speeds', 'Local Flow Speed (m/s)')
plotScatterHeatmapComparison(binaryOutputs[299], stats300Data['distances'], 'Fibres', 'Distance to closest fibre', 'Closest fibre distance (m)')
#plotScatterHeatmapComparison(binaryOutputs[499], stats500Data['u_vox'], 'Fibres', 'Local flow speeds', 'Local Flow Speed (m/s)')
#plotScatterHeatmapComparison(binaryOutputs[499], stats500Data['distances'], 'Fibres', 'Distance to closest fibre', 'Closest fibre distance (m)')