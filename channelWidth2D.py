# calculates channel width estimates based on fibre distances
# calculates local flow speed based on fibre distances
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

zipFile = np.load('./ImageStackFULL.npz')
binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool')

# inverse image i.e 1 to 0, 0 to 1
inverseImage = np.array(np.invert(binaryOutputs), dtype='uint8')
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
def generateDistanceStats(slice, filterMin=True, filterMax=True):
    # for all zeros (air pixels) find nearest 1 pixel (fibres)
    # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga25c259e7e2fa2ac70de4606ea800f12f
    distances = cv2.distanceTransform(slice, cv2.DIST_L2, cv2.DIST_MASK_5)
    
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

    flatDistances, distances, mean, meanSquared, std = generateDistanceStats(slice, False, False)
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
stats500Data, meanU500, stdU500 = generateLocalFlowSpeeds(499)
print('property, min, max, mean, std')
print('u_vox300', np.min(stats300Data['u_vox']), np.max(stats300Data['u_vox']), np.mean(stats300Data['u_vox']), np.std(stats300Data['u_vox']))
print('u_vox500', np.min(stats500Data['u_vox']), np.max(stats500Data['u_vox']), np.mean(stats500Data['u_vox']), np.std(stats500Data['u_vox']))


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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pos[1],
        y=pos[0],
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

    fig.show()

plotScatterHeatmapComparison(binaryOutputs[299], stats300Data['u_vox'], 'Fibres', 'Local flow speeds', 'Local Flow Speed (m/s)')
plotScatterHeatmapComparison(binaryOutputs[299], stats300Data['distances'], 'Fibres', 'Distance to closest fibre', 'Closest fibre distance (m)')
plotScatterHeatmapComparison(binaryOutputs[499], stats500Data['u_vox'], 'Fibres', 'Local flow speeds', 'Local Flow Speed (m/s)')
plotScatterHeatmapComparison(binaryOutputs[499], stats500Data['distances'], 'Fibres', 'Distance to closest fibre', 'Closest fibre distance (m)')