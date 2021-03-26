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

# calculate max possible distance sqr(x^2 + y^2)
maxDistance = np.sqrt(inverseImage.shape[1]**2 + inverseImage.shape[2]**2)
print('max possible distance', maxDistance)

# sliceNumber, mean, mean^2, stdDev, depth, fibreFrac
statsArray = np.zeros([6, nSlices])

# pixel size in micrometres
pixelSize = 0.28

def generateDistanceStats(slice):
    # for all zeros (air pixels) find nearest 1 pixel (fibres)
    distances = cv2.distanceTransform(slice, cv2.DIST_L2, cv2.DIST_MASK_5)

    flatDistances = distances.flatten()

    # get rid of fibres (0) and those that are impossibly large
    maxFilterArray = flatDistances < maxDistance
    flatDistances = flatDistances[maxFilterArray]
    minFilterArray = flatDistances > 0
    flatDistances = flatDistances[minFilterArray]

    # find distance, mean, mean squared and std
    flatDistances = flatDistances * pixelSize
    mean = np.mean(flatDistances)
    meanSquared = mean ** 2
    std = np.std(flatDistances)

    return flatDistances, mean, meanSquared, std

def generateLocalFlowSpeeds(distances):
    distancesSqr = distances ** 2

    u_vox = (5 * 10**-3) * distancesSqr
    meanU = np.mean(u_vox)
    stdU = np.std(u_vox)

    return u_vox, meanU, stdU

# looking at channel width at different slices
for i in range(nSlices):
    print(i, 'of', nSlices-1)
    slice = inverseImage[i]
    flatSlice = slice.flatten()

    flatDistances, mean, meanSquared, std = generateDistanceStats(slice)
    # u_vox, meanU, stdU = generateLocalFlowSpeeds(flatDistances)

    pts = np.where(slice == 0)
    nFibrePixels = len(pts[0])
    fracFibres = nFibrePixels/len(flatSlice)

    statsArray[0, i] = i
    statsArray[1, i] = mean
    statsArray[2, i] = meanSquared
    statsArray[3, i] = std
    statsArray[4, i] = i * pixelSize
    statsArray[5, i] = fracFibres

statsData = {
    'sliceNumber': statsArray[0],
    'mean': statsArray[1],
    'meanSqr': statsArray[2],
    'std': statsArray[3],
    'depth': statsArray[4],
    'fibreFrac': statsArray[5]
}

channelWidthDF = pd.DataFrame(data = statsData)
fig = px.line(
        channelWidthDF, 
        x='depth', 
        y='mean', 
        title='Mean channel width of fibres as a function of depth (z)',
        labels={
                     "depth": "Mask Depth, Z-height, (um)",
                     "mean": "Mean Channel Width (um)"
        },
    )
fig.show()

figFracVsChannel = make_subplots(specs=[[{"secondary_y": True}]])
figFracVsChannel.add_trace(
    go.Scatter(x=statsData['depth'], y=statsData['mean'], name='Mean channel width of  fibres as a function of depth (z)'),
    secondary_y=False
)

figFracVsChannel.add_trace(
    go.Scatter(x=statsData['depth'], y=statsData['fibreFrac'], name='Fraction of image taken up by fibres as a function of depth (z)'),
    secondary_y=True
)

figFracVsChannel.update_layout(
    title_text="Comparison of fibre fraction to channel width"
)

figFracVsChannel.update_xaxes(title_text="Mask Depth, Z-height, (um)")

figFracVsChannel.update_yaxes(title_text="Mean Channel Width (um)", secondary_y=False)
figFracVsChannel.update_yaxes(title_text="Fibre fraction", secondary_y=True)

figFracVsChannel.show()


# now considering slices at 300 and 500 i.e. spots where we've noticed different things
slice300 = inverseImage[299]
slice500 = inverseImage[499]

flatDistances300, mean300, meanSquared300, std300 = generateDistanceStats(slice300)
flatDistances500, mean500, meanSquared500, std500 = generateDistanceStats(slice500)
u_vox300, meanU300, stdU300 = generateLocalFlowSpeeds(flatDistances300)
u_vox500, meanU500, stdU500 = generateLocalFlowSpeeds(flatDistances500)

stats300Data = {
    'flat_distances': flatDistances300,
    'u_vox': u_vox300
}
stats500Data = {
    'flat_distances': flatDistances500,
    'u_vox': u_vox500
}

figSliceComparison = go.Figure()
figSliceComparison.add_trace(go.Histogram(
                    x=stats300Data['flat_distances'],
                    name='Slice 300 (84um) channel widths (lower fibre density)',
                    opacity=0.8,
                    nbinsx=50
                ))

figSliceComparison.add_trace(go.Histogram(
                    x=stats500Data['flat_distances'],
                    name='Slice 500 (140um) channel widths (higher fibre density)',
                    opacity=0.8,
                    nbinsx=50
                ))

figSliceComparison.update_layout(
                    barmode='overlay',
                    title_text='Histogram Comparison of slice 300 and 500 Channel Widths', # plot title
                    xaxis_title_text='Channel Width (um)',
                    yaxis_title_text='Count',
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0.0,
                        dtick=10.0
                    )
                    )

figSliceComparison.show()

figSliceComparisonU = go.Figure()
figSliceComparisonU.add_trace(go.Histogram(
                    x=stats300Data['u_vox'],
                    name='Slice 300 (84um) local flow speeds (lower fibre density)',
                    opacity=0.8,
                    nbinsx=50
                ))

figSliceComparisonU.add_trace(go.Histogram(
                    x=stats500Data['u_vox'],
                    name='Slice 500 (140um) local flow speeds (higher fibre density)',
                    opacity=0.8,
                    nbinsx=50
                ))

figSliceComparisonU.update_layout(
                    barmode='overlay',
                    title_text='Histogram Comparison of slice 300 and 500 Channel Widths', # plot title
                    xaxis_title_text='Local flow speeds (units)',
                    yaxis_title_text='Count',
                    # xaxis=dict(
                    #     tickmode='linear',
                    #     tick0=0.0,
                    #     dtick=10.0
                    # )
                    )

figSliceComparisonU.show()

# filename = './' + 'channelWidth2D' + '.pkl'
# channelWidthDF.to_pickle(filename)