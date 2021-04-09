# finds number of pixels in a fibre and finds volume of fibres
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy.ndimage.measurements import label

zipFile = np.load('./ImageStackFULL.npz')
labelledOutputs = np.asarray(zipFile['labelledOut'], dtype='uint16')

maxN = np.amax(labelledOutputs)
print("There are ", maxN, " components")

pixelSize = 0.28

# 3D DATA STATS
NPixels = np.zeros(maxN)
NVolume = np.zeros(maxN)

for i in range(1, maxN+1):
    print("Considering component ", i )

    pts = np.where(labelledOutputs == i)
    nPts = len(pts[0])
    vol = nPts * (pixelSize ** 2)

    NPixels[i-1] = nPts
    NVolume[i-1] = vol
    
volumeData = {
    'noPixels': NPixels,
    'volume': NVolume
}    

volumeDataFrame = pd.DataFrame(data = volumeData)

fig = make_subplots(rows=2, cols=1)
fig.add_traces(
    go.Histogram(
        x=volumeDataFrame['noPixels'],
        name='number of pixels per component'
    ),
    rows=1, cols=1
)
fig.add_traces(
    go.Histogram(
        x=volumeDataFrame['volume'],
        name='volume of components'
    ),
    rows=2, cols=1
)

fig.show()

filename = './' + '3DstatsFULLSTACK' + '.pkl'
volumeDataFrame.to_pickle(filename)