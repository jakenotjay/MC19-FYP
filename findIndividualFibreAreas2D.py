# calculates fibre sizes on a 2D scale going slice per slice
# plots the results as a function of slice
# improved in findLayerAreaStats2D.py
import numpy as np
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pixelSize = 1.4

# load image and convert to 8 bit binary (byte)
img = np.load('./3DImageStackBinary.npy')
img = img.astype(np.uint8)

# number of slices in the image (i.e. z-axis length)
# area array will store array of all fibre areas
# component count array simply stores number of fibres in each slice
nSlices = img.shape[0]
areaList = []
componentCountArray = np.zeros(nSlices)

for slice in range(nSlices):
    # get slice and find number of individual fibres and an array of where each component is
    sliceImage = img[slice]
    nComponents, componentsArray = cv2.connectedComponents(sliceImage)

    componentCountArray[slice] = nComponents

    sliceCountArray = np.zeros(nComponents)
    sliceAreaArray = np.zeros(nComponents)

    # for each component count the number of pixels that make up that component i.e. fibre size
    for component in range(nComponents):
        sliceCountArray[component] = np.count_nonzero(componentsArray == component)

    # find area of all of these components
    sliceAreaArray = sliceCountArray * pixelSize ** 2

    areaList.append(sliceAreaArray)

# concatenate array list into single array and then flatten to 1D
areaArray = np.concatenate(areaList).flatten()
# print('areaArray shape, should be same as nComponents', areaArray.shape)
print('values range between ', np.amin(areaArray), 'um^2 and ', np.amax(areaArray), ' um^2')
print('this corresponds to ', np.amin(areaArray)/(1.4**2), ' pixels and ', np.amax(areaArray)/(1.4**2), ' pixels')
print('sum of nComponents (no. fibres)', np.sum(componentCountArray))

# assuming gaussian (probably not true, but just quick try at limiting results)
def rejectOutliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# massive outliers even if we limit to "68%" confidence interval
filteredArray = rejectOutliers(areaArray, 1)
print("rejected ", areaArray.shape[0] - filteredArray.shape[0], " values")

fig = make_subplots(rows=3, cols=1)

fig.add_traces(
    go.Histogram(
        x=areaArray,
        name='Fibre Area um^2'
    ),
    rows=1, cols=1
)

fig.add_traces(
    go.Histogram(
        x=componentCountArray,
        name='No. Fibres in a slice'
    ),
    rows=2, cols=1
)

fig.add_traces(
    go.Histogram(
        x=filteredArray,
        name='Filtered Fibre Area um^2'
    ),
    rows=3, cols=1
)

fig.show()

## This function can be used to show components in a slice, labelled with an individual
## colour, keeping for future reference
# def imshow_components(componentsArray):
#     # Map component labels to hue val
#     label_hue = np.uint8(179*componentsArray/np.max(componentsArray))
#     blank_ch = 255*np.ones_like(label_hue)
#     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

#     # cvt to BGR for display
#     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

#     # set bg label to black
#     labeled_img[label_hue==0] = 0

#     cv2.imshow('labeled.png', labeled_img)
#     cv2.waitKey()

# imshow_components(componentsArray)