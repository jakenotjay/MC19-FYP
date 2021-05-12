# LEGACY - DO NOT USE
# attempts to plot data as a 3D scatter in plotly
# use generate3DTif.py and instead view in imageJ
import numpy as np
import plotly.graph_objects as go


# load image as pixel array
zipfile = np.load('./ImageStackBinaryComponentsBigFused.npz')
image = zipfile['binaryOut']
# summarize shape of pixel array
imageShape = image.shape
print(imageShape)

# distance between each "pixel"
d = 0.28

# image[z, x, y]
# nSlices = 5
# sliceJump = np.rint(imageShape[0] / nSlices)
# slices = np.zeros(shape = (nSlices, imageShape[1], imageShape[2]))

# for i in range(nSlices):
#     index = int(i * sliceJump) 
#     print(index)
#     slices[i] = image[index]


pos = np.where(image == 1)
print(pos)
print(pos[0].shape)
print(pos[1].shape)
print(pos[2].shape)

fig = go.Figure(data=[go.Scatter3d(
    x = pos[1],
    y = pos[2],
    z = pos[0],
    mode='markers',
    marker=dict(
        size=1,
    )
)])

fig.show()

