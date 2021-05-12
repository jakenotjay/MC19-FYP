# Proof of concept 3D connected components
import cc3d
import numpy as np
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# load image
img = np.load('./3DImageStackBinary.npy')
img = img.astype(np.uint8)

# finding all connected components in 3D
# only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
connectivity = 6

labelsOut, N = cc3d.connected_components(img, return_N=True, connectivity=connectivity)
print(labelsOut)

print("labels shape ", labelsOut.shape)

print(N, " components")

labelsOutPos = np.where(labelsOut == 222)

fig = go.Figure(data=[
    go.Scatter3d(
        x=labelsOutPos[1],
        y=labelsOutPos[2],
        z=labelsOutPos[0],
        mode='markers',
        marker=dict(
            size=1
        )
    )
])

fig.show()

# canny edge detection
# print considering single image first
# slice = img[0]
# edges = cv2.Canny(slice, 100, 200)
# edgesPos = np.where(edges == 1)
# slicesPos = np.where(slice == 1)

# print(edges.shape)
# print(edges)

# fig = make_subplots(rows = 2, cols = 1)

# fig.add_traces(
#     go.Scatter(
#         x=slicesPos[0],
#         y=slicesPos[1],
#         name='Original slice'
#     ),
#     rows=1, cols=1
# )

# fig.add_traces(
#     go.Scatter(
#         x=edgesPos[0],
#         y=edgesPos[1],
#         name='Slice edges'
#     ),
#     rows=2, cols=1
# )

# fig.show()