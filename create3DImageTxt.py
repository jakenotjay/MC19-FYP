import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tifffile import imread
import numpy as np

# load image as pixel array
image = imread('./For_Richard_New/cropped-1_4umperpixel-binary.tif')
# summarize shape of pixel array
print(image.dtype)
print(image.shape)

# image[z, x, y]
x = image[0:, 0, 0:]
y = image[0:, 0:, 0]
z = image[0, 0:, 0:]
print(x.shape)
print(y.shape)
print(z.shape)

# transpose planes so they are right way round
x = np.transpose(x)
y = np.transpose(y)

# distance between each "pixel"
d = 1.4

# def generatePlaneCoords(plane):
#     for i in range(len(plane)):
#         for j in range(len(plane[i])):
#             plane[i, j] = (i + j) * d

#     return plane

# x = generatePlaneCoords(x)
# y = generatePlaneCoords(y)
# z = generatePlaneCoords(z)

def extendPlaneShapes(x, y, z):
    maxAxisLength = max([max(x.shape), max(y.shape), max(z.shape)])
    
    def extendPlanes(plane):
        squarePlane = np.zeros((maxAxisLength, maxAxisLength))
        for i in range(len(plane)):
            for j in range(len(plane[i])):
                squarePlane[i, j] += plane[i, j]

        return squarePlane
    
    x = extendPlanes(x)
    y = extendPlanes(y)
    z = extendPlanes(z)

    return x, y, z

x, y, z = extendPlaneShapes(x, y, z)

print("extended shapes")
print(x.shape)
print(y.shape)
print(z.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")
plt.show()