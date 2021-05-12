# WIP showing marching cubes algorithm working on voxels
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

zipFile = np.load('./ImageStackFULL.npz')
binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool')
print('binary shape', binaryOutputs.shape)
testOut = binaryOutputs[400:, 0:500, :]

verts, faces, normals, values = measure.marching_cubes(testOut, 0)
print('verts', verts)
print('faces', faces)
print('normals', normals)
print('values', values)

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

plt.tight_layout()
plt.show()