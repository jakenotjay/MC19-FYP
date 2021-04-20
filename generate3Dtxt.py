# code to convert npz to one line txt file for palabos
import numpy as np

zipFile = np.load('./outputs/npz/FinalFusedThresh30.npz')
binaryOutputs = np.asarray(zipFile['binaryOut'], dtype='bool')

nx = binaryOutputs.shape[1]
ny = binaryOutputs.shape[2]
nz = binaryOutputs.shape[0]
print('dimensions of image are', nx, ny, nz)

# need to do it this way because of dimension order in array (z, x, y)
palabosOutput = np.zeros(nx * ny * nz)
count = -1
for ix in range(0, nx):
    for iy in range(0, ny):
        for iz in range(0, nz):
            count+=1
            if(binaryOutputs[iz, ix, iy]==1):
                palabosOutput[count] = 1

print(len(palabosOutput))
print(np.mean(palabosOutput))
print(binaryOutputs.shape)
print(np.mean(binaryOutputs))

filename = './outputs/txt/'+'PalabosFinalFused.txt'
np.savetxt(filename, np.ndarray.flatten(np.transpose(palabosOutput)), fmt='%2d', newline=" ")