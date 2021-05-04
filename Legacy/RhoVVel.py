import numpy as np
import matplotlib.pyplot as plt

# loading numpy zip file
input_file = './tmp.npz'
npz_file = np.load(input_file, allow_pickle=True)

# loading image array
image_array = np.loadtxt('Ioatzin_image.txt')
print(image_array.shape)

# ouputs of pickled files
print(npz_file.files)

# density
rho = npz_file['rho']
# velocity
u_vec = npz_file['u_vec']

print(rho.shape)
print(u_vec.shape)

# calc velocity magnitude
u = np.sqrt(u_vec[0] ** 2 + u_vec[1] ** 2)

def calcAverageRowValues(twoDArray):
    rows = np.zeros([len(twoDArray)])

    for i in range(len(twoDArray)):
        nFreeSpaceColumns = 0

        for j in range(len(twoDArray[i])):
            if(image_array[i, j] == 0):
                nFreeSpaceColumns += 1
                rows[i] += twoDArray[i, j]
        
        # average row value
        rows[i] = rows[i] / nFreeSpaceColumns

    return rows

avg_rho = calcAverageRowValues(rho)
avg_u = calcAverageRowValues(u)

plt.plot(avg_rho, avg_u)
plt.xlabel("Average density")
plt.ylabel("Average velocity")
plt.show()