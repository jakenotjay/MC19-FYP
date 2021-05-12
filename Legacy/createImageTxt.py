# LEGACY code used to create a txt file to process
from matplotlib import image, pyplot
from numpy import asarray, savetxt

# load image as pixel array
image = image.imread('XZ-plane_1.png')
# summarize shape of pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()

# convert image to numpy array
image_array = asarray(image, dtype=int)
print(type(image_array))
# summarise shape
print(image_array.shape)

print(image_array)

print(image_array[0, 0])
filename = 'Ioatzin_image.txt'
savetxt(filename, image_array, fmt='%2d')