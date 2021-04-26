## Proof of concept of normalising image intensity throughout an image
import numpy as np
from skimage import io  # scikit-image
import tifffile

imageFilename = './resources/data/FinalFused.tif'
imageStack = io.imread(imageFilename)
print(np.mean(imageStack))

stdDev = np.std(imageStack)
print('std dev', stdDev)
normalisedStack = imageStack/stdDev
print(np.mean(normalisedStack))

tifffile.imwrite('FinalFusedNormalisationPOC.tif', normalisedStack)