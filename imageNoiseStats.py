## imageNoiseStats.py
# generates stats on the number of noise pixels per layer
# generates stats on the fraction of noise pixels to pixels per layer
import numpy as np
import cv2
from skimage import io
import pandas as pd

imageFilename = './resources/data/Fused3103.tif'

imageStack = io.imread(imageFilename)
print(imageStack.shape[0], ' slices in z stack each ',
      imageStack.shape[1], ' by ', imageStack.shape[2], ' pixels ')

nSlices = imageStack.shape[0]
nNoiseSlice = []
noiseFracSlice = []
nonZeroPixelFracSlice = []
sliceN = []
meanNoiseIntensity = []
nonZeroPixelsSlice = []

for slice in range(nSlices):
    sliceN.append(slice+1)
    print('on slice', slice)
    sliceImage = imageStack[slice]
    nComponents, componentsArray = cv2.connectedComponents(sliceImage)
    print('there are', nComponents, 'components')
    nNoise = 0
    noisePixelIntensity = []

    # calculates number of noise pixels and intensity of that noise
    for component in range(1, nComponents+1):
        # print('considering component', component)
        pts = np.where(componentsArray == component)
        if(len(pts[0]) == 1):
            nNoise +=1
            noisePixelIntensity.append(sliceImage[pts])

    print('max noise intensity', np.max(noisePixelIntensity))
    print('min noise intensity', np.min(noisePixelIntensity))
    print('number of noise particle', nNoise)

    ## pixelFractionCalculations
    # calculate number of pixels with intensity > 0
    nNonZeroPixels = np.count_nonzero(sliceImage)
    nonZeroPixelsSlice.append(nNonZeroPixels)
    totalPixels = sliceImage.shape[0] * sliceImage.shape[1]
    fracNonZeroPixels = nNonZeroPixels / totalPixels
    # calculate that as a fraction of total number of pixels x * y
    # use the number of pixels with intensity to calculate noiseFracSlice

    nonZeroPixelFracSlice.append(fracNonZeroPixels)
    nNoiseSlice.append(nNoise)
    fracNoisePixelsToNonZeroPixels = nNoise / nNonZeroPixels
    noiseFracSlice.append(fracNoisePixelsToNonZeroPixels)
    meanNoiseIntensity.append(np.mean(noisePixelIntensity))

noiseData = {
    'sliceN': sliceN,
    'nNoisePixels': nNoiseSlice,
    'noiseFracSlice': noiseFracSlice,
    'meanNoiseIntensity': meanNoiseIntensity,
    'nonZeroPixelFracSlice': nonZeroPixelFracSlice,
    'nonZeroPixelsSlice': nonZeroPixelsSlice
}

df = pd.DataFrame(data=noiseData)
print(df)
df.to_pickle('./outputs/pkl/Fused3103Full.pkl')

