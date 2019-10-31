import glob
import numpy as np
import skimage.io

files = glob.glob('*.tiff')

for f in files:
    i = skimage.io.imread(f)
    i = np.max(i, axis=-1)
    for j in range(3):
        skimage.io.imsave(f"{f.split('.')[0]}_{j}.tif", i[0,0,j])
