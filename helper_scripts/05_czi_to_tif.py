import os, glob
import numpy as np
import skimage.io
import czifile

files = glob.glob('*.czi')

for f in files:
    i = czifile.imread(f)
    i = np.max(i, axis=2)
    skimage.io.imsave(f"{f.split('.')[0]}.tif", i[0,2,:,:,0])

