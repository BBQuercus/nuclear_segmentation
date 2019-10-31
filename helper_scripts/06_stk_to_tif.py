import glob
import numpy as np
import skimage.io

files = glob.glob('*561dual-triple.stk')

for f in files:
    i = skimage.io.imread(f)
    skimage.io.imsave(f"{f.split('.')[0]}.tif", i[0])
