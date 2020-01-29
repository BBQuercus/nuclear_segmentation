import glob
import numpy as np
import skimage.io

root = '/Users/beichenberger/Files/Labeling/Granules/DM_Franklin200x_G3BP-SNAP/'
files = glob.glob(f'{root}*.stk')

for f in files:
    i = skimage.io.imread(f)
    i = np.max(i, axis=0)
    skimage.io.imsave(f"{f.split('.')[0]}.tif", i)
