import glob
import numpy as np
import skimage.io

root = '/Users/beichenberger/Files/Labeling/Granules/DM_SIM_G3BP-SNAP/'
files = glob.glob(f'{root}*.tiff')

for f in files:
    i = skimage.io.imread(f)
    i = np.max(i, axis=0)
    #for j in range(3):
    skimage.io.imsave(f"{f.split('.')[0]}.tif", i)
