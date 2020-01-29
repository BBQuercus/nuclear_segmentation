import os
import glob
import numpy as np

import tqdm
import skimage.io

# Generate image / mask folder to label images
root = '/Users/beichenberger/Files/Labeling/Granules/DM_SIM_G3BP-SNAP/'
files = glob.glob(f'{root}*.tif')

for f in tqdm.tqdm(files):
    path = f.split('.')[0]
    os.mkdir(path)
    os.mkdir(f'{path}/images/')
    os.mkdir(f'{path}/masks/')
    
    img = skimage.io.imread(f)
    skimage.io.imsave(f"{path}/images/{path.split('/')[-1]}.tif", img.astype(dtype=np.uint16))
