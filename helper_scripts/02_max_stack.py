import os
import glob
import numpy as np

import skimage.io
import tqdm

files = glob.glob(f'{os.getcwd()}/*488.stk')

for f in tqdm.tqdm(files):
    i = skimage.io.imread(f)
    i = np.max(i, axis=0)
    skimage.io.imsave(f"{f.split('.')[0]}.tif", i)
