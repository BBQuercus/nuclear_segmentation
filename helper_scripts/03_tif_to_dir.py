import os
import glob
import shutil
from skimage import io

files = glob.glob('*.tif')

for f in files:
    n = f.split('.')[0]
    os.mkdir(f'{n}')
    os.mkdir(f'{n}/images/')
    os.mkdir(f'{n}/masks/')
    shutil.copy(f, f'{n}/images/{f}')
