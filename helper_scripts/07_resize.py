import os
import glob
import cv2
import numpy as np
import tqdm
import skimage.io


def main():
    indir = '/Users/beichenberger/Downloads/Nuclei_train/train_val'
    outdir= '/Users/beichenberger/Downloads/Stardist_training/'

    img_files = sorted(glob.glob(f'{indir}/images/*.tif'))
    mask_files = sorted(glob.glob(f'{indir}/masks/*tif'))

    # Images
    for i in tqdm.tqdm(img_files):
        out_name = (i.split('/')[-1]).split('.')[0]
        img = skimage.io.imread(i)
        #img = skimage.transform.resize(img, (512, 512), preserve_range=True)
        skimage.io.imsave(f"{outdir}/images/{out_name}.tif", img.astype(dtype=np.uint16), check_contrast=False)

    # Masks
    for i in tqdm(mask_files):
        out_name = (i.split('/')[-1]).split('.')[0]
        mask = skimage.io.imread(i)
        #mask = cv2.resize(mask, (512, 512))
        skimage.io.imsave(f"{outdir}/masks/{out_name}.tif", mask.astype(dtype=np.uint16), check_contrast=False)

if __name__ == "__main__":
    main()


