import os
import glob
import numpy as np
import tqdm
import skimage.io
import skimage.measure
import skimage.transform


def import_masks_bbbc(path):
    '''Imports masks in the format of BBBC labels.
    0 BG, 1 FG, 1++ Touching FG(s) per image.
    '''
    #masks = skimage.io.imread(path)[0:512,0:512,0]
    masks = skimage.io.imread(path)[:,:,0]
    masks_unique = [np.where(masks==i, 1, 0) for i in range(1, np.max(masks)+1)]
    masks_unique = [skimage.measure.label(i)[1:] for i in masks_unique]
    
    counter = 0
    masks_ = []
    for i in masks_unique:
        if counter >= np.min(np.min(i[np.nonzero(i)])):
            i = np.where(i!=0, i+counter+1, 0)
        masks_.append(i)
        counter += len(np.unique(i)[1:])
        
    if len(masks_) == 0:
        return np.zeros((masks.shape[0], masks.shape[1]))
    
    return np.max(masks_, axis=0)

def import_masks_fiji(path):
    '''Imports masks in the format of DSB2018 / FIJI labels.
    Unique files for each label as png.
    '''
    masks = [skimage.io.imread(path + '/masks/' + mask_file)
            for mask_file in next(os.walk(path + '/masks/'))[2]
            if mask_file.endswith('png')]
    masks = [np.where(m, i, 0) for i, m in enumerate(masks)]
    masks = np.max(masks, axis=0)
    return masks

def import_image(path):
    img = skimage.io.imread(f'{path}/images/{id_}.tif')
    img = img[:,:,0] if img.ndim==3 else img

def main():
    indir = '/Users/beichenberger/Documents/Github/nuclear_segmentation/data/BBBC039'
    outdir = '/Users/beichenberger/Documents/Github/nuclear_segmentation/data/test_new'
    fiji = False
    bbbc = True
    
    if fiji:
        indir_ids = next(os.walk(indir))[1]

        for n, id_ in tqdm.tqdm(enumerate(indir_ids), total=len(indir_ids)):
            path = indir + id_
            
            # Images
            img = skimage.io.imread(f'{path}/images/{id_}.tif')
            img = img[:,:,0] if img.ndim==3 else img
            skimage.io.imsave(f'{outdir}/images/{id_}.tif', img.astype(dtype=np.uint16), check_contrast=False)
            
            # Masks
            mask = import_masks_fiji(path)
            skimage.io.imsave(f'{outdir}/masks/{id_}.tif', mask.astype(dtype=np.uint16), check_contrast=False)

    if bbbc:
        img_files = sorted(glob.glob(f'{indir}/images/*.tif'))
        mask_files = sorted(glob.glob(f'{indir}/masks/*.png'))

        for i in tqdm.tqdm(range(len(img_files))):
            out_name = (img_files[i].split('/')[-1]).split('.')[0]

            # Images
            img = skimage.io.imread(img_files[i])
            img = img[:,:,0] if img.ndim==3 else img[:519,:]
            skimage.io.imsave(f"{outdir}/images/{out_name}.tif", img.astype(dtype=np.uint16), check_contrast=False)

            # Masks
            mask = import_masks_bbbc(mask_files[i])
            skimage.io.imsave(f"{outdir}/masks/{out_name}.tif", mask.astype(dtype=np.uint16), check_contrast=False)
            

if __name__ == "__main__":
    main()
