import tqdm
import numpy as np
import tensorflow as tf
import skimage.morphology
import skimage.transform
import skimage.io

import stardist
import csbdeep.utils

def _add_borders(img, size=2):
    borders = np.zeros(img.shape)
    for i in np.unique(img):
        mask = np.where(img==i, 1, 0)
        mask_dil = skimage.morphology.dilation(mask, skimage.morphology.square(size))
        mask_ero = skimage.morphology.erosion(mask, skimage.morphology.square(size))
        mask_border = np.logical_xor(mask_dil, mask_ero)
        borders[mask_border==True] = i
    image = np.where(borders>0, 2, img>0)
    return image

def random_sample_generator(x_list, y_list, batch_size, bit_depth, dim_size):

    do_augmentation = True
    borders = True
    channels = 3 if borders else 1
        
    while(True):
            
        # Buffers for a batch of data
        x = np.zeros((batch_size, dim_size, dim_size, 1))
        y = np.zeros((batch_size, dim_size, dim_size, channels))
        
        # Get one image at a time
        for i in range(batch_size):
                       
            # Get random image
            img_index = np.random.randint(low=0, high=len(x_list))
            
            # Open images and normalize
            x_curr = skimage.io.imread(x_list[img_index]) * (1./(2**bit_depth - 1))
            y_curr = skimage.io.imread(y_list[img_index])

            # Get random crop
            start_dim1 = np.random.randint(low=0, high=x_curr.shape[0] - dim_size) if x_curr.shape[0]>dim_size else 0
            start_dim2 = np.random.randint(low=0, high=x_curr.shape[1] - dim_size) if x_curr.shape[1]>dim_size else 0
            patch_x = x_curr[start_dim1:start_dim1 + dim_size, start_dim2:start_dim2 + dim_size] #* rescale_factor
            patch_y = y_curr[start_dim1:start_dim1 + dim_size, start_dim2:start_dim2 + dim_size] #* rescale_factor_labels

            if borders:
                patch_y = _add_borders(patch_y)
                patch_y = tf.keras.utils.to_categorical(patch_y)

            if(do_augmentation):
                rand_flip = np.random.randint(low=0, high=2)
                rand_rotate = np.random.randint(low=0, high=4)
                
                # Flip
                if(rand_flip == 0):
                    patch_x = np.flip(patch_x, 0)
                    patch_y = np.flip(patch_y, 0)
                
                # Rotate
                for _ in range(rand_rotate):
                    patch_x = np.rot90(patch_x)
                    patch_y = np.rot90(patch_y)

                # Illumination
                illumination_factor = 1 + np.random.uniform(-0.75, 0.75)
                patch_x *= illumination_factor
                    
            # Save image to buffer
            x[i,:,:,0] = patch_x

            if borders:
                y[i, :, :, 0:channels] = patch_y
            else:
                y[i,:,:,0] = patch_y

        # Return the buffer
        yield(x, y)

def stardist_importer(x_list, y_list, axis_norm=(0, 1)):
    '''
    '''
    # Read images
    x_images = list(map(skimage.io.imread, x_list))
    y_images = list(map(skimage.io.imread, y_list))

    # Stardist reshape
    x_images = [csbdeep.utils.normalize(i, 1, 99.8, axis=axis_norm) for i in tqdm.tqdm(x_images, desc='Images: ')]
    y_images = [stardist.fill_label_holes(y) for y in tqdm.tqdm(y_images, desc='Masks: ')]

    # Unfixed stardist bugs
    x_images, y_images = zip(*[(x, y) for x, y in zip(x_images, y_images) if y.max() >= 1])

    # TODO – generalize to all unequal shapes
    x_adjusted = []
    for x, y in zip(x_images, y_images):
        if y.shape==(519, 696):
            x_adjusted.append(x[:519, :696])
        else:
            x_adjusted.append(x)
    x_images = x_adjusted

    return x_images, y_images

# Legacy

# def get_adjusted_images(imgs, size):
#     imgs = [skimage.transform.resize(i, (size, size))
#             for i in tqdm.tqdm(imgs, desc='Images: ')]
#     imgs = np.array(imgs)
#     imgs = np.expand_dims(imgs, axis=-1)
#     return imgs

# def add_borders(img, size):
#     inside = skimage.transform.resize(img, (size, size))
#     borders = get_borders(inside)>0
#     mask = np.where(borders==True, 2, inside>0)
#     return mask

# def get_adjusted_masks(imgs, size, add_borders=False):
#     imgs = [add_borders(i, size)
#             if add_borders
#             else skimage.transform.resize(i, (size, size))>0
#             for i in tqdm.tqdm(imgs, desc='Masks: ')]
#     imgs = np.array(imgs)
#     imgs = np.expand_dims(imgs, axis=-1)
#     return imgs

# def get_borders(img, size=2):
#     out = np.zeros(img.shape)
#     for i in np.unique(img):
#         mask = np.where(img==i, 1, 0)
#         mask_dil = skimage.morphology.dilation(mask, skimage.morphology.square(size))
#         mask_ero = skimage.morphology.erosion(mask, skimage.morphology.square(size))
#         border = np.logical_xor(mask_dil, mask_ero)
#         out[border==True] = i
#     return out