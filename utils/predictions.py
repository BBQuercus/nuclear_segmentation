import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tqdm
import skimage
import csbdeep.utils
import stardist

def _otsu_single(img):
    '''
    '''
    return img>skimage.filters.threshold_otsu(img)

def otsu_batch(imgs):
    '''
    '''
    return [_otsu_single(img) for img in imgs]

def _unet_single(model, img, base=256, bit_depth=16):
    '''
    '''
    pred_img = img * (1./(2**bit_depth - 1))

    # TODO – find sliding window alternative
    # if not img.shape[0]%base==0 and not img.shape[1]%base==0:
    #     def __round(x, base):
    #         return base * round(x/base)

    #     dim = max([__round(i, base) for i in img.shape])
    #     pred_img = skimage.transform.resize(img, (dim, dim), mode='constant', preserve_range=True)

    pred_img = skimage.transform.resize(img, (base*2, base*2), mode='constant', preserve_range=True)

    pred_img = model.predict(pred_img[None,...,None]).squeeze()
    pred_img = skimage.transform.resize(pred_img, img.shape, mode='constant', preserve_range=True)

    return pred_img

def unet_batch(model, imgs):
    '''Predict all images using a UNet model.
    '''
    return [_unet_single(model, img) for img in imgs]

def _stardist_single(model, img, details=True):
    '''
    '''
    pred_img = csbdeep.utils.normalize(img, 1, 99.8, axis=(0, 1))
    pred_labels, pred_details = model.predict_instances(pred_img)
    if details:
        return pred_labels, pred_details
    return pred_labels

def stardist_batch(model, imgs):
    '''Predict all images using the stardist model.
    '''
    return [_stardist_single(model, img, details=False) for img in imgs]

def _starnet_single(model_star, model_unet, img, watershed=True):
    '''Combine stardist instance prediction with UNet segmentation.
    '''
    def __instances_to_centroids(instances):
        centroids = [r.centroid for r in skimage.measure.regionprops(instances)]
        centroids = [tuple(int(round(n)) for n in tup) for tup in centroids]
        img_centroids = np.zeros(instances.shape, dtype=np.int)
        for n, c in enumerate(centroids):
            img_centroids[c[0], c[1]] = n
        return img_centroids

    pred_star = _stardist_single(model_star, img, details=False)
    pred_unet = _unet_single(model_unet, img)

    img_centroids = __instances_to_centroids(pred_star)
    #TODO – Find optimal prediction, prob. with erosion of borders
    img_area = (1 - pred_unet[:,:,0]>0.5).astype(np.int)

    if watershed:
        return skimage.segmentation.watershed(~img_area, img_centroids, watershed_line=True) * img_area
    if not watershed:
        return skimage.segmentation.random_walker(~img_area, img_centroids) * img_area

def starnet_batch(model_star, model_unet, imgs, watershed=True):
    '''
    '''
    return [_starnet_single(model_star, model_unet, img, watershed=True) for img in imgs]