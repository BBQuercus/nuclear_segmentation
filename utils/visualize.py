import sys
import numpy as np
import matplotlib.pyplot as plt

import skimage
import csbdeep.utils
import stardist

sys.path.append('../')
import utils.evaluation
import utils.predictions

lbl_cmap = stardist.random_label_cmap()

def _visualize_intersection(y_true, y_pred):
    '''
    '''
    y_true = y_true > 0
    y_pred = y_pred > 0

    img = np.zeros([y_true.shape[0], y_true.shape[1], 3])
    # Red – false negatives
    img[:,:,0] = np.where(y_true==y_pred, 0, 1)
    # Green – true positives
    img[:,:,1] = np.where(y_true==y_pred, y_pred, 0)
    # Blue – false positives
    img[:,:,2] = np.where(y_true==y_pred, y_true!=y_pred, 0)
    return img

def otsu_example(x_images, y_images, ix=None):
    '''
    '''
    if not ix: 
        ix = np.random.randint(0, len(x_images)-1) if len(x_images)>1 else 0
    x_image = x_images[ix]
    y_image = y_images[ix]

    pred = x_image > skimage.filters.threshold_otsu(x_image)

    _, ax = plt.subplots(1, 4, figsize=(16, 14))
    ax[0].imshow(x_image)
    ax[0].set_title(f'Original Image – #{ix}')
    ax[0].axis('off')
    ax[1].imshow(y_image)
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    ax[2].imshow(pred)
    ax[2].set_title('Otsu prediction')
    ax[2].axis('off')
    ax[3].imshow(_visualize_intersection(y_image>0, pred))
    ax[3].set_title('Intersection')
    ax[3].axis('off')
    plt.show()

def unet_example(model, x_images, y_images, ix=None):
    '''Show an example prediction with the UNet model.
    '''
    if not ix:
        ix = np.random.randint(0, len(x_images)-1) if len(x_images)>1 else 0
    x_image = x_images[ix]
    y_image = y_images[ix]

    pred = utils.predictions._unet_single(model, x_image)

    _, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0,0].set_title(f'Original Image – #{ix}')
    ax[0,0].imshow(x_image)
    ax[0,0].axis('off')
    ax[0,1].set_title('Ground Truth')
    ax[0,1].imshow(y_image, cmap=lbl_cmap)
    ax[0,1].axis('off')
    ax[1,0].set_title('Prediction Nucleus')
    ax[1,0].imshow(pred[:,:,1])
    ax[1,0].axis('off')
    ax[1,1].set_title('Prediction Border')
    ax[1,1].imshow(pred[:,:,2])
    ax[1,1].axis('off')
    plt.show()

def stardist_example(model, x_images, y_images, ix=None):
    '''Show an example prediction with the stardist model.
    '''
    if not ix:
        ix = np.random.randint(0, len(x_images)-1) if len(x_images)>1 else 0
    x_image = x_images[ix]
    y_image = y_images[ix]

    pred, details = utils.predictions._stardist_single(model, x_image)
    coord, points, prob = details['coord'], details['points'], details['prob']

    _, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0,0].set_title(f'Original Image – #{ix}')
    ax[0,0].imshow(x_image)
    ax[0,0].axis('off')
    ax[0,1].set_title('Ground Truth')
    ax[0,1].imshow(y_image, cmap=lbl_cmap)
    ax[0,1].axis('off')
    ax[1,0].set_title('Prediction')
    ax[1,0].imshow(pred, cmap=lbl_cmap)
    ax[1,0].axis('off')
    ax[1,1].set_title('Polygon Prediction')
    ax[1,1].imshow(x_image, cmap='gray')
    stardist._draw_polygons(coord, points, prob, grid=model.config.grid, show_dist=True)
    ax[1,1].axis('off')
    plt.show()

def starnet_example(model_star, model_unet, x_images, y_images, watershed=True, ix=None):
    '''
    '''
    if not ix:
        ix = np.random.randint(0, len(x_images)-1) if len(x_images)>1 else 0
    x_image = x_images[ix]
    y_image = y_images[ix]

    pred_star = utils.predictions._stardist_single(model_star, x_image, details=False)
    pred = utils.predictions._starnet_single(model_star, model_unet, x_image)
    
    _, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax[0,0].set_title(f'Original Image – #{ix}')
    ax[0,0].imshow(x_image)
    ax[0,0].axis('off')
    ax[0,1].set_title('Ground Truth')
    ax[0,1].imshow(y_image, cmap=lbl_cmap)
    ax[0,1].axis('off')
    ax[0,2].axis('off')
    ax[1,0].set_title('Stardist prediction')
    ax[1,0].imshow(pred_star, cmap=lbl_cmap)
    ax[1,0].axis('off')
    ax[1,1].set_title('Starnet prediction')
    ax[1,1].imshow(pred, cmap=lbl_cmap)
    ax[1,1].axis('off')
    ax[1,2].set_title('Intersection')
    ax[1,2].imshow(_visualize_intersection(y_image>0, pred>0))
    ax[1,2].axis('off')
    plt.show()