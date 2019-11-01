import numpy as np
import skimage.transform
import tensorflow as tf
import scipy.ndimage

class StarAugment:
    '''
    '''

    @staticmethod
    def _fix_labeling(Y):
        '''Mirroring augmentations can lead to separate objects having the same label.
        This function relabels these artifacts.
        '''
        next_label = Y.max() + 1
        for region in skimage.measure.regionprops(Y):
            bbox = tuple(slice(low, high)
                         for low, high in zip(region.bbox[:Y.ndim], region.bbox[Y.ndim:]))
            cc, n_components = scipy.ndimage.measurements.label(Y[bbox] == region.label)
            if n_components <= 1:
                continue
            for nth_component in range(2, n_components+1):
                Y[bbox][cc == nth_component] = next_label
                next_label += 1
        return Y
    
    def __init__(self, **data_gen_args):
        '''
        '''
        self.image_transformer = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        self.label_transformer = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        
    def augment(self, X_batch, Y_batch):
        '''
        '''
        seeds = np.random.randint(0, 512, size=len(X_batch))
        # Images
        X_batch = np.asarray([self.image_transformer.random_transform(X[..., None], seed=seed)
                              for X, seed in zip(X_batch, seeds)]).squeeze()
        # Masks
        Y_batch = np.asarray([self._fix_labeling(self.label_transformer.random_transform(Y[..., None], seed=seed))
                              for Y, seed in zip(Y_batch, seeds)]).squeeze()
        
        return X_batch, Y_batch


# Legacy

# # Based on example code from:
# # http://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html

# def deform(image1, image2, points=10, distort=5.0):
#     '''
#     '''
#     # create deformation grid 
#     rows, cols = image1.shape[0], image1.shape[1]
#     src_cols = np.linspace(0, cols, points)
#     src_rows = np.linspace(0, rows, points)
#     src_rows, src_cols = np.meshgrid(src_rows, src_cols)
#     src = np.dstack([src_cols.flat, src_rows.flat])[0]

#     # add distortion to coordinates
#     s = src[:, 1].shape
#     dst_rows = src[:, 1] + np.random.normal(size=s)*np.random.uniform(0.0, distort, size=s)
#     dst_cols = src[:, 0] + np.random.normal(size=s)*np.random.uniform(0.0, distort, size=s)
    
#     dst = np.vstack([dst_cols, dst_rows]).T

#     tform = skimage.transform.PiecewiseAffineTransform()
#     tform.estimate(src, dst)

#     out_rows = rows 
#     out_cols = cols
#     out1 = skimage.transform.warp(image1, tform, output_shape=(out_rows, out_cols), mode="symmetric")
#     out2 = skimage.transform.warp(image2, tform, output_shape=(out_rows, out_cols), mode="symmetric")
    
#     return out1, out2

# def resize(x, y):
#     '''
#     '''
#     wf = 1 + np.random.uniform(-0.25, 0.25)
#     hf = 1 + np.random.uniform(-0.25, 0.25)

#     w,h = x.shape[0:2]

#     wt, ht = int(wf*w), int(hf*h)

#     new_x = skimage.transform.resize(x, (wt,ht))
#     new_y = skimage.transform.resize(y, (wt,ht))

#     return new_x, new_y


# def get_augmented(X, Y, batch_size=8, categorical=False, validation_split=0.2):
#     '''
#     '''
#     import keras # only until tensorflows zip error is solved
#     from sklearn.model_selection import train_test_split
#     seed = np.random.randint(0, 512)
#     x_train, x_valid, y_train, y_valid = train_test_split(X, Y,
#                                                           train_size=1-validation_split,
#                                                           test_size=validation_split,
#                                                           random_state=seed)
    
#     if categorical:
#         y_train = tf.keras.utils.to_categorical(y_train)
#         y_valid = tf.keras.utils.to_categorical(y_valid)
    
#     # Image data generator distortion options
#     data_gen_args = dict(horizontal_flip=True,
#                          vertical_flip=True,
#                          rotation_range=90,
#                          zoom_range=0.5,
#                          shear_range=0.5,
#                          width_shift_range=0.5,
#                          height_shift_range=0.5,
#                          fill_mode='reflect',
#                          data_format='channels_last')
    
#     # Train data – fitting and flowing
#     x_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
#     y_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
#     x_datagen.fit(x_train, augment=True, seed=seed)
#     y_datagen.fit(y_train, augment=True, seed=seed)
#     x_train_augmented = x_datagen.flow(x_train, seed=seed)
#     y_train_augmented = y_datagen.flow(y_train, seed=seed)
    
#     # Validation data – no augmentation
#     x_datagen_val = tf.keras.preprocessing.image.ImageDataGenerator()
#     y_datagen_val = tf.keras.preprocessing.image.ImageDataGenerator()
#     x_datagen_val.fit(x_valid, augment=True, seed=seed)
#     y_datagen_val.fit(y_valid, augment=True, seed=seed)
#     x_val_augmented = x_datagen_val.flow(x_valid, seed=seed)
#     y_val_augmented = y_datagen_val.flow(y_valid, seed=seed)
    
#     # Combine generators into one which yields image and masks
#     # train_generator = zip(x_train_augmented, y_train_augmented)
#     # valid_generator = zip(x_val_augmented, y_val_augmented)
    
#     return train_generator, valid_generator, x_train, x_valid, y_train, y_valid