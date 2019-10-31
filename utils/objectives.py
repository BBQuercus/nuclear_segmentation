import tensorflow as tf

def weighted_crossentropy(y_true, y_pred):
    '''Weighted version of tf.keras.objectives.categorical_crossentropy
    '''
    class_weights = tf.constant([[[[1., 1., 10.]]]])
    weights = tf.reduce_sum(class_weights*y_true, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    weighted_losses = weights*unweighted_losses
    loss = tf.reduce_mean(weighted_losses)
    return loss
