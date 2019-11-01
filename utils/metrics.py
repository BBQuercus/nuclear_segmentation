import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO – add other metrics

# Plot loss function and metrics
def translate_metric(x):
    translations = {'acc': 'Accuracy', 'loss': 'Log-loss (cost function)'}
    if x in translations:
        return translations[x]
    else:
        return x
    
class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, figsize=(16, 4)):
        super(PlotLosses, self).__init__()
        self.figsize = figsize

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs.copy())

        import IPython.display
        IPython.display.clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        
        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)
            
            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label='training')
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         label='validation')
            plt.title(translate_metric(metric))
            plt.xlabel('Epoch')
            plt.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
