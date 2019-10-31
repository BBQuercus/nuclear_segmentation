import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

#TODO – add other metrics

def get_iou_vector(A, B):
    '''
    '''
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        if true == 0:
            metric += (pred == 0)
            continue
        
        # If not empty – Union is never empty 
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # Iou metric – stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        metric += iou
        
    metric /= batch_size
    return metric

def optimize_threshold(Y_true, Y_hat, draw=True):
    '''
    '''
    thresholds = np.linspace(0.2, 0.9, 31)
    ious = [np.average(
        [get_iou_vector(y_true>0, y_hat>t) for y_true, y_hat in zip(Y_true, Y_hat)])
        for t in thresholds]

    threshold_best_index = np.argmax(ious) 
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    if draw:
        plt.plot(thresholds, ious)
        plt.plot(threshold_best, iou_best, 'xr', label='Best threshold')
        plt.xlabel('Threshold')
        plt.ylabel('IoU')
        plt.title(f'Threshold vs IoU ({threshold_best:.2}, {iou_best:.2})')
        plt.legend()
    
    return threshold_best

def measure_iou(Y_true, Y_hat, name=None, prob=0.5):
    '''
    '''
    IOU = np.array([get_iou_vector(y_true>0, y_hat>prob) for y_true, y_hat in zip(Y_true, Y_hat)])
    
    print(f'Average IoU: {np.average(IOU)}')
    print(f'Stdev IoU: {np.std(IOU)}')
    
    sns.boxplot(IOU)
    plt.xlim(0, 1)
    plt.title('Intersection over Union')
    sns.despine(left=True, bottom=False)
    plt.show()
    
    if name:
        np.savetxt(f'./measurements/{name}.csv', IOU, delimiter=',')

    return None