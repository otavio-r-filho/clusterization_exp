import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from scipy import interp

def plot_multiclass_roc_curve(y_true, y_pred, classes=None, title="", average="macro"):
    '''
    Functions to plot ROC curve 
    
    Paramters
    ---------
    
    Returns
    -------
    '''
    
    # Loading session if session is a string
#     if isinstance(session, str):
#         with open(session, "rb") as f:
#             session = pickle.load(f)
    
#     # Setting up figure size
#     plt.figure(figsize=(14,10))
    
#     # Loading y_true and y_pred arrays
#     y_true = np.array(session[y_true])
#     y_pred = np.array(session[y_pred])
    
    
#     # Reshaping arrays if CV was used
    if len(y_true.shape) > 2:
        y_true = y_true.reshape((-1, y_true.shape[-1]))
        
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        
    tprs = []
    fprs = []
    roc_aucs = []
    
    # Calculating ROC curve for each class
    for i in range(y_true.shape[-1]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, 'deepskyblue', alpha=0.15)
        
    if average == 'macro':
        fpr_avg = np.unique(np.concatenate(fprs))
        tpr_avg = np.zeros_like(fpr_avg)
        for fpr, tpr in zip(fprs, tprs):
            tpr_avg += interp(fpr_avg, fpr, tpr)
        tpr_avg /= y_true.shape[-1]
        roc_auc_avg = auc(fpr_avg, tpr_avg)
    elif average == 'micro':
        fpr_avg, tpr_avg, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc_avg = auc(fpr_avg, tpr_avg)
    else:
        print("Invalid average option. Value can be: 'macro' or 'micro'")
        return
    
    # Calculating TPR std dev
    norm_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        norm_tprs.append(interp(fpr_avg, fpr, tpr))
    norm_tprs = np.array(norm_tprs)
    tpr_std = norm_tprs.std(axis=0)
    
    # Calculating maximum and minimum limits for the std dev interval region
    tprs_upper = np.minimum(tpr_avg + tpr_std, 1)
    tprs_lower = np.maximum(tpr_avg - tpr_std, 0)
    
    # Plotting average ROC curve
    plt.plot(fpr_avg, tpr_avg,
             label='{1}-average ROC curve (area = {0:0.2f})'.format(roc_auc_avg, average),
             color='dodgerblue', linewidth=2)
    
    # Plotting std dev region
    plt.fill_between(fpr_avg, tprs_lower, tprs_upper,
                     label=r'$\pm$ 1 std. dev.',
                     color='salmon', alpha=0.3)

    plt.plot([0, 1], [0, 1],
             label='random guess',
             color='dimgray', ls='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(title)
    plt.legend(loc="lower right", fontsize=14)
    ax=plt.gca()
    ax.tick_params(labelsize=18)
    plt.tight_layout(pad=0.5)
