import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def if_authorized(score: float, treshold: float) -> int:
    """
    Check whether the system would allow the user to pass.
    Args:
        score - user score, float [0-1]
        treshold - minimum score to pass, float [0-1]
    """
    if score < treshold:
        return 1
    else:
        return 0

def calculate_general_metrics(actual, scores, treshold):
    """
    Check whether the system would allow the user to pass.
    Args:
        actual - np.array of zeros and ones shwoing the desired output of the system
        scores - user scores, np.array of floats [0-1]
        treshold - minimum score to pass, float [0-1]
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(scores)):
        prediction = if_authorized(scores[i], treshold)

        if(prediction != actual[i] and prediction == 1):
            FP+=1
        if(prediction == actual[i] == 1):
            TP+=1
        if(prediction != actual[i] and prediction == 0):
            FN+=1
        if(prediction == actual[i] == 0):
            TN+=1
    return TP, FP, TN, FN

def calculate_general_metrics_from_preds(actual, prediction):
    """
    Check whether the system would allow the user to pass.
    Args:
        actual - np.array of zeros and ones shwoing the desired output of the system
        scores - user scores, np.array of floats [0-1]
        treshold - minimum score to pass, float [0-1]
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(prediction)):

        if(prediction[i] != actual[i] and prediction[i] == 1):
            FP+=1
        if(prediction[i] == actual[i] == 1):
            TP+=1
        if(prediction[i] != actual[i] and prediction[i] == 0):
            FN+=1
        if(prediction[i] == actual[i] == 0):
            TN+=1
    return TP, FP, TN, FN

def calc_far_frr(FP, FN, totalP, totalN):
    """
    Calculate FAR and FFR using FP and FN values.
    Args:
        FP - False Positive
        FN - False Negative
        totalP - total positives in the desired ouput
        totalN - total negatives in the desired ouput
    """
    FAR = FP/float(totalN) # False Accept Rate in percentage
    FRR = FN/float(totalP) # False Reject Rate in percentage
    # FAR = FP/float(FP+TP) # False Accept Rate in percentage
    # FRR = FN/float(FN+TN) # False Reject Rate in percentage
    return FAR, FRR

def calculate_by_treshold(y_test, y_pred, total_positives, total_negatives):
    """
    Calculating FAR (False Acceptance Rate) and FRR (False Rejection Rate) for given real and predicted values.
    Args:
        y_test - true (desired) ouput of the system
        y_pred - what the system actually predicited
        totalP - total positives in the desired ouput
        totalN - total negatives in the desired ouput
    """
    step = 1
    far = dict()
    frr = dict()

    for i in range(0, 100, step):
        TP, FP, TN, FN = calculate_general_metrics(y_test, y_pred, i/float(100))
        far[i], frr[i] = calc_far_frr(FP, FN, total_positives, total_negatives)
    # return far, frr
    return far, frr

def plot_far_frr(far, frr):

    axisVal = np.arange(0,1.00,0.01)

    # PLOT FAR FRR
    plt.figure(figsize=(10, 6), dpi=100)
    lw = 2
    plt.plot(axisVal, far.values(), label='False Accept Rate', color='blue', lw=lw)
    plt.plot(axisVal, frr.values(), label='False Reject Rate', color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('Treshold')
    plt.ylabel('Errors')
    plt.title('FAR and FRR')
    plt.legend(loc="upper right")
    plt.locator_params(nbins=20)
    plt.savefig('far-ffr.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_matrix(labels, preds, name=None):
    """
    Plot a simple confusion matrix
    Args:
        labels - true labels, list of ones and zeros
        preds - predicted labels, list of floats in range [0, 1]
        name - name for the plot, optional
    """
    preds = [round(x) for x in preds]
    cm=confusion_matrix(labels, preds)
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize = (10,7))
    sn.heatmap(
        df_cm, 
        annot=True, 
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['True Negative', 'True Positive'])
    
    plt.savefig(f'{name}_CM.png')
    plt.show()

##### TEST INPUT - RUN AND CHECK #####
if __name__ == '__main__':
    y_pred = np.load('utils/pred.npy')
    y_real = np.load('utils/real.npy')
    unique, counts = np.unique(y_real, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    total_p = counts_dict[1]
    total_n = counts_dict[0]

    far, ffr = calculate_by_treshold(y_real, y_pred, total_p, total_n)
    print(far)
    plot_far_frr(far, ffr)
