import torch
import numpy as np
from sklearn import metrics

import numpy as np

def AUC_score(SR,GT,threshold=0.5):
    #SR = SR.cpu().numpy()
    #GT = GT.cpu().numpy()
    fpr, tpr, _ = metrics.roc_curve(GT, SR)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc

def AUC_score_multilabel(PRE, GT, threshold=0.5):
    
    val, cnt = np.unique(GT, return_counts=True)
    
    length = len(val)
    
    roc_aucs = 0.0

    for v in val:
        
        # v = val[i]
        
        ind = int(v)
        
        score = PRE[:,ind]
        
        # gt = GT.copy()
        # gt = np.array(gt)
        score = np.array(score)
        
        # gt[gt!=ind]=-1
        # gt[gt==ind]=1
        # gt[gt==-1]=0
        gt = np.array(GT)
        gt = (gt==ind)
        
        fpr, tpr, _ = metrics.roc_curve(gt, score)
        roc_auc = metrics.auc(fpr, tpr)
        
        roc_aucs += roc_auc
        
    auc = roc_aucs / length
        
    return auc


def AUC_score_all(PRE, GT, threshold=0.5):
    
    val, cnt = np.unique(GT, return_counts=True)
    
    length = len(val)
    
    roc_aucs = 0.0
    
    ros = []

    for v in val:
        
        ind = int(v)
        
        score = PRE[:,ind]
        
        score = np.array(score)
        
        gt = np.array(GT)
        gt = (gt==ind)
        fpr, tpr, _ = metrics.roc_curve(gt, score)
        roc_auc = metrics.auc(fpr, tpr)
        roc_aucs += roc_auc
        ros.append({int(v): float('{:.4f}'.format(roc_auc))})
        
    auc = roc_aucs / length
    return auc, ros


def confusion(output, target):
    # output = output.asty
    # target = target.double()
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    p = torch.sum(target == 1).item()
    n = torch.sum(target == 0).item()

    tp = (output * target).sum().item()
    tn = ((1 - output) * (1 - target)).sum().item()
    fp = ((1 - target) * output).sum().item()
    fn = ((1 - output) * target).sum().item()
    epslon = 0.000001
    res = {"P": p, "N": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "TPR": (tp / (tp+fn+epslon)), "TNR": (tn /(tn+fp+epslon) ), "FPR": (fp / (n+epslon)),
           "FNR": (fn / (p+epslon)), "Accuracy": (tp + tn) / (tp+fn+tn+fp+epslon)}
    return res

def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    # FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    # FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    # TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    # TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    FP = float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN

def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0

def recall_multilabel(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res = 0
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        if (TP + FN) <= 0.0:
            continue
        TPR = np.divide(TP, TP + FN)
        res += TPR * wi[i]
    return res * 100.0


def recall_all(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res = 0
    res_cls = []
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        if (TP + FN) <= 0.0:
            res_cls.append({int(val[i]): float('{:.2f}'.format(0.0))})
            continue
        TPR = np.divide(TP, TP + FN)
        res_cls.append({int(val[i]): float('{:.2f}'.format(TPR * 100.0))})
        res += TPR * wi[i]
    return res * 100.0, res_cls


def precision_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return 0.0
    precision = np.divide(TP, TP + FP)
    return precision * 100.0

def precision_multilabel(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res = 0
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        if (TP + FP) <= 0.0:
            continue
        TPR = np.divide(TP, TP + FP)
        res += TPR * wi[i]
    return res * 100.0



def precision_all(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res_cls = []
    res = 0
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        if (TP + FP) <= 0.0:
            res_cls.append({int(val[i]): float('{:.2f}'.format(0.0))})
            continue
        TPR = np.divide(TP, TP + FP)
        res_cls.append({int(val[i]): float('{:.2f}'.format(TPR * 100.0))})
        res += TPR * wi[i]
    return res * 100.0, res_cls


def specificity_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0

def specificity_multilabel(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res = 0
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        if (TN + FP) <= 0.0:
            continue
        TNR = np.divide(TN, TN + FP)
        res += TNR * wi[i]
    return res * 100.0



def specificity_all(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res = 0
    res_cls = []
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        if (TN + FP) <= 0.0:
            res_cls.append({int(val[i]): float('{:.2f}'.format(0.0))})
            continue
        TNR = np.divide(TN, TN + FP)
        res_cls.append({int(val[i]): float('{:.2f}'.format(TNR * 100.0))})
        res += TNR * wi[i]
    return res * 100.0, res_cls


def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0


def accuracy_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def accuracy_multilabel(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res = 0
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        N = FP + FN + TP + TN
        if N <= 0.0:
            continue
        accuracy = np.divide(TP + TN, N)
        res += accuracy * wi[i]
    return res * 100.0


def accuracy_all(groundtruth, prediction):
    val, cnt = np.unique(groundtruth, return_counts=True)
    Nc = len(val)
    wi = cnt/len(groundtruth)
    res = 0
    res_cls = []
    for i in range(len(val)):
        pred = (prediction==val[i])
        gt = (groundtruth==val[i])
        FP, FN, TP, TN = numeric_score(pred, gt)
        N = FP + FN + TP + TN
        if N <= 0.0:
            res_cls.append({int(val[i]): float('{:.2f}'.format(0.0))})
            continue
        accuracy = np.divide(TP + TN, N)
        res_cls.append({int(val[i]): float('{:.2f}'.format(accuracy * 100.0))})
        res += accuracy * wi[i]
    return res * 100.0, res_cls

def f1_score(prediction, groundtruth):
    precision = precision_score(groundtruth, prediction)
    sen = recall_score(groundtruth, prediction)
    if (precision + sen)<=0:
        return 0.0
    F1 = (2 * precision * sen) / (precision + sen)
    return F1

def f1_multilabel(groundtruth, prediction):
    precision = precision_multilabel(prediction, groundtruth)
    sen = recall_multilabel(prediction, groundtruth)
    if (precision + sen)<=0:
        return 0.0
    F1 = (2 * precision * sen) / (precision + sen)
    return F1

def f1_all(groundtruth, prediction):
    precision, precisions = precision_all(groundtruth, prediction)
    sen, sens = recall_all(groundtruth, prediction)
    f1s = []
    for num_index, pre_dict in enumerate(precisions): 
        clsid = list(pre_dict.keys())[0]
        pre_val = list(pre_dict.values())[0]
        sen_dict = sens[num_index]
        sen_val = list(sen_dict.values())[0]
        if (float(pre_val) + float(sen_val))<=0:
            f1s.append({clsid: float('{:.2f}'.format(0.0))})
            continue
        f1 = (2 * float(pre_val) * float(sen_val)) / (float(pre_val) + float(sen_val))
        f1s.append({clsid: float('{:.2f}'.format(f1))})
    if (precision + sen)<=0:
        F1 = 0.0
    else:
        F1 = (2 * precision * sen) / (precision + sen)
    return F1, f1s