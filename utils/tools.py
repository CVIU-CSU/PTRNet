import os
import cv2
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import sys

def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * np.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res

import cmapy
def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w = shape
        cam = cv2.resize(cam, (h, w))

    cam = cam.astype(np.uint8)
    # res = np.zeros((cam.shape[0],cam.shape[1],3), dtype=np.uint8)
    # res[:,:,0] = cam
    # res[:,:,1] = 1 - cam
    # res[:,:,2] = cam
    # return res
    cam = cv2.applyColorMap(cam,colormap=cv2.COLORMAP_JET)#cmapy.cmap('seismic'))
    return cam

def plot_progress(epoch, train_loss, val_loss, f1s, aucs, lrs, path):
    """
    Should probably by improved
    :return:
    """
    font = {'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)
        
    fig = plt.figure(figsize=(30, 24))
    fig2 = plt.figure(figsize=(30, 24))
    
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    
    ax3 = fig2.add_subplot(111)

    x_values = epoch

    ax.plot(x_values, train_loss, color='b', ls='-', label="loss_tr")

    ax.plot(x_values, val_loss, color='r', ls='-', label="loss_val")
    
    ax2.plot(x_values, f1s, color='g', ls='-', label="f1")
    
    ax2.plot(x_values, aucs, color='y', ls='-', label="auc")

    ax3.plot(x_values, lrs, color='y', ls='-', label="learning rate")

        # if len(self.all_val_losses_tr_mode) > 0:
        #     ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
        # if len(self.all_val_eval_metrics) == len(x_values):
        #     ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("evaluation metric")
    ax.legend()
    ax2.legend(loc=9)
    
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("learning rate")

    fig.savefig(os.path.join(path, "progress.png"))
    fig2.savefig(os.path.join(path, "lr.png"))
    plt.close()
            
def plot_progress_(iter, feat_max, feat_mean, path):
    """
    Should probably by improved
    :return:
    """
    font = {'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)
        
    fig = plt.figure(figsize=(30, 24))
    
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    x_values = iter

    ax.plot(x_values, feat_max, color='b', ls='-', label="feat_max")
    ax2.plot(x_values, feat_mean, color='r', ls='-', label="feat_mean")

        # if len(self.all_val_losses_tr_mode) > 0:
        #     ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
        # if len(self.all_val_eval_metrics) == len(x_values):
        #     ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

    ax.set_xlabel("iter")
    ax.set_ylabel("feat")
    ax2.set_ylabel("evaluation metric")
    
    ax.legend()
    ax2.legend()
    

    fig.savefig(os.path.join(path, "feat.png"))
    plt.close()
    
    
def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    # 进行min-max归一化
    x = F.relu(x)
    mmin = torch.min(x)
    mmax = torch.max(x)
    return (x - mmin + epsilon)/(mmax - mmin + epsilon)
    x = F.relu(x)

    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon)

def numeric_score(pred_arr, gt_arr, kernel_size=(1, 1)):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)
    
    FP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))
    
    return FP, FN, TP, TN

def calc_acc(pred_arr, gt_arr, kernel_size=(1, 1), mask_arr=None):
    # pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    acc = (TP + TN) / (FP + FN + TP + TN)
    
    return acc

def get_best_thresh(gt_arr, pred_arr):
    gt_arr = gt_arr
    pred_arr = pred_arr
    fpr, tpr, thresholds = metrics.roc_curve(gt_arr.reshape(-1), pred_arr.reshape(-1), pos_label=1)
    
    best_acc = 0
    thresh_value = 0
    for i in range(thresholds.shape[0]):
        thresh_arr = pred_arr.copy()
        thresh_arr[thresh_arr >= thresholds[i]] = 1
        thresh_arr[thresh_arr < thresholds[i]] = 0
        current_acc = calc_acc(thresh_arr, gt_arr)
        if current_acc >= best_acc:
            best_acc = current_acc
            thresh_value = thresholds[i]
    
    thresh_arr = pred_arr.copy()
    thresh_arr[thresh_arr >= thresh_value] = 255
    thresh_arr[thresh_arr < thresh_value] = 0
    
    return thresh_value
