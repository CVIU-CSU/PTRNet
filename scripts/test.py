import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
from models.ablation_modality_models import union_model

from utils.config import Logger
from utils.dataset_ANCA import ANCAdataset
import torch
from utils.WarmUpLR import WarmupLR

import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from PIL import Image
from torch import optim, nn
import csv
from tqdm import tqdm
from evaluation.matrixs import *
from sklearn.metrics import confusion_matrix
import math

from evaluation.best_thr_utils import six_scores
from evaluation.seg_score import Evaluator
from utils.dice_score import dice_coeff

# set seed
GLOBAL_SEED = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def get_arguments():
    parser = argparse.ArgumentParser(description='group-wise semantic mining for weakly supervised semantic segmentation')
    parser.add_argument("--Adiscription", type=str, default='BiomedCLIP + tabtransformer')
    parser.add_argument("--saveName", type=str, default='try')
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=4e-4)
    parser.add_argument("--init_lr", type=float, default=1e-7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--dropout",type=float,default=0)
    parser.add_argument("--alpha",type=float,default=0.2)
    parser.add_argument("--gpu", type=str, default='2')
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--num_warmup",type=int,default=10)
    parser.add_argument("--lr_way",type=str,default='cos')
    parser.add_argument("--d_model",type=int,default=256)
    parser.add_argument("--model_mode",type=str,default='union',choices=['tabular','image','union','concat'])

    parser.add_argument("--weight_path",type=str,default='log/modality_ablation/model_final.pt')

    return parser.parse_args()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) 
    mx = r_mat_inv.dot(mx)
    return mx 

def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']

def test(network, data_loaderTest,epoch, args, out_path):
    
    network.eval()
    correct = 0
    correct_another = 0
    total = 0
    output_np = []
    pre_all = []
    pre_01_all = []
    label_all = []
    labels_all = []
    preds_all = []
    pred_01_all = []
    csvFile = open(args.savePath + "/%d_TestResult.csv"%epoch, "w")

    cetiria = nn.CrossEntropyLoss()
    val_loss = 0

    evaluator =Evaluator(2)
    dice_scores = []
    
    with torch.no_grad():
        tbar = tqdm(data_loaderTest)
        for j, data in enumerate(tbar):
            tbar.set_description(
                f"No. {j}/{len(data_loaderTest)}"
            )
            labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask = data
            x_categ, x_numer, labels, img_data = x_categ.to(device), x_numer.to(device), labels.to(device), img_data.to(device)
            inner_slice_mask, inter_slice_mask = inner_slice_mask.to(device), inter_slice_mask.to(device)
            
            outputs = network(x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='test')
            outputs = nn.Softmax(dim=1)(outputs)

            #######################################################################
            v_loss = cetiria(outputs, labels)
            val_loss += v_loss.item()
            
            values = outputs[0].clone()
            # values = nn.Softmax(dim=1)(values)
            values = values.cpu().detach().numpy()
            
            preds = outputs.cpu().detach().numpy()
            # value = values[:,1]
            # np.append(pred_01_all, values)
            pred_01_all.append(values)
            # pred_01_all.extend(values)
            
            preds = preds.argmax(1)
            preds_all.extend(preds)
            
            labels_all.extend(labels.cpu().detach().numpy())
            ######################################################################

            value = outputs[:,1]
            threshhold = 0.5
            zero = torch.zeros_like(value)
            one = torch.ones_like(value)
            predicted = torch.where(value > threshhold, one, zero)

            value1 = value.cpu().detach().numpy()
            labels1 = labels.cpu().detach().numpy()
            predicted1 = predicted.cpu().detach().numpy()

            pre_all = np.append(pre_all, value1)
            label_all = np.append(label_all, labels1)
            pre_01_all = np.append(pre_01_all, predicted1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            name = data_loaderTest.dataset.getFileName()
            writer = csv.writer(csvFile)

            value, predicted = value.cpu().detach().numpy(), predicted.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            data = [name, value1[0], predicted[0], labels[0]]
            
            writer.writerow(data)

            seg_pred = network(x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='vis')
            B = inner_slice_mask.shape[1]
            gt_mask = inner_slice_mask[0,:,1:].reshape(B,14,14)

            seg_pred[seg_pred>=0.5] = 1.0
            seg_pred[seg_pred<0.5] = 0.0

            for i in range(B):
                dice_scores.append(dice_coeff(seg_pred[i].unsqueeze(0), gt_mask[i].unsqueeze(0), reduce_batch_first=True).cpu().item())

            seg_pred = seg_pred.reshape(B,14*14).cpu().numpy().astype(np.int32)
            gt_mask = gt_mask.reshape(B,14*14).cpu().numpy().astype(np.int32)
            for i in range(B):
                evaluator.add_batch(gt_mask[i,:], seg_pred[i,:])


    print('Accuracy of the network on the  test images: %.3f %%' % (100.0 * correct / total))
    
    pred_01_all = np.array(pred_01_all)
    AUC = AUC_score(pre_all,label_all)

    with open(out_path, 'w', newline='') as csvfile:
        fieldnames = ['pre_all', 'label_all']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(pre_all)):
            writer.writerow({'pre_all': pre_all[i], 'label_all': label_all[i]})
    
    threshhold = 0.5
    #print('threshhold:',threshhold)
    pre_01_all[pre_all >= threshhold] = 1
    pre_01_all[pre_all < threshhold] = 0
    
    cm = confusion_matrix(labels_all, preds_all)
    
    print('\n')
    print(cm)
    print('\n')

    _Spe = specificity_score(pre_01_all, label_all)
    _Sen = recall_score(pre_01_all, label_all)
    _Acc = accuracy_score(pre_01_all, label_all)
    _precision = precision_score(pre_01_all, label_all)
    _F1 = f1_score(pre_01_all, label_all)
    
    print('SEN', '{:.2f},'.format(_Sen), 
          'SPE', '{:.2f},'.format(_Spe), 
          'ACC', '{:.2f},'.format(_Acc), 
        #   'PRE', '{:.2f},'.format(_precision),
          'F1', '{:.2f},'.format(_F1),
          'AUC', '{:.4f}\n'.format(AUC))

    val_loss = val_loss / len(data_loaderTest)
    csvFile.close()

    accuracy, auc_value, precision, recall, specifity, fscore = six_scores(torch.from_numpy(label_all), torch.from_numpy(pre_all))
    print('SEN', '{:.2f},'.format(recall*100.0), 
        'SPE', '{:.2f},'.format(specifity*100.0), 
        'ACC', '{:.2f},'.format(accuracy*100.0), 
        'F1', '{:.2f},'.format(fscore*100.0),
        'AUC', '{:.4f}'.format(auc_value),
        'PRE', '{:.2f}\n'.format(precision*100.0))

    print('Dice_Score', '{:.3f},'.format(sum(dice_scores) / len(dice_scores)), 
          'PA', '{:.3f},'.format(evaluator.Pixel_Accuracy()),
          'IOU', '{:.3f},'.format(evaluator.Mean_Intersection_over_Union()),
          'FWIOU', '{:.3f},'.format(evaluator.Frequency_Weighted_Intersection_over_Union()))

    return 100.0 * correct / total , output_np,AUC,_Acc,_Sen,_Spe,_F1,val_loss

if __name__ == '__main__':
    args = get_arguments()

    worker_init_fn(args.seed)

    saveName = args.saveName
    args.savePath = os.path.join('log', saveName)
    train_log_path = os.path.join(args.savePath, 'train.log')
    
    num_epochs = args.num_epochs
    base_lr = args.base_lr
    init_lr = args.init_lr

    nclass = args.nclass

    dropout = args.dropout
    alpha = args.alpha

    #GPU
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = args.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    Logger(train_log_path)
    
    print(args)

    model = union_model(d_model=args.d_model)

    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    model = model.to(device)

    ct_data_path = '/home/zhangyinan/2024_05/processed_ct_image_no_resize_v2/feats'
    tabular_data_path = '/home/zhangyinan/2024_05/merged_tables.csv'
    
    datasetTrain = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'train')
    data_loaderTrain = DataLoader(datasetTrain, batch_size=args.batch_size,shuffle=True)

    data_loaderTrainTest = DataLoader(datasetTrain, batch_size=1,shuffle=False)

    datasetVal = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'val')
    data_loaderVal = DataLoader(datasetVal, batch_size=1,shuffle=False)

    datasetTest = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'test')
    data_loaderTest = DataLoader(datasetTest, batch_size=1,shuffle=False)

    with torch.no_grad():
        test_acc, prediction, AUC, Acc, Sen, Spe, _F1, v_loss = test(model,data_loaderTrainTest,0,args,os.path.join(args.savePath, 'train.csv')) # train
        test_acc, prediction, AUC, Acc, Sen, Spe, _F1, v_loss = test(model,data_loaderVal,0,args,os.path.join(args.savePath, 'val.csv')) # val
        test_acc, prediction, AUC1, Acc, Sen, Spe, _F11, v_loss = test(model,data_loaderTest,0,args,os.path.join(args.savePath, 'test.csv')) # test
