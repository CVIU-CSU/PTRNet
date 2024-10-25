import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
from models.ablation_fcn import union_model, image_model, tabular_model
from utils.config import Logger
from utils.dataset_ANCA import ANCAdataset
import torch
from utils.WarmUpLR import WarmupLR
import numpy as np
from torch.utils.data import DataLoader
from torch import optim, nn
import csv
from tqdm import tqdm
from evaluation.matrixs import *
from sklearn.metrics import confusion_matrix
from utils.tools import plot_progress

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
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--init_lr", type=float, default=1e-7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--num_warmup",type=int,default=10)
    parser.add_argument("--hidden_dim",type=int,default=64)
    parser.add_argument("--d_model",type=int,default=256)
    parser.add_argument("--layer_num",type=int,default=4)
    parser.add_argument("--lr_way",type=str,default='cos')
    parser.add_argument("--model_mode",type=str,default='union',choices=['tabular','image','union'])

    return parser.parse_args()

def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']

def test(network, data_loaderTest,epoch, args):
    network.eval()
    correct = 0
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
            values = values.cpu().detach().numpy()
            
            preds = outputs.cpu().detach().numpy()
            pred_01_all.append(values)
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
                dice_scores.append(dice_coeff(seg_pred[i].unsqueeze(0), gt_mask[i].unsqueeze(0)).cpu().item())

            seg_pred = seg_pred.reshape(B,14*14).cpu().numpy().astype(np.int32)
            gt_mask = gt_mask.reshape(B,14*14).cpu().numpy().astype(np.int32)
            for i in range(B):
                evaluator.add_batch(gt_mask[i,:], seg_pred[i,:])

    print('Accuracy of the network on the  test images: %.3f %%' % (100.0 * correct / total))
    
    pred_01_all = np.array(pred_01_all)
    AUC = AUC_score(pre_all,label_all)
    
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
    
    print('Dice_Score', '{:.3f},'.format(sum(dice_scores) / len(dice_scores)), 
          'PA', '{:.3f},'.format(evaluator.Pixel_Accuracy()),
          'IOU', '{:.3f},'.format(evaluator.Mean_Intersection_over_Union()),
          'FWIOU', '{:.3f},'.format(evaluator.Frequency_Weighted_Intersection_over_Union()))

    val_loss = val_loss / len(data_loaderTest)
    csvFile.close()

    return 100.0 * correct / total , output_np,AUC,_Acc,_Sen,_Spe,_F1,val_loss

def train(network: nn.Module, dataloader, dataloader_test, data_loaderTrainTest, data_loaderTest, args):
    
    # Create Optimizer
    lrate = args.base_lr
    optimizer = optim.Adam(network.parameters(), lr = lrate)
    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.num_epochs)
    schedulers = WarmupLR(scheduler_steplr, init_lr=args.init_lr, num_warmup=10, warmup_strategy='cos')

    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    val_loss = []
    aucs = []
    f1 = []
    epochs = []
    best_auc = 0
    best_acc = 0
    best_epoch = -1
    record_test_acc1 = 0
    record_test_auc1 = 0

    lrs = []

    # Train model on the dataset
    for epoch in range(args.num_epochs):

        schedulers.step()

        print('-' * 10)
        print('Train Epoch %d/%d' % (epoch, args.num_epochs - 1))
        runing_loss = 0.0
        network.train(mode=True)

        tbar = tqdm(dataloader)

        for i, data in enumerate(tbar):
            labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask = data

            x_categ, x_numer, labels, img_data = x_categ.to(device), x_numer.to(device), labels.to(device), img_data.to(device)
            inner_slice_mask, inter_slice_mask = inner_slice_mask.to(device), inter_slice_mask.to(device)

            optimizer.zero_grad()

            if args.model_mode == 'union':
                outputs,loss_extra = network(x_categ, x_numer, img_data,inner_slice_mask, inter_slice_mask, mode='train')
                if epoch < args.num_epochs * 0.3:
                    loss = criterion(outputs, labels)*0 + loss_extra
                    loss_cls = criterion(outputs, labels)*0
                    loss_seg = loss_extra
                else:
                    loss = criterion(outputs, labels) + loss_extra*0.1
                    loss_cls = criterion(outputs, labels)
                    loss_seg = loss_extra*0.1
            elif args.model_mode == 'tabular':
                outputs = network(x_categ, x_numer, img_data,inner_slice_mask, inter_slice_mask, mode='train')
                loss = criterion(outputs, labels)
                loss_cls = criterion(outputs, labels)
                loss_seg = torch.tensor([0])
            else:
                outputs, loss_extra = network(x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='train')
                if epoch < args.num_epochs * 0.3:
                    loss = criterion(outputs, labels)*0 + loss_extra # default
                    loss_cls = criterion(outputs, labels)*0
                    loss_seg = loss_extra
                else:
                    loss = criterion(outputs, labels) + loss_extra*0.1
                    loss_cls = criterion(outputs, labels)
                    loss_seg = loss_extra

            loss.backward()

            optimizer.step()
            runing_loss += loss.item()

            current_lr = get_lr(optimizer)
            tbar.set_description(
                f"Epoch {epoch}/{args.num_epochs-1}, Batch {i}/{(len(data_loaderTrain.dataset) - 1) // data_loaderTrain.batch_size + 1},Batch Loss={loss.item():.4f},Batch Loss_cls={loss_cls.item():.4f},Batch Loss_seg={loss_seg.item():.4f},Total Loss={runing_loss/(i+1):.4f},Cur_lr={current_lr:.6f}"
            )

        current_lr = get_lr(optimizer)
        if bool(epoch % 1) is False:
            test_acc, prediction, AUC, Acc, Sen, Spe, _F1, v_loss = test(network,data_loaderTrainTest,epoch,args) # train
            test_acc, prediction, AUC, Acc, Sen, Spe, _F1, v_loss = test(network,dataloader_test,epoch,args) # val
            test_acc1, prediction1, AUC1, Acc1, Sen1, Spe1, _F11, v_loss1 = test(network,data_loaderTest,epoch,args) # test
            
            train_loss.append(runing_loss/len(data_loaderTrain.dataset))
            val_loss.append(v_loss)
            f1.append(_F1)
            aucs.append(AUC * 100.0)
            epochs.append(epoch + 1)
            lrs.append(current_lr)

            
            isExists = os.path.exists(save_dir)
            if not isExists:
                os.makedirs(save_dir)
                
            plot_progress(epochs, train_loss, val_loss, f1, aucs, lrs, os.path.join(save_dir))
            
            save_path = os.path.join(save_dir, 'model_final.pt'.format(epoch + 1, i + 1,AUC))

            performance_up = (Acc >= best_acc) if args.model_mode == 'union' else (Acc + AUC*100 > best_acc + best_auc*100)
            if performance_up:
                best_auc = AUC
                best_acc = Acc
                best_sen = Sen
                best_spe = Spe
                best_epoch = epoch
                state_dict = network.state_dict()
                torch.save(state_dict, save_path)

                # store test set result
                record_test_auc1 = AUC1
                record_test_acc1 = Acc1

    print('-------------------')
    print('val best_auc:', best_auc)
    print('val best_acc:', best_acc)
    print('val best_epoch:', best_epoch)
    print('-------------------')
    print('test auc:', record_test_auc1)
    print('test acc:', record_test_acc1)
    print('-------------------')

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

    #GPU
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # batch run
    # device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu") # single run

    save_dir = args.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    Logger(train_log_path)
    
    print(args)

    if args.model_mode == 'union':
        model = union_model(d_model=args.d_model, layer_num=args.layer_num)
    elif args.model_mode == 'tabular':
        model = tabular_model(d_model=args.d_model)
    elif args.model_mode == 'image':
        model = image_model(d_model=args.d_model)

    print(model)
    model = model.to(device)
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable pytorch params:{} MB'.format(trainable_pytorch_total_params*1e-6))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total pytorch params:{} MB'.format(pytorch_total_params*1e-6))

    ct_data_path = '/home/zhangyinan/2024_05/processed_ct_image_no_resize_v2/feats'
    tabular_data_path = 'data/merged_tables.csv'

    datasetTrain = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'train')
    data_loaderTrain = DataLoader(datasetTrain, batch_size=args.batch_size,shuffle=True)

    datasetTrainTest = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'train')
    data_loaderTrainTest = DataLoader(datasetTrainTest, batch_size=1,shuffle=False)

    datasetVal = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'val')
    data_loaderVal = DataLoader(datasetVal, batch_size=1,shuffle=False)

    datasetTest = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'test')
    data_loaderTest = DataLoader(datasetTest, batch_size=1,shuffle=False)

    train(model, data_loaderTrain, data_loaderVal, data_loaderTrainTest, data_loaderTest, args)
