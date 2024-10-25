import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
from models.ablation_modality_models import union_model
from utils.config import Logger
from utils.dataset_ANCA import ANCAdataset
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

from collections import defaultdict
import matplotlib.colors as mcolors

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
    parser.add_argument("--Adiscription", type=str, default='biomedclip_tabtransformer')
    parser.add_argument("--saveName", type=str, default='vis')
    parser.add_argument("--batch_size", type=int, default=1) # 2
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--d_model",type=int,default=256)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--weight_path",type=str,default='log/modality_ablation/model_final.pt')
    
    return parser.parse_args()

vis_save_path = '/home/zhangyinan/2024_05/fuxian_res/att_atten_lines_v4/'
all_nums = {'train':86,'val':38,'test':27}

plt.clf()
plt.cla()
config = {
# "font.family":'serif',
"font.family":'Times New Roman',
"font.size": 10,
"mathtext.fontset":'stix',
'font.family': 'Times New Roman'}

plt.rcdefaults()
plt.rcParams.update(config)  

fig1, ax1 = plt.subplots(1,2,figsize=(16,6),dpi=300)

colors = plt.cm.tab10.colors
new_colors = ('#A0BDD5', '#93C8c0', '#E9DDAF', '#C2C48D', '#93AC93', '#CAC8EF')

str2title = {'train': 'train','val': 'valid','test':'test'}
str2id = {'val': 0,'test':1}
plt.tight_layout()

def check_gradient(network, data_loaderTest, args, cam=None):

    if data_loaderTest.dataset.TrainValTest == 'train':
        return
    
    network.eval()

    att2num = defaultdict(int)

    tbar = tqdm(data_loaderTest)
    for j, data in enumerate(tbar):
        tbar.set_description(
            f"No. {j}/{len(data_loaderTest)}"
        )
        network.zero_grad()
        labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask = data
        x_categ, x_numer, labels, img_data = x_categ.to(device), x_numer.to(device), labels.to(device), img_data.to(device)
        inner_slice_mask, inter_slice_mask = inner_slice_mask.to(device), inter_slice_mask.to(device)

        x_numer.requires_grad = True

        outputs = network(x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='test')

        outputs = F.softmax(outputs, dim=1)
        loss = sum([output[target.item()] for target, output in zip(labels, outputs)])

        loss.backward(retain_graph=True)

        x_categ = x_categ + network.tabular_transformer.categories_offset
        gradient1 = x_numer.grad
        gradient3 = network.tabular_transformer.category_embed.weight.grad[x_categ].mean(dim=-1)

        all_attr_gradient = torch.cat((gradient1, gradient3), dim=1)
        all_attr_gradient = torch.abs(all_attr_gradient)

        all_attr_gradient = (all_attr_gradient - torch.min(all_attr_gradient)) / (torch.max(all_attr_gradient) - torch.min(all_attr_gradient) )

        all_attr_gradient = all_attr_gradient.cpu().detach()
        k = 5
        score, topk_indices = torch.topk(all_attr_gradient, k=k, dim=1)

        now_id = data_loaderTest.dataset.getFileId()

        att_names = ['White blood cells\n(109/L)','Hemoglobin\n(g/L)','Platelet\n(109/L)','Neutrophil\n(109/L)','Lymphocyte\n(109/L)',\
                     'Eosinophil\n(109/L)','Serum albumin\n(g/L)','Serum globulin\n(g/L)','ALT(U/L)','AST(U/L)','TBIL\n(umol/L)','DBIL(umol/L)',\
                     'BUN(mmol/L)','Serum creatinine\n(umol/L)','ESR(mm/h)','CRP(mg/L)','C3(mg/L)','C4(mg/L)','IgA(mg/L)','IgG(g/L)',\
                     'IgM(mg/L)','Total Prednisolone(g)','Total CTX(g)','Age','Gender','MP','PE']
        
        att2num[att_names[topk_indices[0][0]]] += 1

    sorted_dict = dict(sorted(att2num.items(), key=lambda item: item[1], reverse=True))
    now_fig_id = str2id[data_loaderTest.dataset.TrainValTest]
    ax1[now_fig_id].pie(sorted_dict.values(), labels=sorted_dict.keys(), autopct='%1.1f%%', startangle=0,pctdistance=0.8,textprops={"size":28},shadow=False, labeldistance=1.1,colors=new_colors) # default 32
    
    ax1[now_fig_id].set_title('Visualization on the {} set'.format(str2title[data_loaderTest.dataset.TrainValTest]), fontsize=36, pad=0.1)

if __name__ == '__main__':
    args = get_arguments()

    worker_init_fn(args.seed)

    saveName = args.saveName
    args.savePath = os.path.join('log', saveName)
    train_log_path = os.path.join(args.savePath, 'train.log')
    
    nclass = args.nclass

    #GPU
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")

    save_dir = args.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    Logger(train_log_path)
    
    print(args)

    model = union_model(d_model=args.d_model)

    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))

    model = model.to(device)

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

    check_gradient(model, data_loaderTrainTest, args)
    check_gradient(model, data_loaderVal, args)
    check_gradient(model, data_loaderTest, args)

    plt.savefig(vis_save_path + 'val_test.png', bbox_inches='tight')
    plt.savefig(vis_save_path + 'val_test.pdf', bbox_inches='tight')