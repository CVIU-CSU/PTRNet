import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
from models.ablation_modality_models import union_model
from utils.config import Logger
from utils.dataset_ANCA import ANCAdataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import imageio


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
    parser.add_argument("--Adiscription", type=str, default='biomedclip_tabtransformer')
    parser.add_argument("--saveName", type=str, default='vis')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--weight_path",type=str,default='log/modality_ablation/model_final.pt')
    parser.add_argument("--d_model",type=int,default=256)
    parser.add_argument("--blend_img_path",type=str,default='/home/zhangyinan/2024_05/processed_ct_image_no_resize_v2/blend_imgs')
    parser.add_argument("--out_path",type=str,default='/home/zhangyinan/2024_05/fuxian_res/new_cam_show')
    
    return parser.parse_args()


def test(network, data_loaderTest, args, cam=None):
    
    network.eval()
    
    tbar = tqdm(data_loaderTest)
    for j, data in enumerate(tbar):
        tbar.set_description(
            f"No. {j}/{len(data_loaderTest)}"
        )
        labels, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask = data
        x_categ, x_numer, labels, img_data = x_categ.to(device), x_numer.to(device), labels.to(device), img_data.to(device)
        inner_slice_mask, inter_slice_mask = inner_slice_mask.to(device), inter_slice_mask.to(device)
        
        heatmaps = network(x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask, mode='vis')

        N = heatmaps.shape[0]
        heatmaps[heatmaps>=0.5] = 1
        heatmaps[heatmaps<0.5] = 0
        heatmaps = heatmaps.cpu().detach().numpy()
        now_id = data_loaderTest.dataset.getFileId()
        now_out_path = os.path.join(args.out_path, now_id)
        if not os.path.exists(now_out_path):
            os.makedirs(now_out_path)
        blended_images = []
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]

            now_blend_path = os.path.join(args.blend_img_path, now_id, str(i)+'.png')
            img = cv2.imread(now_blend_path)
            heatmap1 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

            heatmap1= np.uint8(255 * heatmap1)

            heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)

            blended_image = cv2.addWeighted(img, 0.7, heatmap1, 0.3, 0)

            final_out_path = os.path.join(now_out_path, str(i)+'.png')
            cv2.imwrite(final_out_path, blended_image)
            save_blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
            blended_images.append(save_blended_image)
        final_out_path = os.path.join(now_out_path, 'vis'+str(now_id)+'.gif')
        imageio.mimsave(final_out_path, blended_images, fps=20)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = args.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    Logger(train_log_path)
    
    print(args)

    model = union_model(d_model=args.d_model)

    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    model.eval()

    model = model.to(device)

    ct_data_path = '/home/zhangyinan/2024_05/processed_ct_image_no_resize_v2/feats'
    tabular_data_path = '/home/zhangyinan/2024_05/merged_tables.csv'

    datasetTrain = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'train')
    data_loaderTrain = DataLoader(datasetTrain, batch_size=args.batch_size,shuffle=True)

    datasetTrainTest = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'train')
    data_loaderTrainTest = DataLoader(datasetTrainTest, batch_size=1,shuffle=False)

    datasetVal = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'val')
    data_loaderVal = DataLoader(datasetVal, batch_size=1,shuffle=False)

    datasetTest = ANCAdataset(root=ct_data_path,csv_path=tabular_data_path,TrainValTest = 'test')
    data_loaderTest = DataLoader(datasetTest, batch_size=1,shuffle=False)

    test(model,data_loaderTrainTest,args) # train
    test(model,data_loaderVal,args) # val
    test(model,data_loaderTest,args) # test