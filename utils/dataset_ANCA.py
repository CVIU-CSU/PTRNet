import torch.utils.data as data
import torch.nn.functional as F
import torch
import numpy as np
import os
from PIL import Image
import random
from torchvision import transforms
import csv
import torchvision.transforms.functional as TF
import nrrd
import nibabel as nib

class ANCAdataset(data.Dataset):
    def __init__(self, root, csv_path, TrainValTest = 'train'):
        self.root=root
        self.csv_path = csv_path
        self.TrainValTest = TrainValTest
        self.pathAndLabel = self.getPathAndLabel(csv_path, TrainValTest)
        self.name=None
        self.file_id = None

    def __getitem__(self, index):
        pathList = self.pathAndLabel[index]
        self.file_id = pathList[0]
        self.name=pathList[1]
        label = pathList[2]
        tabel_data=pathList[3]

        x_categ = torch.tensor(tabel_data[-9:]).float()
        x_categ = torch.nonzero(x_categ==1).flatten() - torch.tensor([0,3,5,7])
        x_numer = torch.tensor(tabel_data[:-9]).float()

        file_path=os.path.join(self.root,pathList[0])
        file_list=os.listdir(file_path)
        tensor_data=torch.load(os.path.join(file_path,file_list[0]))

        img_data = tensor_data['patch_embed']
        inner_slice_mask = tensor_data['mask_attention']
        inter_slice_mask = tensor_data['slices_weight']

        return label, x_categ, x_numer, img_data, inner_slice_mask, inter_slice_mask
        
    def __len__(self):
        return len(self.pathAndLabel)
    
    def getPathAndLabel(self, csv_path, TrainValTest):
        if TrainValTest=='train':
            train_mark=1
        elif TrainValTest=='val':
            train_mark=0
        elif TrainValTest=='test':
            train_mark=2
        items = []
        file = open(csv_path,'r')
        fileReader = csv.reader(file)
        for line in fileReader:
            if int(line[3])==train_mark:
                item_data=[]
                item_data.append(line[0])
                item_data.append(line[1])
                item_data.append(int(line[2]))
                tabel_data=[]
                for data in line[4:]:
                    tabel_data.append(float(data))
                tabel_data=np.stack(tabel_data)
                item_data.append(tabel_data)
                items.append(item_data)
        return items

    def getFileName(self):
        return self.name

    def getFileId(self):
        return self.file_id