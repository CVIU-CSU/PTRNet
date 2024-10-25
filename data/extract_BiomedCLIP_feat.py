import open_clip
from PIL import Image
import torch
import os
import numpy as np
import cv2
import nibabel as nib
import random
import csv
from lungmask import LMInferer
import SimpleITK as sitk
import nrrd
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans

inferer = LMInferer()
# inferer = LMInferer(force_cpu=True)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
model.to(device)
model.eval()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

dataset_path = '/home/zhangyinan/2024_01/ANCA/data'
nrrd_path = '/home/zhangyinan/2024_01/ANCA/converted_data'
csv_path = '/home/zhangyinan/2024_01/fuxian/biomedclip_tabtransformer_test_process/merged_tables.csv'

file_out_path = './processed_ct_image_no_resize_v2'
if not os.path.exists(file_out_path):
    os.makedirs(file_out_path, exist_ok=True)
blend_out_path = os.path.join(file_out_path, 'blend_imgs')
if not os.path.exists(blend_out_path):
    os.makedirs(blend_out_path, exist_ok=True)
feat_out_path = os.path.join(file_out_path, 'feats')
if not os.path.exists(feat_out_path):
    os.makedirs(feat_out_path, exist_ok=True)

debug = False
lungs_area_thr = 10
set_seed(2024)


def get_12():
    res_dic = {}
    with open(csv_path, 'r', encoding='utf-8') as file:
        fileReader = csv.reader(file)
        for line in fileReader:
            if line[3] == '1' or line[3] == '0':
                res_dic[line[0]] = 1
            else:
                res_dic[line[0]] = 2
    return res_dic
source_dict = get_12()


def window_width_level(nii_data, mode='1'):
    if mode == '1':
        clip_data = np.clip(nii_data, -1200, 0) # -600，1200
    else:
        clip_data = np.clip(nii_data, -1350, 150) # -600, 1500
    min_v = np.min(clip_data)
    max_v = np.max(clip_data)

    clip_data = (clip_data - min_v) / (max_v - min_v)
    clip_data = clip_data*255
    return clip_data


def kmeans_get_mask(img):

    middle = img[int(0.1*img.shape[0]):int(0.9*img.shape[0]),int(0.1*img.shape[1]):int(0.9*img.shape[1])]  
    mean = np.mean(middle)  

    kmeans = KMeans(n_clusters=2,n_init='auto').fit(np.reshape(middle,[np.prod(middle.shape),1]))

    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)  
    thresh_img = np.where(img<threshold,1.0,0.0)

    eroded = morphology.erosion(thresh_img,np.ones([4,4]))  
    dilation = morphology.dilation(eroded,np.ones([10,10]))  
    labels = measure.label(dilation)  

    if debug:
        cv2.imwrite('./debug0.png', dilation*255)

    return dilation


def compute_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / float(h), [x,y, w, h]


def erase_cut(clip_data):

    binary_image = kmeans_get_mask(clip_data)
    binary_image = binary_image.astype(np.uint8)*255
    binary_image_reverse = 255 - binary_image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_image_reverse = cv2.erode(binary_image_reverse, kernel)
    if debug:
        cv2.imwrite('./debug1.png', binary_image_reverse)
    
    contours, _ = cv2.findContours(binary_image_reverse, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i,contour in enumerate(contours):
        w_h_ratio, [x, y, w, h] = compute_aspect_ratio(contour)

        x2 = x +w
        y2 = y + h

        return [x, y, x2, y2]

with torch.no_grad():
    for folder_name in os.listdir(dataset_path):
        if debug:
            folder_name = '34'
        folder_path = os.path.join(dataset_path, folder_name)
        print(folder_path)
        all_image_features = None
        all_image_mask = None
        all_slices_weight = None
        
        files = os.listdir(folder_path)
        for name in files:
            if name.endswith('.nii'):
                nii_path = os.path.join(folder_path, name)
                break
        nii_data = nib.load(nii_path).get_fdata()
        nii_data = np.transpose(nii_data, axes=(1,0,2))
        nrrd_name = os.listdir(os.path.join(nrrd_path, folder_name))[0]
        nrrd_data, option = nrrd.read(os.path.join(nrrd_path, folder_name, nrrd_name))
        nrrd_data = np.array(nrrd_data).transpose(1,0,2)
        now_source = source_dict[folder_name]

        nii_data = window_width_level(nii_data, now_source)
        input_image = sitk.ReadImage(nii_path)
        segmentation = inferer.apply(input_image)
        segmentation[segmentation != 0] = 1

        now_feat_out_path = os.path.join(feat_out_path, folder_name)
        if not os.path.exists(now_feat_out_path):
            os.makedirs(now_feat_out_path, exist_ok=True)

        now_blend_out_path = os.path.join(blend_out_path, folder_name)
        if not os.path.exists(now_blend_out_path):
            os.makedirs(now_blend_out_path, exist_ok=True)

        bboxes = [[],[],[],[]]
        for i in range(segmentation.shape[0]):
            if np.sum(segmentation[i]) <= lungs_area_thr:
                continue
            res = erase_cut(nii_data[:,:,i])
            for j in range(4):
                bboxes[j].append(res[j])
        final_bbox = [min(bbox) if i <=1 else max(bbox) for i,bbox in enumerate(bboxes)]

        saved_id = 0
        for i in range(segmentation.shape[0]):
            if np.sum(segmentation[i]) <= lungs_area_thr:
                continue
            
            [x1, y1, x2, y2] = final_bbox

            process_data = nii_data[:,:,i][y1:y2, x1:x2]
            img = Image.fromarray(process_data).convert('L')
            input = preprocess_val(img).to(device)
            input = input.unsqueeze(0)

            image_features = model.visual.trunk.extract_patch_embed(input)
            N, D = image_features.shape[0], image_features.shape[2]
            image_features = image_features[:,1:,:].reshape(N, 14, 14, D)

            if all_image_features is None:
                all_image_features = image_features
            else:
                all_image_features = torch.cat((all_image_features, image_features), dim=0)

            mask = nrrd_data[:,:,i] * 255
            mask = mask[y1:y2, x1:x2]
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            process_data = process_data.astype(np.uint8)
            process_data = cv2.cvtColor(process_data, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(process_data, contours, -1, (0, 0, 255), 2)
            final_blend_out_path = os.path.join(now_blend_out_path, str(saved_id)+'.png')
            cv2.imwrite(final_blend_out_path, process_data)

            resize_mask = cv2.resize(mask, (224, 224),interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(resize_mask)
            save_data = [1.0]
            for j in range(0,mask_tensor.shape[0],16):
                for k in range(0,mask_tensor.shape[1],16):
                    if torch.sum(mask_tensor[j:j+16,k:k+16]) > 0:
                        save_data.append(1.0)
                    else:
                        save_data.append(0.0)
            save_data = torch.tensor([save_data])
            if debug:
                debug_mask = save_data[0,1:].reshape(14,14)*255
                cv2.imwrite('debug_mask.png',debug_mask.numpy())
            all_image_mask = torch.cat((all_image_mask, save_data),dim=0) if all_image_mask is not None else save_data

            if np.sum(mask) > 0:
                weight = torch.tensor([1.0])
            else:
                weight = torch.tensor([0.0])
            all_slices_weight = torch.cat((all_slices_weight, weight),dim=0) if all_slices_weight is not None else weight

            saved_id += 1

        final_feat_out_path = os.path.join(now_feat_out_path, folder_name+'.pth')
        all_image_features = all_image_features.cpu()
        all_image_mask = all_image_mask.cpu()
        all_slices_weight = all_slices_weight.cpu()
        torch.save({'patch_embed':all_image_features,'mask_attention':all_image_mask,'slices_weight':all_slices_weight}, final_feat_out_path)

        if debug:
            exit(0)