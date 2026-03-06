import yaml
import os
import torch
import json
import random
from collections import defaultdict
import numpy as np
import cv2
import torchvision.transforms.functional as F
import monai.transforms as mt

def split_dataset(clinical_path, ratio=0.8):
    with open(clinical_path, 'r', encoding='utf-8') as f:
        clinical_info = json.load(f)

    dic = defaultdict(list)
    for info in clinical_info:
        dic[info['label']].append(info)
    
    train_info = []
    val_info = []

    for label, data in dic.items():
        random.shuffle(data)
        num_samples = len(data)
        train_num = int(ratio * num_samples)
        train_info.extend(data[:train_num])
        val_info.extend(data[train_num:])
    
    return train_info, val_info


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, infos, config_path, use_seg=False, is_train=False):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        data_dir = config['data_dir']
        img_MG = config['imgMG_dir']
        mask_MG = config['maskMG_dir']
        img_US = config['imgUS_dir']
        mask_US = config['maskUS_dir']
        
        self.clinical = {info['id']: info for info in infos}
        self.img_MG_dir = os.path.join(data_dir, img_MG)
        self.mask_MG_dir = os.path.join(data_dir, mask_MG)  
        self.img_US_dir = os.path.join(data_dir, img_US)
        self.mask_US_dir = os.path.join(data_dir, mask_US)
        self.ids = [info['id'] for info in infos]
        self.use_seg = use_seg
        self.is_train = is_train
        self.age_mean = config['age_mean']
        self.age_std = config['age_std']
        self.diameter_mean = config['diameter_mean']
        self.diameter_std = config['diameter_std']
        self.size = (config['img_rows'], config['img_cols'])
        
        self.transform = mt.Compose([
            mt.LoadImage(image_only=True), 
            mt.EnsureChannelFirst(channel_dim=-1),
            mt.ScaleIntensity(),         
            mt.ToTensor(),               
        ])
        
        self.mask_transform = mt.Compose([
            mt.LoadImage(image_only=True),
            mt.EnsureChannelFirst(channel_dim=-1),
            mt.ToTensor(),
        ])

    def process_mg(self, img_tensor, mask_tensor):
        img_np = img_tensor[0].numpy()
        mask_np = mask_tensor[0].numpy()
        
        rows, cols = np.where(mask_np > 0)
        if len(rows) > 0:
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)
            img_crop = img_np[min_r:max_r+1, min_c:max_c+1]
            mask_crop = mask_np[min_r:max_r+1, min_c:max_c+1]
        else:
            img_crop = img_np
            mask_crop = mask_np
            
        h, w = img_crop.shape
        if w > 1:
            left_sum = np.sum(img_crop[:, :w//2])
            right_sum = np.sum(img_crop[:, w//2:])
            
            if right_sum > left_sum:
                img_crop = np.fliplr(img_crop)
                mask_crop = np.fliplr(mask_crop)
            
        img_uint8 = (np.clip(img_crop, 0, 1) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_eq = clahe.apply(img_uint8)
        img_crop = img_eq.astype(np.float32) / 255.0
        
        h, w = img_crop.shape
        target_size = 512
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h > 0 and new_w > 0:
            img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = img_crop
            mask_resized = mask_crop

        h, w = img_resized.shape
        delta_w = target_size - w
        delta_h = target_size - h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        mask_padded = cv2.copyMakeBorder(mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        return torch.tensor(img_padded).unsqueeze(0), torch.tensor(mask_padded).unsqueeze(0)

    def process_us(self, img_tensor, mask_tensor):
        img_np = img_tensor[0].numpy()
        mask_np = mask_tensor[0].numpy()
        h_orig, w_orig = img_np.shape
        
        rows, cols = np.where(mask_np > 0)
        if len(rows) > 0:
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)
            
            tumor_h = max_r - min_r
            tumor_w = max_c - min_c
            
            center_r = (min_r + max_r) / 2
            center_c = (min_c + max_c) / 2
            
            new_h = tumor_h * 1.5
            new_w = tumor_w * 1.5
            
            roi_min_r = int(max(0, center_r - new_h / 2))
            roi_max_r = int(min(h_orig, center_r + new_h / 2))
            roi_min_c = int(max(0, center_c - new_w / 2))
            roi_max_c = int(min(w_orig, center_c + new_w / 2))
            
            img_crop = img_np[roi_min_r:roi_max_r, roi_min_c:roi_max_c]
            mask_crop = mask_np[roi_min_r:roi_max_r, roi_min_c:roi_max_c]
        else:
             img_crop = img_np
             mask_crop = mask_np

        h, w = img_crop.shape
        target_size = 512
        if max(h, w) > 0:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = img_crop
            mask_resized = mask_crop

        h, w = img_resized.shape
        delta_w = target_size - w
        delta_h = target_size - h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        mask_padded = cv2.copyMakeBorder(mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        return torch.tensor(img_padded).unsqueeze(0), torch.tensor(mask_padded).unsqueeze(0)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        clinical = []
        age = self.clinical[self.ids[index]]['age']
        clinical.append((age - self.age_mean) / self.age_std)
        diameter = self.clinical[self.ids[index]]['tumor_diameter_mm']
        clinical.append((diameter - self.diameter_mean) / self.diameter_std)
        clinical.append(float(self.clinical[self.ids[index]]['menopause_status']))
        loc = self.clinical[self.ids[index]]['tumor_location']
        clinical.append(float(loc))
        fiber_map = {'a类': 0, 'b类': 1, 'c类': 2, 'd类': 3}
        fiber_onehot = [0.0] * 4
        fiber_onehot[fiber_map[self.clinical[self.ids[index]]['breast_fiber_type']]] = 1.0
        clinical.extend(fiber_onehot)
        clinical = torch.tensor(clinical, dtype=torch.float32)

        img_MG_CC_path = os.path.join(self.img_MG_dir, f"{self.ids[index]}_{loc * 2 + 1}.nii.gz")
        mask_MG_CC_path = os.path.join(self.mask_MG_dir, f"{self.ids[index]}_{loc * 2 + 1}.nii.gz")
        
        img_MG_MLO_path = os.path.join(self.img_MG_dir, f"{self.ids[index]}_{loc * 2 + 2}.nii.gz")
        mask_MG_MLO_path = os.path.join(self.mask_MG_dir, f"{self.ids[index]}_{loc * 2 + 2}.nii.gz")
        
        img_US_path = os.path.join(self.img_US_dir, f"{self.ids[index]}_{loc + 1}.nii.gz")
        mask_US_path = os.path.join(self.mask_US_dir, f"{self.ids[index]}_{loc + 1}.nii.gz")
        
        img_MG_CC = self.transform(img_MG_CC_path)
        img_MG_MLO = self.transform(img_MG_MLO_path)
        img_US = self.transform(img_US_path)
        
        mask_MG_CC = self.mask_transform(mask_MG_CC_path)
        mask_MG_MLO = self.mask_transform(mask_MG_MLO_path)
        mask_US = self.mask_transform(mask_US_path)

        mask_MG_CC = (mask_MG_CC > 0.5).float()
        mask_MG_MLO = (mask_MG_MLO > 0.5).float()
        mask_US = (mask_US > 0.5).float()

        img_MG_CC, mask_MG_CC = self.process_mg(img_MG_CC, mask_MG_CC)
        img_MG_MLO, mask_MG_MLO = self.process_mg(img_MG_MLO, mask_MG_MLO)
        img_US, mask_US = self.process_us(img_US, mask_US)
        
        label = torch.tensor(self.clinical[self.ids[index]]['label'], dtype=torch.long)
        
        if self.use_seg:
            return (img_MG_CC, mask_MG_CC), (img_MG_MLO, mask_MG_MLO), (img_US, mask_US), label, clinical
        else:
            return img_MG_CC, img_MG_MLO, img_US, label, clinical
