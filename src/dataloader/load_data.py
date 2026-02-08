import yaml
import os
import torch
import json
import random
from collections import defaultdict
import numpy as np
import monai.transforms as mt

def split_dataset(clinical_path, ratio=0.8):
    '''根据临床信息划分数据集,训练集、验证集比例8:2'''
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
    def __init__(self, infos, config_path, use_seg=False):
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
        self.age_mean = config['age_mean']
        self.age_std = config['age_std']
        self.diameter_mean = config['diameter_mean']
        self.diameter_std = config['diameter_std']
        self.size = (config['img_rows'], config['img_cols'])
        
        # MONAI transformation pipeline
        # LoadImage loads data.nii.gz is (H, W, 1), we want (1, H, W).
        # We use channel_dim=-1 to treat the last dimension as channel, moving it to front.
        self.transform = mt.Compose([
            mt.LoadImage(image_only=True), 
            mt.EnsureChannelFirst(channel_dim=-1),
            mt.ScaleIntensity(),         # Scale intensity to [0, 1]
            mt.ToTensor(),               # Convert to Tensor
        ])
        
        # Mask transform: Load -> Channel First -> Threshold -> Tensor
        self.mask_transform = mt.Compose([
            mt.LoadImage(image_only=True),
            mt.EnsureChannelFirst(channel_dim=-1),
            mt.ToTensor(),
        ])

    def crop_tumor(self, img_tensor, mask_tensor):
        '''
        Resize based on the location of the tumor in the masked image.
        1. Extract an image outward from the center of the outer rectangle of the tumor with target_size.
        2. If the cropped size cannot encompass the entire tumor, use a tumor region bounding rectangle to crop the original image and expand it by 20%, then resize it to target_size(downsample).
        3. If the original image size is smaller than target_size, use zero padding.
        '''
        _, h, w = img_tensor.shape
        target_size = self.size[0]
        
        # Get tumor mask indices
        mask_np = mask_tensor[0].numpy()
        rows, cols = np.where(mask_np > 0)
        
        if len(rows) == 0:
             # No tumor found, center cropping
             center_r, center_c = h // 2, w // 2
             tumor_h, tumor_w = 0, 0
             min_r, max_r, min_c, max_c = center_r, center_r, center_c, center_c
        else:
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)
            center_r = int((min_r + max_r) // 2)
            center_c = int((min_c + max_c) // 2)
            tumor_h = int(max_r - min_r + 1)
            tumor_w = int(max_c - min_c + 1)

        # Check if 512x512 centered crop contains the tumor
        crop_half = target_size // 2
        crop_min_r = center_r - crop_half
        crop_max_r = crop_min_r + target_size
        crop_min_c = center_c - crop_half
        crop_max_c = crop_min_c + target_size
        
        # Check containment
        is_contained = (crop_min_r <= min_r and crop_max_r >= max_r and 
                        crop_min_c <= min_c and crop_max_c >= max_c)

        if is_contained:
            pad_l = max(0, -crop_min_c)
            pad_t = max(0, -crop_min_r)
            pad_r = max(0, crop_max_c - w)
            pad_b = max(0, crop_max_r - h)
            
            if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                 img_tensor = torch.nn.functional.pad(img_tensor, (pad_l, pad_r, pad_t, pad_b), value=0)
                 mask_tensor = torch.nn.functional.pad(mask_tensor, (pad_l, pad_r, pad_t, pad_b), value=0)
                 # Update coordinates after padding
                 crop_min_c += pad_l
                 crop_min_r += pad_t
            
            img_crop = img_tensor[:, crop_min_r:crop_min_r+target_size, crop_min_c:crop_min_c+target_size]
            mask_crop = mask_tensor[:, crop_min_r:crop_min_r+target_size, crop_min_c:crop_min_c+target_size]
            return img_crop, mask_crop

        else:
            expand_ratio = 0.2 # 外扩多少可修改
            exp_h = int(tumor_h * expand_ratio)
            exp_w = int(tumor_w * expand_ratio)
            
            b_min_r = max(0, min_r - exp_h)
            b_max_r = min(h, max_r + exp_h)
            b_min_c = max(0, min_c - exp_w)
            b_max_c = min(w, max_c + exp_w)
            
            img_crop = img_tensor[:, b_min_r:b_max_r, b_min_c:b_max_c]
            mask_crop = mask_tensor[:, b_min_r:b_max_r, b_min_c:b_max_c]
            
            # Resize to 512x512
            img_crop = torch.nn.functional.interpolate(img_crop.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
            mask_crop = torch.nn.functional.interpolate(mask_crop.unsqueeze(0), size=(target_size, target_size), mode='nearest').squeeze(0)
            
            return img_crop, mask_crop

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

        # Apply tumor crop logic
        img_MG_CC, mask_MG_CC = self.crop_tumor(img_MG_CC, mask_MG_CC)
        img_MG_MLO, mask_MG_MLO = self.crop_tumor(img_MG_MLO, mask_MG_MLO)
        img_US, mask_US = self.crop_tumor(img_US, mask_US)
            
        # Clinical Label, 后续需要根据损失函数修改
        label = torch.tensor(self.clinical[self.ids[index]]['label'], dtype=torch.long)
        
        if self.use_seg:
            return (img_MG_CC, mask_MG_CC), (img_MG_MLO, mask_MG_MLO), (img_US, mask_US), label, clinical
        else:
            return img_MG_CC, img_MG_MLO, img_US, label, clinical

def test():
    clinical_path = '/home/yukino/Research/MultimodalPredictionMS/configs/clinical.json'
    config_path = '/home/yukino/Research/MultimodalPredictionMS/configs/config.yaml'
    
    train_info, val_info = split_dataset(clinical_path)
    
    print(f"Train samples: {len(train_info)}")
    
    # Test with use_seg=False
    print("Testing use_seg=False...")
    train_dataset = MyDataset(train_info, config_path, use_seg=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    for batch in train_loader:
        img_cc, img_mlo, img_us, labels, clinical = batch
        print(f"img_CC: {img_cc.shape}, img_MLO: {img_mlo.shape}, img_US: {img_us.shape}, labels: {labels.shape}, clinical: {clinical.shape}")
        break  # Just one batch
    
    # Test with use_seg=True
    print("\nTesting use_seg=True...")
    train_dataset_seg = MyDataset(train_info, config_path, use_seg=True)
    train_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=2, shuffle=True)
    
    for batch in train_loader_seg:
        (img_cc, mask_cc), (img_mlo, mask_mlo), (img_us, mask_us), labels, clinical = batch
        print(f"img_CC: {img_cc.shape}, mask_CC: {mask_cc.shape}")
        print(f"img_MLO: {img_mlo.shape}, mask_MLO: {mask_mlo.shape}")
        print(f"img_US: {img_us.shape}, mask_US: {mask_us.shape}")
        print(f"labels: {labels.shape}, clinical: {clinical.shape}")
        break

if __name__ == '__main__':
    test()
