'''
Usage:
# 查看 ID 为 104 的乳腺钼靶 (MG) 原图（会自动展示 CC 和 MLO 两个视角）
python src/utils/show_png.py --id 104 --modality mg
# 查看 ID 为 104 的乳腺钼靶 (MG) Mask 掩码图像
python src/utils/show_png.py --id 104 --modality mg --type mask
# 查看 ID 为 104 的超声 (US) 原图
python src/utils/show_png.py --id 104 --modality us
'''

import os
import yaml
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import monai.transforms as mt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

def get_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def convert_nii_to_png(src_dir, dst_dir, is_mask=False):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    if is_mask:
        transform = mt.Compose([
            mt.LoadImage(image_only=True),
            mt.EnsureChannelFirst(channel_dim=-1)
        ])
    else:
        transform = mt.Compose([
            mt.LoadImage(image_only=True), 
            mt.EnsureChannelFirst(channel_dim=-1),
            mt.ScaleIntensity()        
        ])
    
    files = [f for f in os.listdir(src_dir) if f.endswith(".nii.gz") or f.endswith(".nii")]
    print(f"Found {len(files)} files in {src_dir}")
    for filename in files:
        file_path = os.path.join(src_dir, filename)
        img_tensor = transform(file_path)
        img_np = img_tensor[0].numpy()
        
        if is_mask:
            img_uint8 = (img_np > 0.5).astype(np.uint8) * 255
        else:
            img_uint8 = (img_np * 255.0).astype(np.uint8)
        
        png_filename = filename.replace(".nii.gz", ".png").replace(".nii", ".png")
        png_path = os.path.join(dst_dir, png_filename)
        cv2.imwrite(png_path, img_uint8)
    
    print(f"Finished converting files to {dst_dir}")

def convert_all_to_png(config_path="configs/config.yaml", output_base_dir="data/png"):
    config = get_config(config_path)
    data_dir = config['data_dir']
    
    dirs_to_convert = [
        (config['imgMG_dir'], False),
        (config['maskMG_dir'], True),
        (config['imgUS_dir'], False),
        (config['maskUS_dir'], True)
    ]
    
    for rel_dir, is_mask in dirs_to_convert:
        src_dir = os.path.join(data_dir, rel_dir)
        dst_dir = os.path.join(output_base_dir, rel_dir.replace("nii", "png"))
        print(f"Converting {src_dir} to {dst_dir}...")
        convert_nii_to_png(src_dir, dst_dir, is_mask)
        
    print("All conversions completed.")

def show_image(patient_id, modality, is_mask=False, config_path="configs/config.yaml", use_png=False):
    config = get_config(config_path)
    
    with open(config['clinical_dir'], 'r', encoding='utf-8') as f:
        clinical_info = json.load(f)
        
    patient_data = None
    for info in clinical_info:
        if str(info['id']) == str(patient_id):
            patient_data = info
            break
            
    if not patient_data:
        print(f"Patient ID {patient_id} not found in clinical data.")
        return
        
    loc = int(patient_data['tumor_location'])
    
    if use_png:
        base_dir = "data/png"
        ext = ".png"
        def map_dir(d): return d.replace("nii", "png")
    else:
        base_dir = config['data_dir']
        ext = ".nii.gz"
        def map_dir(d): return d
        
    if modality.lower() == 'mg':
        if is_mask:
            dir_path = os.path.join(base_dir, map_dir(config['maskMG_dir']))
        else:
            dir_path = os.path.join(base_dir, map_dir(config['imgMG_dir']))
            
        cc_filename = f"{patient_id}_{loc * 2 + 1}{ext}"
        mlo_filename = f"{patient_id}_{loc * 2 + 2}{ext}"
        
        paths_to_show = [
            (f"MG CC {'Mask' if is_mask else 'Orig'} (ID: {patient_id})", os.path.join(dir_path, cc_filename)),
            (f"MG MLO {'Mask' if is_mask else 'Orig'} (ID: {patient_id})", os.path.join(dir_path, mlo_filename))
        ]
        
    elif modality.lower() == 'us':
        if is_mask:
            dir_path = os.path.join(base_dir, map_dir(config['maskUS_dir']))
        else:
            dir_path = os.path.join(base_dir, map_dir(config['imgUS_dir']))
            
        us_filename = f"{patient_id}_{loc + 1}{ext}"
        
        paths_to_show = [
            (f"US {'Mask' if is_mask else 'Orig'} (ID: {patient_id})", os.path.join(dir_path, us_filename))
        ]
    else:
        print("Invalid modality. Choose 'mg' or 'us'.")
        return
        
    fig, axes = plt.subplots(1, len(paths_to_show), figsize=(6 * len(paths_to_show), 6))
    if len(paths_to_show) == 1:
        axes = [axes]
        
    for ax, (title, path) in zip(axes, paths_to_show):
        if not os.path.exists(path):
            ax.set_title(f"{title}\nNOT FOUND:\n{os.path.basename(path)}")
            ax.axis('off')
            continue
            
        if use_png:
            img = plt.imread(path)
            if len(img.shape) == 3:
                img = np.mean(img, axis=-1)
        else:
            if is_mask:
                transform = mt.Compose([
                    mt.LoadImage(image_only=True),
                    mt.EnsureChannelFirst(channel_dim=-1)
                ])
            else:
                transform = mt.Compose([
                    mt.LoadImage(image_only=True), 
                    mt.EnsureChannelFirst(channel_dim=-1),
                    mt.ScaleIntensity()        
                ])
            img_tensor = transform(path)
            img = img_tensor[0].numpy()
                
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read dataset images, convert them to PNG, or show a specific image.")
    parser.add_argument('--convert', action='store_true', help='Read all images in dataset and convert them to PNG')
    parser.add_argument('--id', type=str, help='Patient ID to show e.g. "123"')
    parser.add_argument('--modality', type=str, choices=['mg', 'us'], help='modality: mg or us')
    parser.add_argument('--type', type=str, choices=['orig', 'mask'], default='orig', help='show original image or mask (orig/mask)')
    parser.add_argument('--use_png', action='store_true', help='Read the converted .png files instead of .nii.gz when showing')
    
    args = parser.parse_args()
    
    if args.convert:
        convert_all_to_png()
        
    if args.id and args.modality:
        is_mask = (args.type == 'mask')
        show_image(args.id, args.modality, is_mask=is_mask, use_png=args.use_png)
        
    if not args.convert and not (args.id and args.modality):
        parser.print_help()
