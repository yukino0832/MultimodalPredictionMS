import os
import yaml
import datetime
import argparse
import csv
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from models.MSHF import MSHF
from models.MSHF_ViT import MSHF_ViT
from dataloader.load_data import split_dataset, MyDataset
from utils.losses import FocalLoss

def train(args):
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    clinical_path = config['clinical_dir']
    if not os.path.isabs(clinical_path):
        clinical_path = os.path.join(os.path.dirname(args.config_path), 'clinical.json')
        if not os.path.exists(clinical_path):
             clinical_path = config['clinical_dir']

    print("Loading data...")
    train_info, val_info = split_dataset(clinical_path)
    
    train_dataset = MyDataset(train_info, args.config_path, use_seg=True, is_train=True)
    val_dataset = MyDataset(val_info, args.config_path, use_seg=True, is_train=False)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    


    batch_size = args.batch_size
    num_workers = 4
    
    train_labels = [info['label'] for info in train_info]
    counts = Counter(train_labels)
    class_sample_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sample_weights = [class_sample_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if args.model == 'MSHF_ViT':
        model = MSHF_ViT(num_classes=args.num_classes, backbone=args.backbone).to(device)
    else:
        model = MSHF(num_classes=args.num_classes, backbone=args.backbone).to(device)
    for name, param in model.mg_backbone.named_parameters():
        if '7' in name.split('.')[0] or 'denseblock4' in name or 'norm5' in name or 'Mixed_7' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    for name, param in model.us_backbone.named_parameters():
        if '7' in name.split('.')[0] or 'denseblock4' in name or 'norm5' in name or 'Mixed_7' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    head_params = [p for n, p in model.named_parameters() if 'mg_backbone' not in n and 'us_backbone' not in n]
    
    mg_params = [p for p in model.mg_backbone.parameters() if p.requires_grad]
    us_params = [p for p in model.us_backbone.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': mg_params, 'lr': 1e-6},
        {'params': us_params, 'lr': 1e-6},
        {'params': head_params, 'lr': args.lr}
    ], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    total = sum(counts.values())
    num_classes_detected = args.num_classes
    alpha_vals = [total / (num_classes_detected * counts.get(i, 1)) for i in range(num_classes_detected)]
    alpha_sum  = sum(alpha_vals)
    alpha = torch.tensor([v / alpha_sum for v in alpha_vals], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-7
    )
    
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{start_time}_{args.model}_{args.backbone}"
    
    save_dir = os.path.join('checkpoints', experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = os.path.join('runs', experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    scaler = GradScaler()
    
    epochs = args.num_epochs
    best_val_auc = 0.0
    
    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Train_F1', 'Train_Rec', 'Train_Pre', 'Train_AUC',
                             'Val_Loss', 'Val_Acc', 'Val_F1', 'Val_Rec', 'Val_Pre', 'Val_AUC'])
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in loop:
            (img_cc, mask_cc), (img_mlo, mask_mlo), (img_us, mask_us), labels, clinical = batch
            
            img_cc = img_cc.to(device)
            img_mlo = img_mlo.to(device)
            img_us = img_us.to(device)
            labels = labels.to(device)
            clinical = clinical.to(device)
            mask_cc = mask_cc.to(device)
            mask_mlo = mask_mlo.to(device)
            mask_us = mask_us.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs, out_mg, out_us = model(img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
                loss_main = criterion(outputs, labels)
                loss_mg = criterion(out_mg, labels)
                loss_us = criterion(out_us, labels)
                loss = loss_main + 0.3 * loss_mg + 0.3 * loss_us
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        try:
            if args.num_classes == 2:
                epoch_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
            else:
                epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            epoch_auc = 0.0
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('F1/train', epoch_f1, epoch)
        writer.add_scalar('Recall/train', epoch_rec, epoch)
        writer.add_scalar('Precision/train', epoch_pre, epoch)
        writer.add_scalar('AUC/train', epoch_auc, epoch)
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                (img_cc, mask_cc), (img_mlo, mask_mlo), (img_us, mask_us), labels, clinical = batch
                
                img_cc = img_cc.to(device)
                img_mlo = img_mlo.to(device)
                img_us = img_us.to(device)
                labels = labels.to(device)
                clinical = clinical.to(device)
                mask_cc = mask_cc.to(device)
                mask_mlo = mask_mlo.to(device)
                mask_us = mask_us.to(device)
                
                outputs = model(img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().detach().numpy())
        
        val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_pre = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        try:
            if args.num_classes == 2:
                val_auc = roc_auc_score(val_labels, np.array(val_probs)[:, 1])
            else:
                val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro')
        except ValueError:
            val_auc = 0.0
        
        print(f"Epoch {epoch+1} Results:")
        print(f"Train - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}, Rec: {epoch_rec:.4f}, Pre: {epoch_pre:.4f}, AUC: {epoch_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Rec: {val_rec:.4f}, Pre: {val_pre:.4f}, AUC: {val_auc:.4f}")
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Recall/val', val_rec, epoch)
        writer.add_scalar('Precision/val', val_pre, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)

        # Save metrics to CSV
        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch+1, epoch_loss, epoch_acc, epoch_f1, epoch_rec, epoch_pre, epoch_auc,
                                 val_loss, val_acc, val_f1, val_rec, val_pre, val_auc])
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
            best_metrics_path = os.path.join(save_dir, 'best_metrics.txt')
            with open(best_metrics_path, 'w') as f_best:
                f_best.write(f"Best Model Metrics (Epoch {epoch+1}):\n")
                f_best.write(f"Accuracy: {val_acc:.4f}\n")
                f_best.write(f"F1 Score: {val_f1:.4f}\n")
                f_best.write(f"Recall: {val_rec:.4f}\n")
                f_best.write(f"Precision: {val_pre:.4f}\n")
                f_best.write(f"AUC: {val_auc:.4f}\n")
            
            print("Saved Best Model!")
            
    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model', type=str, default='MSHF',
                        choices=['MSHF', 'MSHF_ViT'],
                        help='Model variant to train (MSHF: original, MSHF_ViT: ViT-based fusion)')
    parser.add_argument('--backbone', type=str, default='ResNet50', 
                        choices=['ResNet50', 'DenseNet121', 'InceptionV3'],
                        help='Backbone model for feature extraction')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification')
    args = parser.parse_args()
    
    train(args)
