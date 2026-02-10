import os
import yaml
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score
import numpy as np
from tqdm import tqdm

from models.SMHF import SMHF
from dataloader.load_data import split_dataset, MyDataset

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
    
    train_dataset = MyDataset(train_info, args.config_path, use_seg=False)
    val_dataset = MyDataset(val_info, args.config_path, use_seg=False)
    
    batch_size = args.batch_size
    num_workers = 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SMHF(num_classes=4).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    writer = SummaryWriter(log_dir='runs/mdl_iia_experiment')
    
    epochs = args.num_epochs
    best_val_f1 = 0.0
    
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize CSV logging
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
            # Unpack batch
            # img_MG_CC, img_MG_MLO, img_US, label, clinical
            img_cc, img_mlo, img_us, labels, clinical = batch
            
            img_cc = img_cc.to(device)
            img_mlo = img_mlo.to(device)
            img_us = img_us.to(device)
            labels = labels.to(device)
            clinical = clinical.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(img_mlo, img_cc, img_us, clinical)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        epoch_rec = recall_score(all_labels, all_preds, average='macro')
        epoch_pre = precision_score(all_labels, all_preds, average='macro')
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            epoch_auc = 0.0
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('F1/train', epoch_f1, epoch)
        writer.add_scalar('Recall/train', epoch_rec, epoch)
        writer.add_scalar('Precision/train', epoch_pre, epoch)
        writer.add_scalar('AUC/train', epoch_auc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                img_cc, img_mlo, img_us, labels, clinical = batch
                
                img_cc = img_cc.to(device)
                img_mlo = img_mlo.to(device)
                img_us = img_us.to(device)
                labels = labels.to(device)
                clinical = clinical.to(device)
                
                outputs = model(img_mlo, img_cc, img_us, clinical)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().detach().numpy())
        
        val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_rec = recall_score(val_labels, val_preds, average='macro')
        val_pre = precision_score(val_labels, val_preds, average='macro')
        try:
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
        
        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Saved Best Model!")
            
    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    
    train(args)
