"""
Fast LSTM training using pre-extracted features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
from tqdm import tqdm
from models.lstm import PSL_LSTM

CONFIG = {
    'batch_size': 16, 'train_split': 0.8, 'num_epochs': 50, 'learning_rate': 0.001,
    'weight_decay': 1e-4, 'patience': 10, 'features_dir': 'features'
}

class PreExtractedFeaturesDataset(Dataset):
    def __init__(self, features_dir):
        self.feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.pt')]
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.feature_files[idx], weights_only=True)
        return data['features'], data['label'], data['mask']

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for features, labels, masks in tqdm(dataloader, desc='Training'):
        features, labels, masks = features.to(device), labels.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(features, mask=masks)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for features, labels, masks in tqdm(dataloader, desc='Validation'):
            features, labels, masks = features.to(device), labels.to(device), masks.to(device)
            outputs = model(features, mask=masks)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load pre-extracted features
    print(f"Loading pre-extracted features from {CONFIG['features_dir']}/")
    full_dataset = PreExtractedFeaturesDataset(CONFIG['features_dir'])
    print(f"Loaded {len(full_dataset)} samples")
    
    train_size = int(CONFIG['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                             generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    model = PSL_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc, patience_counter = 0.0, 0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Train: {train_loss:.4f} loss, {train_acc:.2f}% acc | Val: {val_loss:.4f} loss, {val_acc:.2f}% acc")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc}, 'checkpoints/lstm_psl_best.pth')
            print(f"Saved best model! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print("Early stopping!")
                break
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
