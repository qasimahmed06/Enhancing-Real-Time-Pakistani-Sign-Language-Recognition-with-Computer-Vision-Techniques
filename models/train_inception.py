import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import timm
import sys
import os
from tqdm import tqdm
import time

# Add parent directory to path to import dataloader modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataloader.dataset_prep import PSLDataset, train_transform, val_transform

# Training Configuration
CONFIG = {
    'model_name': 'vgg16',
    'num_classes': 36,
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'train_split': 0.8,  # 80% train, 20% validation
    'num_workers': 2,
    'save_dir': 'checkpoints',
    'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"🔥 Training Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# Create save directory
os.makedirs(CONFIG['save_dir'], exist_ok=True)

def create_model(num_classes, freeze_layers=True, unfreeze_from_layer=17):
    """Create VGG16 model with option to freeze/unfreeze layers"""
    print(f"\n🔥 Loading {CONFIG['model_name']} from TIMM...")
    
    model = timm.create_model(
        CONFIG['model_name'],
        pretrained=True,
        num_classes=num_classes,
    )
    
    if freeze_layers:
        print(f"❄️  Freezing early layers, unfreezing from layer {unfreeze_from_layer}...")
        
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in model.head.parameters():
            param.requires_grad = True
        
        # Unfreeze last conv blocks
        for name, param in model.named_parameters():
            if 'features' in name:
                layer_parts = name.split('.')
                if len(layer_parts) >= 2 and layer_parts[1].isdigit():
                    layer_num = int(layer_parts[1])
                    if layer_num >= unfreeze_from_layer:
                        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded!")
    print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    
    return model

def create_dataloaders():
    """Create train and validation dataloaders"""
    print(f"\n📁 Loading PSL Dataset...")
    
    # Create full dataset
    full_dataset = PSLDataset(
        root_dir='Dataset',
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    print(f"✅ Dataset loaded: {len(full_dataset)} images")
    print(f"   Classes: {len(full_dataset.label_map)}")
    print(f"   Class mapping: {full_dataset.label_map}")
    
    # Split dataset
    train_size = int(CONFIG['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] != 'cpu' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] != 'cpu' else False
    )
    
    return train_loader, val_loader, full_dataset.label_map

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, train_acc, val_acc, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'config': CONFIG
    }
    filepath = os.path.join(CONFIG['save_dir'], filename)
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")

def main():
    print(f"\n{'='*60}")
    print(f"🚀 Starting VGG16 Training on PSL Dataset")
    print(f"{'='*60}\n")
    
    # Setup device
    device = torch.device(CONFIG['device'])
    print(f"🖥️  Using device: {device}\n")
    
    # Create dataloaders
    train_loader, val_loader, label_map = create_dataloaders()
    
    # Create model
    model = create_model(CONFIG['num_classes'], freeze_layers=True, unfreeze_from_layer=17)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    print(f"\n{'='*60}")
    print(f"🏋️  Starting Training")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}/{CONFIG['num_epochs']} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, train_acc, val_acc,
                f'vgg16_psl_best.pth'
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_acc, val_acc,
                f'vgg16_psl_epoch_{epoch+1}.pth'
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, CONFIG['num_epochs']-1, train_acc, val_acc,
        f'vgg16_psl_final.pth'
    )
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time/60:.2f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Models saved in: {CONFIG['save_dir']}/")
    print(f"\n🎯 Class mapping:")
    for idx, name in label_map.items():
        print(f"   {idx}: {name}")

if __name__ == '__main__':
    main()
