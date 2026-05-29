import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from models.lstm import PSL_LSTM
from models.vgg_feature_extractor import VGGFeatureExtractor
from dataloader.dataset_prep_videos import PSLVideoDataset, train_transform

CONFIG = {
    'batch_size': 8,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'num_epochs': 250,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 15,
    'min_delta': 0.001,
    'max_frames': 30,
    'sample_rate': 2,
    'vgg_checkpoint': 'checkpoints/vgg16_psl_best.pth'
}


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, video_dataset, vgg_extractor, max_frames=30):
        self.video_dataset = video_dataset
        self.vgg_extractor = vgg_extractor
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.video_dataset)
    
    def __getitem__(self, idx):
        frames, label, _ = self.video_dataset[idx]
        
        with torch.no_grad():
            features = self.vgg_extractor.extract_features(frames)
        
        num_frames, feature_dim = features.shape
        
        if num_frames < self.max_frames:
            padding = torch.zeros(self.max_frames - num_frames, feature_dim, device=features.device)
            features = torch.cat([features, padding], dim=0)
        elif num_frames > self.max_frames:
            features = features[:self.max_frames]
        
        mask = torch.ones(self.max_frames, device=features.device)
        if num_frames < self.max_frames:
            mask[num_frames:] = 0
        
        return features, label, mask


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


def test_epoch(model, dataloader, criterion, device):
    """Evaluate model on test set with detailed metrics"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, masks in tqdm(dataloader, desc='Testing'):
            features, labels, masks = features.to(device), labels.to(device), masks.to(device)
            
            outputs = model(features, mask=masks)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and labels for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / total
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, all_predictions, all_labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)
    
    vgg_checkpoint = CONFIG['vgg_checkpoint'] if os.path.exists(CONFIG['vgg_checkpoint']) else None
    vgg_extractor = VGGFeatureExtractor(checkpoint_path=vgg_checkpoint, device=device)
    
    full_dataset = PSLVideoDataset('/kaggle/input/augmented-videos-psl/Dataset_Augmented', transform=train_transform, 
                                  max_frames=CONFIG['max_frames'], sample_rate=CONFIG['sample_rate'])
    
    print(f"Dataset loaded: {len(full_dataset)} videos found")
    print(f"Label mapping: {full_dataset.label_map}")
    
    if len(full_dataset) == 0:
        print("❌ ERROR: No videos found in dataset!")
        print("Please check that:")
        print("1. 'Dataset_Augmented' directory exists")
        print("2. It contains subdirectories for each class")
        print("3. Each subdirectory has video files (.mp4, .avi, .mov, .mkv)")
        return
    
    # Three-way split: train/val/test
    total_size = len(full_dataset)
    train_size = int(CONFIG['train_split'] * total_size)
    val_size = int(CONFIG['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    if train_size == 0 or val_size == 0 or test_size == 0:
        print("❌ ERROR: Dataset too small for train/validation/test split!")
        print(f"Need at least {int(1/min(CONFIG['train_split'], CONFIG['val_split'], CONFIG['test_split']))} videos for proper split")
        return
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(FeatureDataset(train_dataset, vgg_extractor, CONFIG['max_frames']), 
                             batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(FeatureDataset(val_dataset, vgg_extractor, CONFIG['max_frames']), 
                           batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(FeatureDataset(test_dataset, vgg_extractor, CONFIG['max_frames']), 
                            batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    model = PSL_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping variables
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    epochs_without_improvement = 0
    convergence_threshold = CONFIG['min_delta']
    
    print(f"Training for up to {CONFIG['num_epochs']} epochs with early stopping...")
    print(f"Early stopping criteria: patience={CONFIG['patience']}, min_delta={convergence_threshold}")
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Train: {train_loss:.4f} loss, {train_acc:.2f}% acc | Val: {val_loss:.4f} loss, {val_acc:.2f}% acc")
        
        # Check for improvement in validation accuracy
        improved = False
        if val_acc > best_val_acc + convergence_threshold:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            epochs_without_improvement = 0
            improved = True
            
            torch.save({
                'model_state_dict': model.state_dict(), 
                'val_acc': val_acc,
                'val_loss': val_loss,
                'epoch': epoch
            }, '/kaggle/working/lstm_psl_best.pth')
            print(f"✅ Saved best model! Val Acc: {val_acc:.2f}% (+{improvement:.3f}%)")
        
        # Check for convergence (no significant improvement)
        if not improved:
            patience_counter += 1
            epochs_without_improvement += 1
            
            # Check if validation loss is at least stable (not increasing significantly)
            if val_loss > best_val_loss * 1.02:  # 2% tolerance for loss increase
                print(f"⚠️  Validation loss increasing: {val_loss:.4f} vs best {best_val_loss:.4f}")
            
            print(f"No improvement for {epochs_without_improvement} epochs (patience: {patience_counter}/{CONFIG['patience']})")
        
        # Early stopping decision
        if patience_counter >= CONFIG['patience']:
            print(f"\n🛑 Early stopping triggered!")
            print(f"   - No improvement for {CONFIG['patience']} consecutive epochs")
            print(f"   - Best validation accuracy: {best_val_acc:.3f}%")
            print(f"   - Training converged at epoch {epoch+1}")
            break
            
        # Additional convergence check: if very little improvement over longer period
        if epoch > 50 and epochs_without_improvement > CONFIG['patience'] * 2:
            print(f"\n🎯 Convergence detected!")
            print(f"   - No significant improvement for {epochs_without_improvement} epochs")
            print(f"   - Model appears to have converged")
            break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.3f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total Epochs Trained: {epoch+1}")
    print(f"Model saved at: /kaggle/working/lstm_psl_best.pth")
    
    # Final Test Evaluation
    print(f"\n{'='*60}")
    print(f"FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    # Load best model for testing
    print("Loading best model for final evaluation...")
    checkpoint = torch.load('/kaggle/working/lstm_psl_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run test evaluation
    test_loss, test_acc, predictions, true_labels = test_epoch(model, test_loader, criterion, device)
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   - Test Loss: {test_loss:.4f}")
    print(f"   - Test Accuracy: {test_acc:.3f}%")
    print(f"   - Test Samples: {len(predictions)}")
    
    # Calculate per-class accuracy
    print(f"\n📊 PER-CLASS TEST ACCURACY:")
    unique_labels = sorted(set(true_labels))
    for label_idx in unique_labels:
        class_mask = [i for i, true_label in enumerate(true_labels) if true_label == label_idx]
        if class_mask:
            class_predictions = [predictions[i] for i in class_mask]
            class_true = [true_labels[i] for i in class_mask]
            class_acc = sum(p == t for p, t in zip(class_predictions, class_true)) / len(class_mask) * 100
            class_name = full_dataset.label_map.get(label_idx, f"Class_{label_idx}")
            print(f"   - {class_name}: {class_acc:.1f}% ({len(class_mask)} samples)")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Training Accuracy: {train_acc:.3f}%")
    print(f"Validation Accuracy: {best_val_acc:.3f}%") 
    print(f"Test Accuracy: {test_acc:.3f}%")
    print(f"Total Dataset: {len(full_dataset)} videos")
    print(f"Train/Val/Test Split: {train_size}/{val_size}/{test_size}")
    print(f"{'='*60}")


