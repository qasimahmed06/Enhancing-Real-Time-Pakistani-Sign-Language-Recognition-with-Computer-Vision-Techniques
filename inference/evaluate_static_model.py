#!/usr/bin/env python3
"""
Static Model (VGG16) Evaluation Script

Calculates precision, recall, and F1-score for the VGG16 static classifier
using the PSL dataset. Generates confusion matrix and per-class metrics.

Author: Adyaan Ahmed
"""

import argparse
import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import timm
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader.dataset_prep import PSLDataset, val_transform

# IEEE-style plot configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colorblind-friendly color scheme
COLORS = {
    'primary': '#0077BB',
    'secondary': '#CC3311',
    'accent': '#009988',
}


def load_model(checkpoint_path, device, num_classes=36):
    """Load VGG16 model from checkpoint."""
    print(f"Loading VGG16 model from {checkpoint_path}...")
    
    model = timm.create_model('vgg16', pretrained=False, num_classes=num_classes)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_acc' in checkpoint:
                print(f"   Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")
            if 'epoch' in checkpoint:
                print(f"   Checkpoint epoch: {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️  Warning: Checkpoint not found, using random weights")
    
    model = model.to(device).eval()
    print(f"✅ Model loaded successfully")
    return model


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return predictions and labels."""
    print("\n🔍 Running inference on test set...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return all_preds, all_labels, all_probs


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate precision, recall, F1-score, and accuracy."""
    print("\n📊 Calculating metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class': {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(class_names))
        }
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Generate confusion matrix visualization."""
    print("\n📈 Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw counts
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=axes[0], linewidths=0.1)
    axes[0].set_xlabel('Predicted Label', fontweight='bold')
    axes[0].set_ylabel('True Label', fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, ax=axes[1], 
                linewidths=0.1, vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted Label', fontweight='bold')
    axes[1].set_ylabel('True Label', fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.pdf', format='pdf')
    plt.savefig(output_dir / 'confusion_matrix.png', format='png')
    plt.close()
    print(f"   ✓ Saved confusion_matrix.pdf/png")


def plot_per_class_metrics(metrics, output_dir):
    """Plot per-class precision, recall, and F1-score."""
    print("\n📊 Generating per-class metrics plot...")
    
    class_names = list(metrics['per_class'].keys())
    precision = [metrics['per_class'][c]['precision'] for c in class_names]
    recall = [metrics['per_class'][c]['recall'] for c in class_names]
    f1 = [metrics['per_class'][c]['f1_score'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', 
                   color=COLORS['primary'], alpha=0.85, edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x, recall, width, label='Recall',
                   color=COLORS['secondary'], alpha=0.85, edgecolor='black', linewidth=0.7)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score',
                   color=COLORS['accent'], alpha=0.85, edgecolor='black', linewidth=0.7)
    
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Per-Class Precision, Recall, and F1-Score', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, ha='center')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.pdf', format='pdf')
    plt.savefig(output_dir / 'per_class_metrics.png', format='png')
    plt.close()
    print(f"   ✓ Saved per_class_metrics.pdf/png")


def plot_metrics_summary(metrics, output_dir):
    """Plot overall metrics summary."""
    print("\n📈 Generating metrics summary plot...")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = ['Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)',
                    'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1-Score\n(Weighted)']
    metric_values = [
        metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro'],
        metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted']
    ]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']] * 2
    
    bars = ax.bar(metric_names, metric_values, color=colors, 
                  alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'Model Performance Summary (Accuracy: {metrics["accuracy"]*100:.2f}%)', 
                 fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.pdf', format='pdf')
    plt.savefig(output_dir / 'metrics_summary.png', format='png')
    plt.close()
    print(f"   ✓ Saved metrics_summary.pdf/png")


def save_metrics_report(metrics, output_dir):
    """Save detailed metrics report."""
    print("\n💾 Saving metrics report...")
    
    # JSON format
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Saved evaluation_metrics.json")
    
    # Text format
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VGG16 Static Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:              {metrics['accuracy']*100:6.2f}%\n")
        f.write(f"Precision (Macro):     {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro):        {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro):      {metrics['f1_macro']:.4f}\n")
        f.write(f"Precision (Weighted):  {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted):     {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted):   {metrics['f1_weighted']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 80 + "\n")
        
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(f"{class_name:<20} "
                   f"{class_metrics['precision']:>10.4f} "
                   f"{class_metrics['recall']:>10.4f} "
                   f"{class_metrics['f1_score']:>10.4f} "
                   f"{class_metrics['support']:>10d}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"   ✓ Saved evaluation_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VGG16 static model on PSL dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vgg16_psl_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='Dataset',
                       help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Fraction of data to use for testing')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu), auto-detected if not specified')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print("=" * 80)
    print("VGG16 Static Model Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Split: {args.test_split * 100:.0f}%")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    full_dataset = PSLDataset(args.dataset, val_transform=val_transform)
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Number of classes: {len(full_dataset.label_map)}")
    
    class_names = [full_dataset.label_map[i] for i in range(len(full_dataset.label_map))]
    
    # Split dataset
    test_size = int(args.test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type in ['cuda', 'mps'] else False
    )
    
    # Load model
    model = load_model(args.checkpoint, device, num_classes=len(class_names))
    
    # Evaluate
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device, class_names)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:              {metrics['accuracy']*100:6.2f}%")
    print(f"Precision (Macro):     {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):        {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro):      {metrics['f1_macro']:.4f}")
    print(f"Precision (Weighted):  {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted):     {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted):   {metrics['f1_weighted']:.4f}")
    print("=" * 80)
    
    # Generate visualizations
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    plot_per_class_metrics(metrics, output_dir)
    plot_metrics_summary(metrics, output_dir)
    
    # Save reports
    save_metrics_report(metrics, output_dir)
    
    print("\n" + "=" * 80)
    print(f"✅ Evaluation complete! Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
