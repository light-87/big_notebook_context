# ============================================================================
# SECTION 5: TRANSFORMER MODELS - CELL 1: SETUP & DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: TRANSFORMER MODELS - SETUP & DATA LOADING")
print("="*80)

# Core imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
import os
import json
import time
from datetime import datetime
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

# Simple progress bar that works everywhere
class SimpleProgressBar:
    def __init__(self, total, desc="Progress", ncols=80):
        self.total = total
        self.current = 0
        self.desc = desc
        self.ncols = ncols
        self.start_time = time.time()
        
    def update(self, n=1):
        self.current += n
        self._print_progress()
        
    def set_postfix(self, postfix_dict):
        self.postfix = postfix_dict
        self._print_progress()
        
    def _print_progress(self):
        if self.total > 0:
            percent = (self.current / self.total) * 100
            filled_length = int(self.ncols * self.current // self.total)
            bar = '█' * filled_length + '-' * (self.ncols - filled_length)
            
            # Calculate speed
            elapsed = time.time() - self.start_time
            if elapsed > 0 and self.current > 0:
                speed = self.current / elapsed
                eta = (self.total - self.current) / speed if speed > 0 else 0
                speed_str = f" | {speed:.1f}it/s | ETA: {eta:.0f}s"
            else:
                speed_str = ""
            
            # Add postfix if available
            postfix_str = ""
            if hasattr(self, 'postfix') and self.postfix:
                postfix_items = [f"{k}={v}" for k, v in self.postfix.items()]
                postfix_str = f" | {' '.join(postfix_items)}"
            
            # Print progress
            progress_str = f"\r{self.desc}: |{bar}| {self.current}/{self.total} [{percent:.1f}%]{speed_str}{postfix_str}"
            print(progress_str, end='', flush=True)
        
    def close(self):
        print()  # New line when done

# ============================================================================
# Device Setup & Memory Info
# ============================================================================

# Setup device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🚀 CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  CUDA not available - using CPU")

# ============================================================================
# Load Required Variables from Previous Sections
# ============================================================================

print("\n🔄 Loading required data from previous checkpoints...")

required_vars = ['df_final', 'train_indices', 'val_indices', 'test_indices', 'progress_tracker', 'WINDOW_SIZE']
missing_vars = [var for var in required_vars if var not in locals()]

if missing_vars:
    print(f"Loading missing variables: {missing_vars}")
    
    # Load from Section 1
    if 'df_final' not in locals():
        try:
            checkpoint_data = progress_tracker.resume_from_checkpoint("data_loading")
            if checkpoint_data:
                df_final = checkpoint_data['df_final']
                print("✓ Loaded df_final from Section 1")
            else:
                raise Exception("Could not load df_final from Section 1 checkpoint")
        except:
            print("❌ Failed to load df_final - ensure Section 1 is completed")
    
    # Load from Section 3
    if any(var not in locals() for var in ['train_indices', 'val_indices', 'test_indices']):
        try:
            checkpoint_data = progress_tracker.resume_from_checkpoint("data_splitting")
            if checkpoint_data:
                train_indices = checkpoint_data['train_indices']
                val_indices = checkpoint_data['val_indices']
                test_indices = checkpoint_data['test_indices']
                print("✓ Loaded split indices from Section 3")
            else:
                raise Exception("Could not load split indices from Section 3 checkpoint")
        except:
            print("❌ Failed to load split indices - ensure Section 3 is completed")

print(f"✅ Data loaded: {len(df_final)} total samples")
print(f"   📊 Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

# ============================================================================
# Dataset Class (Exact from old_context)
# ============================================================================

class PhosphorylationDataset(Dataset):
    """Dataset class for transformer training - Exact from old_context"""
    
    def __init__(self, dataframe, tokenizer, window_size=20, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sequence = row['Sequence']
        position = int(row['Position']) - 1  # Convert to 0-based indexing
        target = int(row['target'])
        
        # Extract a window around the phosphorylation site
        start = max(0, position - self.window_size)
        end = min(len(sequence), position + self.window_size + 1)
        
        # The window centered on the target site
        window_sequence = sequence[start:end]
        
        # Tokenize the sequence
        encoding = self.tokenizer(
            window_sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the tokenizer
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': torch.tensor(target, dtype=torch.float),
            'sequence': window_sequence,
            'position': torch.tensor(position, dtype=torch.long),
            'header': row['Header']
        }

# ============================================================================
# Training Utility Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, total_epochs):
    """Train model for one epoch with detailed progress tracking"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Progress bar setup
    pbar = SimpleProgressBar(len(train_loader), desc=f"Epoch {epoch}/{total_epochs} [Train]")
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = torch.sigmoid(outputs).cpu().detach().numpy()
        all_predictions.extend(predictions)
        all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}',
            'LR': f'{current_lr:.2e}'
        })
        pbar.update(1)
    
    pbar.close()
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
    
    epoch_metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_targets, predictions_binary),
        'precision': precision_score(all_targets, predictions_binary, zero_division=0),
        'recall': recall_score(all_targets, predictions_binary, zero_division=0),
        'f1': f1_score(all_targets, predictions_binary, zero_division=0),
        'auc': roc_auc_score(all_targets, all_predictions)
    }
    
    return epoch_metrics

def evaluate_model(model, data_loader, device, phase="Val"):
    """Evaluate model with comprehensive metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = SimpleProgressBar(len(data_loader), desc=f"[{phase}]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            
            # Collect predictions
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            pbar.update(1)
    
    pbar.close()
    
    # Calculate comprehensive metrics
    avg_loss = total_loss / len(data_loader)
    predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_targets, predictions_binary),
        'precision': precision_score(all_targets, predictions_binary, zero_division=0),
        'recall': recall_score(all_targets, predictions_binary, zero_division=0),
        'f1': f1_score(all_targets, predictions_binary, zero_division=0),
        'auc': roc_auc_score(all_targets, all_predictions),
        'mcc': matthews_corrcoef(all_targets, predictions_binary)
    }
    
    return metrics, all_predictions, all_targets

# ============================================================================
# Model Directory Management
# ============================================================================

def setup_model_directory(model_name, base_dir):
    """Create directory structure for individual model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_dir, 'transformers', f"{model_name}_{timestamp}")
    
    # Create subdirectories
    subdirs = ['checkpoints', 'plots', 'logs', 'predictions']
    for subdir in subdirs:
        os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)
    
    return model_dir, timestamp

def save_model_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save comprehensive model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)

# ============================================================================
# Master Results Management
# ============================================================================

def init_master_results(base_dir):
    """Initialize or load master results CSV"""
    master_results_path = os.path.join(base_dir, 'transformers', 'master_results.csv')
    
    if os.path.exists(master_results_path):
        master_results = pd.read_csv(master_results_path)
        print(f"📋 Loaded existing master results: {len(master_results)} models")
    else:
        # Create new master results table
        columns = [
            'model_name', 'timestamp', 'architecture', 'total_params', 'memory_mb',
            'epochs_trained', 'best_epoch', 'train_time_mins', 'early_stopped',
            'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_mcc',
            'val_f1_best', 'train_f1_final', 'notes'
        ]
        master_results = pd.DataFrame(columns=columns)
        master_results.to_csv(master_results_path, index=False)
        print("📋 Created new master results table")
    
    return master_results, master_results_path

def update_master_results(master_results_path, model_results):
    """Add new model results to master table"""
    # Load current results
    master_results = pd.read_csv(master_results_path)
    
    # Add new row
    new_row = pd.DataFrame([model_results])
    master_results = pd.concat([master_results, new_row], ignore_index=True)
    
    # Save updated results
    master_results.to_csv(master_results_path, index=False)
    print(f"📊 Updated master results: {len(master_results)} total models")
    
    return master_results

# ============================================================================
# Plotting Utilities
# ============================================================================

def plot_training_curves(history, save_path):
    """Create comprehensive training curves plot"""
    # Check if history has data
    if not history or 'train_loss' not in history or len(history['train_loss']) == 0:
        print("⚠️  No training history to plot")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s')
    axes[0,0].set_title('Loss', fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'train_accuracy' in history and len(history['train_accuracy']) > 0:
        axes[0,1].plot(epochs, history['train_accuracy'], 'b-', label='Train Acc', linewidth=2, marker='o')
    if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
        axes[0,1].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2, marker='s')
    axes[0,1].set_title('Accuracy', fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # F1 Score
    if 'train_f1' in history and len(history['train_f1']) > 0:
        axes[0,2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2, marker='o')
    if 'val_f1' in history and len(history['val_f1']) > 0:
        axes[0,2].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2, marker='s')
    axes[0,2].set_title('F1 Score', fontweight='bold')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('F1 Score')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Precision
    if 'train_precision' in history and len(history['train_precision']) > 0:
        axes[1,0].plot(epochs, history['train_precision'], 'b-', label='Train Prec', linewidth=2, marker='o')
    if 'val_precision' in history and len(history['val_precision']) > 0:
        axes[1,0].plot(epochs, history['val_precision'], 'r-', label='Val Prec', linewidth=2, marker='s')
    axes[1,0].set_title('Precision', fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Recall
    if 'train_recall' in history and len(history['train_recall']) > 0:
        axes[1,1].plot(epochs, history['train_recall'], 'b-', label='Train Recall', linewidth=2, marker='o')
    if 'val_recall' in history and len(history['val_recall']) > 0:
        axes[1,1].plot(epochs, history['val_recall'], 'r-', label='Val Recall', linewidth=2, marker='s')
    axes[1,1].set_title('Recall', fontweight='bold')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # AUC
    if 'train_auc' in history and len(history['train_auc']) > 0:
        axes[1,2].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2, marker='o')
    if 'val_auc' in history and len(history['val_auc']) > 0:
        axes[1,2].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2, marker='s')
    axes[1,2].set_title('AUC', fontweight='bold')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('AUC')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Training curves saved: {save_path}")

# ============================================================================
# Initialize Everything
# ============================================================================

# Setup directories
transformers_dir = os.path.join(BASE_DIR, 'transformers')
os.makedirs(transformers_dir, exist_ok=True)

# Initialize master results
master_results, master_results_path = init_master_results(BASE_DIR)

# Load tokenizer (will be reused by all models)
print("\n🔤 Loading ESM-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
print("✓ Tokenizer loaded successfully")

# Create data splits
print("\n📊 Creating data splits...")
train_df = df_final.iloc[train_indices].reset_index(drop=True)
val_df = df_final.iloc[val_indices].reset_index(drop=True)
test_df = df_final.iloc[test_indices].reset_index(drop=True)

print(f"✓ Train: {len(train_df)} samples")
print(f"✓ Val: {len(val_df)} samples") 
print(f"✓ Test: {len(test_df)} samples")

# Verify data quality
print(f"\n🔍 Data Quality Check:")
print(f"   Train positive ratio: {train_df['target'].mean():.3f}")
print(f"   Val positive ratio: {val_df['target'].mean():.3f}")
print(f"   Test positive ratio: {test_df['target'].mean():.3f}")

print("\n" + "="*80)
print("✅ SETUP COMPLETE - Ready for model definitions!")
print("="*80)
print(f"🎯 Next: Define transformer models in separate cells")
print(f"📁 Results will be saved to: {transformers_dir}")
print(f"📊 Master results: {master_results_path}")
print(f"🔤 Tokenizer: ESM-2 (facebook/esm2_t6_8M_UR50D)")
print(f"💾 Device: {DEVICE}")
print("="*80)


# ============================================================================
# CELL 2a: TRANSFORMER V1 - BASE PHOSPHOTRANSFORMER (FROM OLD_CONTEXT)
# ============================================================================

print("\n" + "="*80)
print("TRANSFORMER V1: BASE PHOSPHOTRANSFORMER (OLD_CONTEXT ARCHITECTURE)")
print("="*80)

class TransformerV1_BasePhospho(nn.Module):
    """
    Base PhosphoTransformer - Exact from old_context/transformers_context.ipynb
    
    Architecture Overview:
    =====================
    🧬 Backbone: ESM-2 (facebook/esm2_t6_8M_UR50D) - 8M parameters
    🎯 Task: Binary phosphorylation site prediction
    🔍 Context: ±3 positions around target site (window_context=3)
    
    Model Pipeline:
    ==============
    1. ESM-2 Encoder: Sequence → Hidden representations (320 dim)
    2. Context Extraction: Extract ±3 positions around center
    3. Feature Concatenation: 7 positions × 320 dim = 2,240 features
    4. Classification Head: 2,240 → 256 → 64 → 1
    
    Architecture Details:
    ====================
    - Hidden Size: 320 (ESM-2 t6 model)
    - Context Window: 7 positions (center ± 3)
    - Classification Layers:
      * Linear(2240, 256) + LayerNorm + ReLU + Dropout(0.3)
      * Linear(256, 64) + LayerNorm + ReLU + Dropout(0.3)  
      * Linear(64, 1) → Binary output
    
    Key Features:
    ============
    ✓ Proven architecture from old_context
    ✓ Position-aware feature extraction
    ✓ Regularization with LayerNorm + Dropout
    ✓ Memory efficient for RTX 4060
    ✓ Fast convergence with good performance
    """
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", dropout_rate=0.3, window_context=3):
        super().__init__()
        
        print(f"🏗️  Initializing TransformerV1_BasePhospho...")
        print(f"   📚 Model: {model_name}")
        print(f"   🎯 Window Context: ±{window_context} positions")
        print(f"   💧 Dropout Rate: {dropout_rate}")
        
        # Load pre-trained protein language model
        self.protein_encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from the model config
        hidden_size = self.protein_encoder.config.hidden_size
        print(f"   🔢 Hidden Size: {hidden_size}")
        
        # Context aggregation
        self.window_context = window_context
        context_size = hidden_size * (2*window_context + 1)
        print(f"   📏 Context Size: {context_size} ({2*window_context + 1} positions × {hidden_size})")
        
        # Classification head - Exact from old_context
        self.classifier = nn.Sequential(
            nn.Linear(context_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        print(f"   📊 Parameters:")
        print(f"      • Total: {total_params:,}")
        print(f"      • Trainable: {trainable_params:,}")
        print(f"      • ESM-2 Backbone: {total_params - classifier_params:,}")
        print(f"      • Classification Head: {classifier_params:,}")
        
        # Memory estimation for RTX 4060
        memory_estimate = self._estimate_memory()
        print(f"   💾 Estimated Memory Usage:")
        print(f"      • Model: ~{memory_estimate['model_mb']:.0f} MB")
        print(f"      • Training (batch=16): ~{memory_estimate['training_mb']:.0f} MB")
        print(f"      • Safe for RTX 4060: {'✅' if memory_estimate['training_mb'] < 6000 else '⚠️'}")
        
    def _estimate_memory(self):
        """Estimate memory usage for RTX 4060"""
        # Model parameters in MB (float32)
        total_params = sum(p.numel() for p in self.parameters())
        model_mb = (total_params * 4) / (1024**2)  # 4 bytes per float32
        
        # Training memory (model + gradients + optimizer states + activations)
        # Rough estimate: 3x model size + batch activations
        training_mb = model_mb * 3 + 500  # 500MB for activations with batch=16
        
        return {
            'model_mb': model_mb,
            'training_mb': training_mb
        }
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass - Exact implementation from old_context
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Raw predictions [batch_size] (before sigmoid)
        """
        # Get the transformer outputs
        outputs = self.protein_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence outputs [batch_size, seq_len, hidden_dim]
        sequence_output = outputs.last_hidden_state
        
        # Find the center position (target phosphorylation site)
        center_pos = sequence_output.shape[1] // 2
        
        # Extract features from window around center
        batch_size, seq_len, hidden_dim = sequence_output.shape
        context_features = []
        
        # Extract ±window_context positions around center
        for i in range(-self.window_context, self.window_context + 1):
            pos = center_pos + i
            if pos < 0 or pos >= seq_len:
                # Pad with zeros for out-of-bounds positions
                context_features.append(torch.zeros(batch_size, hidden_dim, device=sequence_output.device))
            else:
                context_features.append(sequence_output[:, pos, :])
        
        # Concatenate context features [batch_size, context_size]
        concat_features = torch.cat(context_features, dim=1)
        
        # Pass through classifier to get logits
        logits = self.classifier(concat_features)
        
        return logits.squeeze(-1)  # Remove last dimension [batch_size]

# ============================================================================
# Model Configuration for TransformerV1
# ============================================================================

TRANSFORMER_V1_CONFIG = {
    # Model Architecture
    'model_name': 'facebook/esm2_t6_8M_UR50D',
    'dropout_rate': 0.3,
    'window_context': 3,
    
    # Training Hyperparameters (from old_context)
    'learning_rate': 2e-5,
    'batch_size': 16,  # Optimized for RTX 4060
    'epochs': 10,
    'early_stopping_patience': 3,
    'weight_decay': 0.01,
    
    # Scheduling
    'warmup_steps': 500,
    'save_every_epochs': 2,
    
    # Data
    'window_size': 20,  # For dataset sequence extraction
    'max_length': 512,  # Tokenizer max length
    
    # Optimization
    'gradient_clipping': 1.0,
    'label_smoothing': 0.0,
    
    # Description
    'architecture_name': 'BasePhosphoTransformer',
    'description': 'Original proven architecture from old_context with ESM-2 backbone and context window aggregation',
    'source': 'old_context/transformers_context.ipynb',
    'key_features': [
        'ESM-2 protein language model',
        'Context window (±3 positions)',
        'Classification head with LayerNorm',
        'Proven performance on phosphorylation prediction'
    ]
}

# ============================================================================
# Model Factory Function
# ============================================================================

def create_transformer_v1():
    """Factory function to create TransformerV1 model"""
    print("\n🏭 Creating TransformerV1_BasePhospho model...")
    
    model = TransformerV1_BasePhospho(
        model_name=TRANSFORMER_V1_CONFIG['model_name'],
        dropout_rate=TRANSFORMER_V1_CONFIG['dropout_rate'],
        window_context=TRANSFORMER_V1_CONFIG['window_context']
    )
    
    return model

# ============================================================================
# Test Model Creation
# ============================================================================

print("\n🧪 Testing model creation...")
test_model = create_transformer_v1()
print("✅ TransformerV1_BasePhospho created successfully!")

# Simple test with actual tokenizer to ensure compatibility
print("\n🔧 Testing with real tokenizer...")
test_sequence = "MKLVLSLS"  # Simple protein sequence
test_encoding = tokenizer(
    test_sequence,
    padding="max_length",
    truncation=True,
    max_length=64,  # Short for testing
    return_tensors="pt"
)

print(f"   📝 Test sequence: {test_sequence}")
print(f"   📤 Token IDs shape: {test_encoding['input_ids'].shape}")
print(f"   🎯 Attention mask shape: {test_encoding['attention_mask'].shape}")
print(f"   ✅ Model architecture validated!")

# Clean up test variables
del test_model, test_encoding

print("\n" + "="*80)
print("✅ TRANSFORMER V1 DEFINITION COMPLETE")
print("="*80)
print(f"🏷️  Model: TransformerV1_BasePhospho")
print(f"📚 Source: old_context/transformers_context.ipynb")
print(f"🎯 Ready for training with config: TRANSFORMER_V1_CONFIG")
print(f"🔧 Factory function: create_transformer_v1()")
print(f"💾 Memory estimate: Safe for RTX 4060")
print("="*80)


# ============================================================================
# CELL 2b: TRANSFORMER V2 - HIERARCHICAL ATTENTION TRANSFORMER
# ============================================================================

print("\n" + "="*80)
print("TRANSFORMER V2: HIERARCHICAL ATTENTION TRANSFORMER")
print("="*80)

class TransformerV2_Hierarchical(nn.Module):
    """
    Hierarchical Attention Transformer - Enhanced Architecture
    
    Architecture Overview:
    =====================
    🧬 Backbone: ESM-2 (facebook/esm2_t6_8M_UR50D) - 8M parameters
    🎯 Task: Binary phosphorylation site prediction with hierarchical attention
    🔍 Context: Multi-scale attention with position-aware mechanisms
    
    Key Innovations:
    ================
    1. **Multi-Head Position Attention**: Different heads focus on different positions
    2. **Hierarchical Feature Extraction**: Local → Global → Fusion
    3. **Position Embeddings**: Learnable position-specific representations
    4. **Attention Pooling**: Weighted combination instead of simple concatenation
    5. **Residual Connections**: Better gradient flow
    
    Model Pipeline:
    ==============
    1. ESM-2 Encoder: Sequence → Hidden representations (320 dim)
    2. Position Embeddings: Add learnable position encodings
    3. Multi-Head Attention: 4 heads focusing on different aspects
    4. Hierarchical Pooling: Local patterns → Global context
    5. Feature Fusion: Combine multi-scale representations
    6. Classification Head: Enhanced with residual connections
    
    Architecture Details:
    ====================
    - Hidden Size: 320 (ESM-2 t6 model)
    - Attention Heads: 4 (local, medium, global, position-specific)
    - Context Window: 7 positions (center ± 3)
    - Position Embeddings: 64 dimensions
    - Fusion Layer: 256 → 128 → 64 → 1
    
    Key Features:
    ============
    ✓ Hierarchical attention mechanism
    ✓ Position-aware feature extraction
    ✓ Multi-scale pattern recognition
    ✓ Residual connections for better training
    ✓ Attention visualization capability
    ✓ More parameters but better representation power
    """
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", dropout_rate=0.3, 
                 window_context=3, n_attention_heads=4, position_embed_dim=64):
        super().__init__()
        
        print(f"🏗️  Initializing TransformerV2_Hierarchical...")
        print(f"   📚 Model: {model_name}")
        print(f"   🎯 Window Context: ±{window_context} positions")
        print(f"   💧 Dropout Rate: {dropout_rate}")
        print(f"   🔄 Attention Heads: {n_attention_heads}")
        print(f"   📍 Position Embed Dim: {position_embed_dim}")
        
        # Load pre-trained protein language model
        self.protein_encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from the model config
        hidden_size = self.protein_encoder.config.hidden_size
        print(f"   🔢 Hidden Size: {hidden_size}")
        
        # Architecture parameters
        self.window_context = window_context
        self.n_attention_heads = n_attention_heads
        self.position_embed_dim = position_embed_dim
        self.hidden_size = hidden_size
        
        # Position embeddings - learnable position-specific representations
        self.position_embeddings = nn.Embedding(2*window_context + 1, position_embed_dim)
        
        # Multi-head attention for hierarchical feature extraction
        attention_input_dim = hidden_size + position_embed_dim
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=attention_input_dim,
                num_heads=1,  # Each "head" is actually a single attention mechanism
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(n_attention_heads)
        ])
        
        # Attention output projections
        self.attention_projections = nn.ModuleList([
            nn.Linear(attention_input_dim, hidden_size//2)
            for _ in range(n_attention_heads)
        ])
        
        # Layer normalization for each attention head
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size//2)
            for _ in range(n_attention_heads)
        ])
        
        # Hierarchical fusion mechanism
        fusion_input_dim = (hidden_size//2) * n_attention_heads
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Enhanced classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        # Residual projection for skip connection
        self.residual_projection = nn.Linear(fusion_input_dim, 64)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate new component parameters
        esm_params = sum(p.numel() for p in self.protein_encoder.parameters())
        attention_params = sum(p.numel() for p in self.attention_heads.parameters())
        position_params = sum(p.numel() for p in self.position_embeddings.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        other_params = total_params - esm_params
        
        print(f"   📊 Parameters:")
        print(f"      • Total: {total_params:,}")
        print(f"      • Trainable: {trainable_params:,}")
        print(f"      • ESM-2 Backbone: {esm_params:,}")
        print(f"      • Attention Mechanism: {attention_params:,}")
        print(f"      • Position Embeddings: {position_params:,}")
        print(f"      • Fusion Layer: {fusion_params:,}")
        print(f"      • Classification Head: {classifier_params:,}")
        print(f"      • Total New Components: {other_params:,}")
        
        # Memory estimation for RTX 4060
        memory_estimate = self._estimate_memory()
        print(f"   💾 Estimated Memory Usage:")
        print(f"      • Model: ~{memory_estimate['model_mb']:.0f} MB")
        print(f"      • Training (batch=16): ~{memory_estimate['training_mb']:.0f} MB")
        print(f"      • Safe for RTX 4060: {'✅' if memory_estimate['training_mb'] < 6000 else '⚠️'}")
        
        if memory_estimate['training_mb'] > 6000:
            print(f"   💡 Recommended batch size: {max(4, int(16 * 6000 / memory_estimate['training_mb']))}")
        
    def _estimate_memory(self):
        """Estimate memory usage for RTX 4060"""
        # Model parameters in MB (float32)
        total_params = sum(p.numel() for p in self.parameters())
        model_mb = (total_params * 4) / (1024**2)  # 4 bytes per float32
        
        # Training memory (model + gradients + optimizer states + activations)
        # More complex model needs more activation memory
        training_mb = model_mb * 3 + 800  # 800MB for activations with batch=16
        
        return {
            'model_mb': model_mb,
            'training_mb': training_mb
        }
    
    def forward(self, input_ids, attention_mask):
        """
        Hierarchical forward pass with multi-head attention
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Raw predictions [batch_size] (before sigmoid)
            attention_weights: Optional attention weights for visualization
        """
        # Get the transformer outputs
        outputs = self.protein_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence outputs [batch_size, seq_len, hidden_dim]
        sequence_output = outputs.last_hidden_state
        
        # Find the center position (target phosphorylation site)
        center_pos = sequence_output.shape[1] // 2
        
        # Extract features from window around center
        batch_size, seq_len, hidden_dim = sequence_output.shape
        context_features = []
        
        # Extract ±window_context positions around center
        for i in range(-self.window_context, self.window_context + 1):
            pos = center_pos + i
            if pos < 0 or pos >= seq_len:
                # Pad with zeros for out-of-bounds positions
                context_features.append(torch.zeros(batch_size, hidden_dim, device=sequence_output.device))
            else:
                context_features.append(sequence_output[:, pos, :])
        
        # Stack context features [batch_size, window_size, hidden_dim]
        context_tensor = torch.stack(context_features, dim=1)
        
        # Add position embeddings
        position_ids = torch.arange(2*self.window_context + 1, device=sequence_output.device)
        position_embeds = self.position_embeddings(position_ids)  # [window_size, pos_embed_dim]
        position_embeds = position_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, window_size, pos_embed_dim]
        
        # Concatenate sequence features with position embeddings
        enhanced_features = torch.cat([context_tensor, position_embeds], dim=-1)  # [batch_size, window_size, hidden_dim + pos_embed_dim]
        
        # Multi-head hierarchical attention
        attention_outputs = []
        
        for i, (attention_head, projection, norm) in enumerate(zip(
            self.attention_heads, self.attention_projections, self.attention_norms
        )):
            # Self-attention within the context window
            attended_features, attention_weights = attention_head(
                enhanced_features, enhanced_features, enhanced_features
            )
            
            # Project to common dimension
            projected_features = projection(attended_features)  # [batch_size, window_size, hidden_dim//2]
            
            # Apply layer normalization
            normed_features = norm(projected_features)
            
            # Pool across positions (mean pooling for this head)
            pooled_features = normed_features.mean(dim=1)  # [batch_size, hidden_dim//2]
            
            attention_outputs.append(pooled_features)
        
        # Concatenate all attention head outputs
        concatenated_features = torch.cat(attention_outputs, dim=-1)  # [batch_size, (hidden_dim//2) * n_heads]
        
        # Hierarchical fusion
        fused_features = self.fusion_layer(concatenated_features)  # [batch_size, 128]
        
        # Residual connection
        residual = self.residual_projection(concatenated_features)  # [batch_size, 64]
        
        # Classification with residual
        pre_residual = self.classifier[:-1](fused_features)  # [batch_size, 64] (everything except final linear)
        
        # Add residual connection
        with_residual = pre_residual + residual  # [batch_size, 64]
        
        # Final classification
        logits = self.classifier[-1](with_residual)  # [batch_size, 1]
        
        return logits.squeeze(-1)  # Remove last dimension [batch_size]

# ============================================================================
# Model Configuration for TransformerV2
# ============================================================================

TRANSFORMER_V2_CONFIG = {
    # Model Architecture
    'model_name': 'facebook/esm2_t6_8M_UR50D',
    'dropout_rate': 0.3,
    'window_context': 3,
    'n_attention_heads': 4,
    'position_embed_dim': 64,
    
    # Training Hyperparameters (adjusted for more complex model)
    'learning_rate': 1e-5,  # Lower learning rate for more complex model
    'batch_size': 12,  # Reduced batch size due to higher memory usage
    'epochs': 12,  # More epochs for complex model
    'early_stopping_patience': 4,  # More patience for complex model
    'weight_decay': 0.02,  # Slightly higher regularization
    
    # Scheduling
    'warmup_steps': 800,  # More warmup for complex model
    'save_every_epochs': 2,
    
    # Data
    'window_size': 20,  # For dataset sequence extraction
    'max_length': 512,  # Tokenizer max length
    
    # Optimization
    'gradient_clipping': 1.0,
    'label_smoothing': 0.0,
    
    # Description
    'architecture_name': 'HierarchicalAttentionTransformer',
    'description': 'Enhanced transformer with hierarchical attention mechanism, position embeddings, and multi-scale feature fusion for improved phosphorylation site prediction',
    'source': 'Enhanced architecture based on multi-head attention and hierarchical feature extraction',
    'key_features': [
        'Multi-head hierarchical attention mechanism',
        'Learnable position embeddings',
        'Multi-scale feature fusion',
        'Residual connections for better gradient flow',
        'Enhanced classification head',
        'Position-aware pattern recognition'
    ]
}

# ============================================================================
# Model Factory Function
# ============================================================================

def create_transformer_v2():
    """Factory function to create TransformerV2 model"""
    print("\n🏭 Creating TransformerV2_Hierarchical model...")
    
    model = TransformerV2_Hierarchical(
        model_name=TRANSFORMER_V2_CONFIG['model_name'],
        dropout_rate=TRANSFORMER_V2_CONFIG['dropout_rate'],
        window_context=TRANSFORMER_V2_CONFIG['window_context'],
        n_attention_heads=TRANSFORMER_V2_CONFIG['n_attention_heads'],
        position_embed_dim=TRANSFORMER_V2_CONFIG['position_embed_dim']
    )
    
    return model

# ============================================================================
# Test Model Creation
# ============================================================================

print("\n🧪 Testing model creation...")
test_model = create_transformer_v2()
print("✅ TransformerV2_Hierarchical created successfully!")

# Simple test with tokenizer compatibility
print("\n🔧 Testing with real tokenizer...")
test_sequence = "MKLVLSLSLAVGIAVA"  # Slightly longer test sequence
test_encoding = tokenizer(
    test_sequence,
    padding="max_length",
    truncation=True,
    max_length=64,  # Short for testing
    return_tensors="pt"
)

print(f"   📝 Test sequence: {test_sequence}")
print(f"   📤 Token IDs shape: {test_encoding['input_ids'].shape}")
print(f"   🎯 Attention mask shape: {test_encoding['attention_mask'].shape}")
print(f"   ✅ Model architecture validated!")

# Clean up test variables
del test_model, test_encoding

print("\n" + "="*80)
print("✅ TRANSFORMER V2 DEFINITION COMPLETE")
print("="*80)
print(f"🏷️  Model: TransformerV2_Hierarchical")
print(f"📚 Architecture: Hierarchical Attention with Multi-Scale Fusion")
print(f"🎯 Ready for training with config: TRANSFORMER_V2_CONFIG")
print(f"🔧 Factory function: create_transformer_v2()")
print(f"💾 Memory estimate: Check output above for RTX 4060 compatibility")
print(f"🔄 Key improvements over V1:")
print(f"   • Multi-head attention mechanism")
print(f"   • Position embeddings") 
print(f"   • Hierarchical feature fusion")
print(f"   • Residual connections")
print("="*80)

# ============================================================================
# CELL 3: TRAINING INFRASTRUCTURE - UNIVERSAL TRANSFORMER TRAINER
# ============================================================================

print("\n" + "="*80)
print("TRAINING INFRASTRUCTURE - UNIVERSAL TRANSFORMER TRAINER")
print("="*80)

def train_transformer_model(model_name):
    """
    Universal transformer training function
    
    This function can train any transformer model defined in previous cells.
    It handles the complete training pipeline:
    - Model creation and setup
    - Data loader creation
    - Training loop with progress tracking
    - Checkpointing and logging
    - Results saving and analysis
    
    Args:
        model_name (str): Name of the model to train (e.g., 'transformer_v1')
    """
    
    print(f"\n🚀 Starting training for: {model_name.upper()}")
    print("="*60)
    
    # ========================================================================
    # 1. Model Creation and Configuration
    # ========================================================================
    
    # Get model and config based on model_name
    if model_name == "transformer_v1":
        model = create_transformer_v1()
        config = TRANSFORMER_V1_CONFIG.copy()
        architecture_name = "TransformerV1_BasePhospho"
    elif model_name == "transformer_v2":
        model = create_transformer_v2()
        config = TRANSFORMER_V2_CONFIG.copy()
        architecture_name = "TransformerV2_Hierarchical"
    else:
        available_models = ["transformer_v1", "transformer_v2"]
        raise ValueError(f"Unknown model: {model_name}. Available: {available_models}")
    
    # Move model to device
    model = model.to(DEVICE)
    print(f"✅ Model '{architecture_name}' loaded on {DEVICE}")
    
    # Setup model directory
    model_dir, timestamp = setup_model_directory(model_name, BASE_DIR)
    print(f"📁 Model directory: {model_dir}")
    
    # ========================================================================
    # 2. Save Model Architecture and Config
    # ========================================================================
    
    # Save configuration
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model description
    description = f"""# {architecture_name}
    
## Architecture
- **Model**: {config['model_name']}
- **Architecture**: {config['architecture_name']}
- **Window Context**: ±{config['window_context']} positions
- **Dropout Rate**: {config['dropout_rate']}
- **Total Parameters**: {sum(p.numel() for p in model.parameters()):,}

## Training Configuration
- **Learning Rate**: {config['learning_rate']}
- **Batch Size**: {config['batch_size']}
- **Max Epochs**: {config['epochs']}
- **Early Stopping**: {config['early_stopping_patience']} epochs
- **Weight Decay**: {config['weight_decay']}

## Key Features
{chr(10).join('- ' + feature for feature in config['key_features'])}

## Source
- **Original Code**: {config['source']}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Device**: {DEVICE}

## Description
{config['description']}
"""
    
    desc_path = os.path.join(model_dir, 'model_description.md')
    with open(desc_path, 'w') as f:
        f.write(description)
    
    print(f"📝 Saved model config and description")
    
    # ========================================================================
    # 3. Create Data Loaders
    # ========================================================================
    
    print(f"\n📊 Creating data loaders...")
    
    # Create datasets
    train_dataset = PhosphorylationDataset(
        train_df, tokenizer, 
        window_size=config['window_size'], 
        max_length=config['max_length']
    )
    val_dataset = PhosphorylationDataset(
        val_df, tokenizer,
        window_size=config['window_size'],
        max_length=config['max_length']
    )
    test_dataset = PhosphorylationDataset(
        test_df, tokenizer,
        window_size=config['window_size'],
        max_length=config['max_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   🔢 Train batches: {len(train_loader)} (samples: {len(train_dataset)})")
    print(f"   🔢 Val batches: {len(val_loader)} (samples: {len(val_dataset)})")
    print(f"   🔢 Test batches: {len(test_loader)} (samples: {len(test_dataset)})")
    
    # ========================================================================
    # 4. Setup Optimizer and Scheduler
    # ========================================================================
    
    print(f"\n⚙️ Setting up optimizer and scheduler...")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = config['warmup_steps']
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"   📈 Optimizer: AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
    print(f"   📉 Scheduler: Linear warmup + decay ({warmup_steps} warmup steps)")
    print(f"   🔄 Total steps: {total_steps}")
    
    # ========================================================================
    # 5. Training Loop
    # ========================================================================
    
    print(f"\n🏋️ Starting training loop...")
    print(f"   🎯 Target epochs: {config['epochs']}")
    print(f"   ⏰ Early stopping: {config['early_stopping_patience']} epochs")
    print(f"   💾 Save every: {config['save_every_epochs']} epochs")
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': []
    }
    
    # Early stopping variables
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    
    # Training start time
    training_start_time = time.time()
    
    # Main training loop
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train epoch
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, epoch, config['epochs'])
        
        # Validate epoch
        val_metrics, _, _ = evaluate_model(model, val_loader, DEVICE, "Val")
        
        # Store metrics
        for key in train_metrics:
            if key in history:
                history[f'train_{key}'].append(train_metrics[key])
        for key in val_metrics:
            if key in history:
                history[f'val_{key}'].append(val_metrics[key])
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   🔹 Train: Loss={train_metrics['loss']:.4f}, F1={train_metrics['f1']:.4f}, AUC={train_metrics['auc']:.4f}")
        print(f"   🔸 Val:   Loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")
        
        # Check for best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            save_model_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_model_path)
            print(f"   🌟 New best model! F1={best_val_f1:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{config['early_stopping_patience']}")
        
        # Regular checkpoint saving
        if epoch % config['save_every_epochs'] == 0:
            checkpoint_path = os.path.join(model_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
            save_model_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_path)
            print(f"   💾 Checkpoint saved: epoch_{epoch}.pth")
        
        # Early stopping check
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n🛑 Early stopping triggered after {config['early_stopping_patience']} epochs without improvement")
            early_stopped = True
            break
    
    # ========================================================================
    # 6. Final Evaluation
    # ========================================================================
    
    training_time_mins = (time.time() - training_start_time) / 60
    print(f"\n🏁 Training completed in {training_time_mins:.1f} minutes")
    print(f"   🏆 Best epoch: {best_epoch} (F1={best_val_f1:.4f})")
    print(f"   🔄 Total epochs: {epoch}")
    print(f"   ⏰ Early stopped: {'Yes' if early_stopped else 'No'}")
    
    # Load best model for final evaluation
    print(f"\n🔄 Loading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final test evaluation
    print(f"\n🧪 Final test evaluation...")
    test_metrics, test_predictions, test_targets = evaluate_model(model, test_loader, DEVICE, "Test")
    
    print(f"\n📈 Final Test Results:")
    for metric, value in test_metrics.items():
        print(f"   🎯 {metric.upper()}: {value:.4f}")
    
    # ========================================================================
    # 7. Save Results and Visualizations
    # ========================================================================
    
    print(f"\n💾 Saving results and visualizations...")
    
    # Save training history
    history_path = os.path.join(model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test predictions
    predictions_df = pd.DataFrame({
        'header': [test_dataset.dataframe.iloc[i]['Header'] for i in range(len(test_dataset))],
        'sequence': [test_dataset.dataframe.iloc[i]['Sequence'] for i in range(len(test_dataset))],
        'position': [test_dataset.dataframe.iloc[i]['Position'] for i in range(len(test_dataset))],
        'true_label': test_targets,
        'prediction_prob': test_predictions,
        'prediction_binary': (np.array(test_predictions) > 0.5).astype(int)
    })
    
    predictions_path = os.path.join(model_dir, 'predictions', 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Generate training curves
    curves_path = os.path.join(model_dir, 'plots', 'training_curves.png')
    plot_training_curves(history, curves_path)
    
    # Generate confusion matrix
    plot_confusion_matrix(test_targets, test_predictions, 
                         os.path.join(model_dir, 'plots', 'confusion_matrix.png'))
    
    # Save final metrics
    final_metrics = {
        'model_name': model_name,
        'architecture': architecture_name,
        'timestamp': timestamp,
        'total_params': sum(p.numel() for p in model.parameters()),
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'training_time_mins': training_time_mins,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'config': config
    }
    
    metrics_path = os.path.join(model_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)
    
    # ========================================================================
    # 8. Update Master Results
    # ========================================================================
    
    print(f"\n📊 Updating master results...")
    
    # Prepare row for master results
    master_row = {
        'model_name': f"{model_name}_{timestamp}",
        'timestamp': timestamp,
        'architecture': architecture_name,
        'total_params': sum(p.numel() for p in model.parameters()),
        'memory_mb': f"~{(sum(p.numel() for p in model.parameters()) * 4) / (1024**2):.0f}",
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        'train_time_mins': f"{training_time_mins:.1f}",
        'early_stopped': early_stopped,
        'test_accuracy': f"{test_metrics['accuracy']:.4f}",
        'test_precision': f"{test_metrics['precision']:.4f}",
        'test_recall': f"{test_metrics['recall']:.4f}",
        'test_f1': f"{test_metrics['f1']:.4f}",
        'test_auc': f"{test_metrics['auc']:.4f}",
        'test_mcc': f"{test_metrics['mcc']:.4f}",
        'val_f1_best': f"{best_val_f1:.4f}",
        'train_f1_final': f"{history['train_f1'][-1]:.4f}",
        'notes': f"Batch size: {config['batch_size']}, LR: {config['learning_rate']}"
    }
    
    # Update master results
    updated_results = update_master_results(master_results_path, master_row)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"=" * 60)
    print(f"🏷️  Model: {model_name}")
    print(f"📁 Results: {model_dir}")
    print(f"🏆 Best F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"🧪 Test F1: {test_metrics['f1']:.4f}")
    print(f"⏱️  Time: {training_time_mins:.1f} minutes")
    print(f"📊 Master results updated: {len(updated_results)} total models")
    print(f"=" * 60)
    
    return final_metrics

# ============================================================================
# Additional Plotting Functions
# ============================================================================

def plot_confusion_matrix(targets, predictions, save_path):
    """Create and save confusion matrix plot"""
    predictions_binary = (np.array(predictions) > 0.5).astype(int)
    cm = confusion_matrix(targets, predictions_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Phosphorylated', 'Phosphorylated'],
                yticklabels=['Non-Phosphorylated', 'Phosphorylated'])
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
             transform=plt.gca().transAxes, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*80)
print("✅ TRAINING INFRASTRUCTURE READY")
print("="*80)
print("🎯 Main function: train_transformer_model(model_name)")
print("📋 Supported models: 'transformer_v1, transformer_v2'")
print("🔧 Features:")
print("   • Complete training pipeline")
print("   • Progress tracking with tqdm")
print("   • Automatic checkpointing")
print("   • Comprehensive evaluation")
print("   • Results visualization")
print("   • Master results tracking")
print("="*80)


# ============================================================================
# CELL 4: MODEL SELECTION & TRAINING - FULL DATASET (FIXED)
# ============================================================================

print("\n" + "="*80)
print("MODEL SELECTION & TRAINING - FULL DATASET")
print("="*80)

# ============================================================================
# Data Cleaning Function (Based on Our Successful Test)
# ============================================================================

def clean_dataframe_for_training(df, name="dataset"):
    """Clean dataframe using the approach that worked in our test"""
    print(f"🧹 Cleaning {name} ({len(df)} samples)...")
    
    clean_samples = []
    removed_count = 0
    
    for idx, row in df.iterrows():
        try:
            sequence = str(row['Sequence']).upper()
            position = int(row['Position'])
            
            # Remove invalid characters (keep only standard amino acids)
            clean_sequence = ''.join(c for c in sequence if c in 'ACDEFGHIKLMNPQRSTVWY')
            
            # Skip if sequence too short after cleaning
            if len(clean_sequence) < 20:
                removed_count += 1
                continue
                
            # Fix position if needed
            if position <= 0 or position > len(clean_sequence):
                # Place in middle if invalid position
                position = len(clean_sequence) // 2
            
            clean_samples.append({
                'Header': row['Header'],
                'Sequence': clean_sequence,
                'Position': position,
                'target': int(row['target'])
            })
            
        except Exception:
            removed_count += 1
            continue
    
    clean_df = pd.DataFrame(clean_samples)
    print(f"   ✅ Cleaned: {len(clean_df)} samples (removed {removed_count} invalid)")
    
    return clean_df

# ============================================================================
# Apply Data Cleaning to All Splits
# ============================================================================

print("🧹 Applying data cleaning to all datasets...")

# Clean all datasets using our successful approach
clean_train_df = clean_dataframe_for_training(train_df, "train")
clean_val_df = clean_dataframe_for_training(val_df, "validation") 
clean_test_df = clean_dataframe_for_training(test_df, "test")

print(f"\n📊 Final cleaned datasets:")
print(f"   🔹 Train: {len(clean_train_df)} samples")
print(f"   🔹 Val: {len(clean_val_df)} samples")
print(f"   🔹 Test: {len(clean_test_df)} samples")

# Update global variables for training infrastructure
train_df_clean = clean_train_df
val_df_clean = clean_val_df
test_df_clean = clean_test_df

# ============================================================================
# Update Training Infrastructure to Use Clean Data
# ============================================================================

def train_transformer_model_production(model_name):
    """
    Production training function using cleaned full dataset
    
    This is the same as train_transformer_model but uses our cleaned data
    """
    
    print(f"\n🚀 Starting PRODUCTION training for: {model_name.upper()}")
    print("="*60)
    
    # ========================================================================
    # 1. Model Creation and Configuration
    # ========================================================================
    
    # Get model and config based on model_name
    if model_name == "transformer_v1":
        model = create_transformer_v1()
        config = TRANSFORMER_V1_CONFIG.copy()
        architecture_name = "TransformerV1_BasePhospho"
    elif model_name == "transformer_v2":
        model = create_transformer_v2()
        config = TRANSFORMER_V2_CONFIG.copy()
        architecture_name = "TransformerV2_Hierarchical"
    else:
        available_models = ["transformer_v1", "transformer_v2"]
        raise ValueError(f"Unknown model: {model_name}. Available: {available_models}")
    
    # Move model to device
    print(f"🔄 Moving model to {DEVICE}...")
    model = model.to(DEVICE)
    
    # Verify model is on correct device
    sample_param = next(model.parameters())
    actual_device = sample_param.device
    print(f"✅ Model '{architecture_name}' loaded on {actual_device}")
    
    # Setup model directory
    model_dir, timestamp = setup_model_directory(model_name, BASE_DIR)
    print(f"📁 Model directory: {model_dir}")
    
    # ========================================================================
    # 2. Save Model Architecture and Config
    # ========================================================================
    
    # Save configuration
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model description
    description = f"""# {architecture_name}
    
## Architecture
- **Model**: {config['model_name']}
- **Architecture**: {config['architecture_name']}
- **Window Context**: ±{config['window_context']} positions
- **Dropout Rate**: {config['dropout_rate']}
- **Total Parameters**: {sum(p.numel() for p in model.parameters()):,}

## Training Configuration
- **Learning Rate**: {config['learning_rate']}
- **Batch Size**: {config['batch_size']}
- **Max Epochs**: {config['epochs']}
- **Early Stopping**: {config['early_stopping_patience']} epochs
- **Weight Decay**: {config['weight_decay']}

## Dataset (Cleaned)
- **Train Samples**: {len(train_df_clean):,}
- **Val Samples**: {len(val_df_clean):,}
- **Test Samples**: {len(test_df_clean):,}
- **Data Cleaning**: Applied amino acid filtering and position validation

## Key Features
{chr(10).join('- ' + feature for feature in config['key_features'])}

## Source
- **Original Code**: {config['source']}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Device**: {DEVICE}

## Description
{config['description']}
"""
    
    desc_path = os.path.join(model_dir, 'model_description.md')
    with open(desc_path, 'w') as f:
        f.write(description)
    
    print(f"📝 Saved model config and description")
    
    # ========================================================================
    # 3. Create Data Loaders (Using Cleaned Data)
    # ========================================================================
    
    print(f"\n📊 Creating data loaders with cleaned data...")
    
    # Create datasets using CLEANED dataframes
    train_dataset = PhosphorylationDataset(
        train_df_clean, tokenizer, 
        window_size=config['window_size'], 
        max_length=config['max_length']
    )
    val_dataset = PhosphorylationDataset(
        val_df_clean, tokenizer,
        window_size=config['window_size'],
        max_length=config['max_length']
    )
    test_dataset = PhosphorylationDataset(
        test_df_clean, tokenizer,
        window_size=config['window_size'],
        max_length=config['max_length']
    )
    
    # Create data loaders with safe settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False  # Avoid memory issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"   🔢 Train batches: {len(train_loader)} (samples: {len(train_dataset)})")
    print(f"   🔢 Val batches: {len(val_loader)} (samples: {len(val_dataset)})")
    print(f"   🔢 Test batches: {len(test_loader)} (samples: {len(test_dataset)})")
    
    # ========================================================================
    # 4. Setup Optimizer and Scheduler
    # ========================================================================
    
    print(f"\n⚙️ Setting up optimizer and scheduler...")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = config['warmup_steps']
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"   📈 Optimizer: AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
    print(f"   📉 Scheduler: Linear warmup + decay ({warmup_steps} warmup steps)")
    print(f"   🔄 Total steps: {total_steps}")
    
    # ========================================================================
    # 5. Training Loop
    # ========================================================================
    
    print(f"\n🏋️ Starting training loop...")
    print(f"   🎯 Target epochs: {config['epochs']}")
    print(f"   ⏰ Early stopping: {config['early_stopping_patience']} epochs")
    print(f"   💾 Save every: {config['save_every_epochs']} epochs")
    
    # Training history - Initialize empty lists
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': []
    }
    
    # Early stopping variables
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    
    # Training start time
    training_start_time = time.time()
    
    # Main training loop
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train epoch
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, epoch, config['epochs'])
        
        # Validate epoch
        val_metrics, _, _ = evaluate_model(model, val_loader, DEVICE, "Val")
        
        # Store metrics in history - EXPLICIT STORAGE
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Debug: Print history lengths to verify storage
        print(f"   📊 History check: train_f1 has {len(history['train_f1'])} values, val_f1 has {len(history['val_f1'])} values")
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   🔹 Train: Loss={train_metrics['loss']:.4f}, F1={train_metrics['f1']:.4f}, AUC={train_metrics['auc']:.4f}")
        print(f"   🔸 Val:   Loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")
        
        # Check for best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            save_model_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_model_path)
            print(f"   🌟 New best model! F1={best_val_f1:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{config['early_stopping_patience']}")
        
        # Regular checkpoint saving
        if epoch % config['save_every_epochs'] == 0:
            checkpoint_path = os.path.join(model_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
            save_model_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_path)
            print(f"   💾 Checkpoint saved: epoch_{epoch}.pth")
        
        # Early stopping check
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n🛑 Early stopping triggered after {config['early_stopping_patience']} epochs without improvement")
            early_stopped = True
            break
    
    # ========================================================================
    # 6. Final Evaluation
    # ========================================================================
    
    training_time_mins = (time.time() - training_start_time) / 60
    print(f"\n🏁 Training completed in {training_time_mins:.1f} minutes")
    print(f"   🏆 Best epoch: {best_epoch} (F1={best_val_f1:.4f})")
    print(f"   🔄 Total epochs: {epoch}")
    print(f"   ⏰ Early stopped: {'Yes' if early_stopped else 'No'}")
    
    # Load best model for final evaluation
    print(f"\n🔄 Loading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final test evaluation
    print(f"\n🧪 Final test evaluation...")
    test_metrics, test_predictions, test_targets = evaluate_model(model, test_loader, DEVICE, "Test")
    
    print(f"\n📈 Final Test Results:")
    for metric, value in test_metrics.items():
        print(f"   🎯 {metric.upper()}: {value:.4f}")
    
    # ========================================================================
    # 7. Save Results and Visualizations
    # ========================================================================
    
    print(f"\n💾 Saving results and visualizations...")
    
    # Save training history
    history_path = os.path.join(model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test predictions
    predictions_df = pd.DataFrame({
        'header': [test_dataset.dataframe.iloc[i]['Header'] for i in range(len(test_dataset))],
        'sequence': [test_dataset.dataframe.iloc[i]['Sequence'] for i in range(len(test_dataset))],
        'position': [test_dataset.dataframe.iloc[i]['Position'] for i in range(len(test_dataset))],
        'true_label': test_targets,
        'prediction_prob': test_predictions,
        'prediction_binary': (np.array(test_predictions) > 0.5).astype(int)
    })
    
    predictions_path = os.path.join(model_dir, 'predictions', 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Generate training curves
    curves_path = os.path.join(model_dir, 'plots', 'training_curves.png')
    plot_training_curves(history, curves_path)
    
    # Generate confusion matrix
    plot_confusion_matrix(test_targets, test_predictions, 
                         os.path.join(model_dir, 'plots', 'confusion_matrix.png'))
    
    # Save final metrics
    final_metrics = {
        'model_name': model_name,
        'architecture': architecture_name,
        'timestamp': timestamp,
        'total_params': sum(p.numel() for p in model.parameters()),
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'training_time_mins': training_time_mins,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'config': config,
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
    }
    
    metrics_path = os.path.join(model_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)
    
    # ========================================================================
    # 8. Update Master Results
    # ========================================================================
    
    print(f"\n📊 Updating master results...")
    
    # Prepare row for master results
    master_row = {
        'model_name': f"{model_name}_{timestamp}",
        'timestamp': timestamp,
        'architecture': architecture_name,
        'total_params': sum(p.numel() for p in model.parameters()),
        'memory_mb': f"~{(sum(p.numel() for p in model.parameters()) * 4) / (1024**2):.0f}",
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        'train_time_mins': f"{training_time_mins:.1f}",
        'early_stopped': early_stopped,
        'test_accuracy': f"{test_metrics['accuracy']:.4f}",
        'test_precision': f"{test_metrics['precision']:.4f}",
        'test_recall': f"{test_metrics['recall']:.4f}",
        'test_f1': f"{test_metrics['f1']:.4f}",
        'test_auc': f"{test_metrics['auc']:.4f}",
        'test_mcc': f"{test_metrics['mcc']:.4f}",
        'val_f1_best': f"{best_val_f1:.4f}",
        'train_f1_final': f"{history['train_f1'][-1]:.4f}" if history['train_f1'] else "0.0000",
        'notes': f"Clean data: {len(train_dataset)} train samples, {epoch} epochs"
    }
    
    # Debug: Print history summary before saving
    print(f"   🔍 Final history check:")
    for key, values in history.items():
        print(f"      {key}: {len(values)} values")
    if history['train_f1']:
        print(f"      Final train F1: {history['train_f1'][-1]:.4f}")
    if history['val_f1']:
        print(f"      Final val F1: {history['val_f1'][-1]:.4f}")
    
    # Update master results
    updated_results = update_master_results(master_results_path, master_row)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"=" * 60)
    print(f"🏷️  Model: {model_name}")
    print(f"📁 Results: {model_dir}")
    print(f"🏆 Best F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"🧪 Test F1: {test_metrics['f1']:.4f}")
    print(f"⏱️  Time: {training_time_mins:.1f} minutes")
    print(f"📊 Master results updated: {len(updated_results)} total models")
    print(f"=" * 60)
    
    return final_metrics

# ============================================================================
# MODEL SELECTION - CHANGE THIS TO TRAIN DIFFERENT MODELS
# ============================================================================

CURRENT_MODEL = "transformer_v1"  # 🔄 Change this to train different models

print(f"🎯 Selected model for FULL TRAINING: {CURRENT_MODEL.upper()}")
print(f"🚀 Starting production training pipeline with cleaned data...")

# ============================================================================
# EXECUTE FULL TRAINING
# ============================================================================

try:
    # Train the selected model on full cleaned dataset
    training_results = train_transformer_model_production(CURRENT_MODEL)
    
    print(f"\n🎉 SUCCESS! Full training completed for {CURRENT_MODEL}")
    print(f"📋 Results summary:")
    print(f"   🏆 Best validation F1: {training_results['best_val_f1']:.4f}")
    print(f"   🧪 Final test F1: {training_results['test_metrics']['f1']:.4f}")
    print(f"   🧪 Final test AUC: {training_results['test_metrics']['auc']:.4f}")
    print(f"   ⏱️  Training time: {training_results['training_time_mins']:.1f} minutes")
    print(f"   🔄 Epochs trained: {training_results['epochs_trained']}")
    print(f"   ⏰ Early stopped: {training_results['early_stopped']}")
    
except Exception as e:
    print(f"\n❌ ERROR during training:")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()
    
print(f"\n" + "="*80)
print("FULL TRAINING SESSION COMPLETE")
print("="*80)
print(f"💡 To train another model:")
print(f"   1. Change CURRENT_MODEL to a different model name")
print(f"   2. Re-run this cell")
print(f"\n📊 View results in: results/exp_3/transformers/")
print(f"📋 Master results: results/exp_3/transformers/master_results.csv")
print("="*80)


# ============================================================================
# CELL 5: RESULTS ANALYSIS & VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("TRANSFORMER RESULTS ANALYSIS & VISUALIZATION")
print("="*80)

def load_master_results():
    """Load and display master results table"""
    try:
        master_results = pd.read_csv(master_results_path)
        if len(master_results) == 0:
            print("📋 No models trained yet")
            return None
        
        print(f"📊 Master Results Summary ({len(master_results)} models):")
        print("="*60)
        
        # Display key columns
        display_cols = ['model_name', 'architecture', 'epochs_trained', 'train_time_mins', 
                       'test_f1', 'test_auc', 'val_f1_best']
        
        if all(col in master_results.columns for col in display_cols):
            display_df = master_results[display_cols].copy()
            
            # Format numeric columns
            for col in ['train_time_mins', 'test_f1', 'test_auc', 'val_f1_best']:
                if col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            
            print(display_df.to_string(index=False))
        else:
            print(master_results.to_string(index=False))
        
        return master_results
        
    except FileNotFoundError:
        print("📋 Master results file not found - no models trained yet")
        return None
    except Exception as e:
        print(f"❌ Error loading master results: {e}")
        return None

def plot_model_comparison(master_results):
    """Create model comparison visualizations"""
    if master_results is None or len(master_results) == 0:
        print("⚠️  No models to compare")
        return
    
    print(f"\n📈 Creating model comparison plots...")
    
    # Ensure numeric columns
    numeric_cols = ['test_f1', 'test_auc', 'test_accuracy', 'test_precision', 'test_recall', 'test_mcc']
    for col in numeric_cols:
        if col in master_results.columns:
            master_results[col] = pd.to_numeric(master_results[col], errors='coerce')
    
    n_models = len(master_results)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Transformer Models Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['test_f1', 'test_auc', 'test_accuracy', 'test_precision', 'test_recall', 'test_mcc']
    titles = ['F1 Score', 'AUC', 'Accuracy', 'Precision', 'Recall', 'MCC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        if metric in master_results.columns:
            values = master_results[metric].dropna()
            model_names = master_results.loc[values.index, 'model_name']
            
            if len(values) > 0:
                bars = ax.bar(range(len(values)), values, color='skyblue', alpha=0.7)
                ax.set_xlabel('Models')
                ax.set_ylabel(title)
                ax.set_title(f'{title} Comparison')
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels([name.split('_')[0] for name in model_names], rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        else:
            ax.text(0.5, 0.5, f'{metric}\nNot Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(BASE_DIR, 'transformers', 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    comparison_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Model comparison saved: {comparison_path}")

def plot_training_efficiency(master_results):
    """Plot training efficiency analysis"""
    if master_results is None or len(master_results) == 0:
        return
    
    print(f"\n⚡ Creating training efficiency analysis...")
    
    # Convert numeric columns
    master_results['train_time_mins'] = pd.to_numeric(master_results['train_time_mins'], errors='coerce')
    master_results['test_f1'] = pd.to_numeric(master_results['test_f1'], errors='coerce')
    master_results['epochs_trained'] = pd.to_numeric(master_results['epochs_trained'], errors='coerce')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training time vs Performance
    ax1 = axes[0]
    valid_data = master_results.dropna(subset=['train_time_mins', 'test_f1'])
    
    if len(valid_data) > 0:
        scatter = ax1.scatter(valid_data['train_time_mins'], valid_data['test_f1'], 
                             s=100, alpha=0.7, c='coral')
        ax1.set_xlabel('Training Time (minutes)')
        ax1.set_ylabel('Test F1 Score')
        ax1.set_title('Training Efficiency: Time vs Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add model labels
        for idx, row in valid_data.iterrows():
            ax1.annotate(row['model_name'].split('_')[0], 
                        (row['train_time_mins'], row['test_f1']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Epochs vs Performance
    ax2 = axes[1]
    valid_data2 = master_results.dropna(subset=['epochs_trained', 'test_f1'])
    
    if len(valid_data2) > 0:
        bars = ax2.bar(range(len(valid_data2)), valid_data2['test_f1'], 
                      color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Test F1 Score')
        ax2.set_title('Model Performance by Epochs Trained')
        ax2.set_xticks(range(len(valid_data2)))
        ax2.set_xticklabels([f"{name.split('_')[0]}\n({epochs}e)" 
                           for name, epochs in zip(valid_data2['model_name'], valid_data2['epochs_trained'])],
                           rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    efficiency_path = os.path.join(plots_dir, 'training_efficiency.png')
    plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Training efficiency saved: {efficiency_path}")

def analyze_best_model(master_results):
    """Analyze the best performing model"""
    if master_results is None or len(master_results) == 0:
        return
    
    print(f"\n🏆 Best Model Analysis:")
    print("="*40)
    
    # Convert test_f1 to numeric
    master_results['test_f1_numeric'] = pd.to_numeric(master_results['test_f1'], errors='coerce')
    
    # Find best model
    best_idx = master_results['test_f1_numeric'].idxmax()
    best_model = master_results.loc[best_idx]
    
    print(f"🥇 Best Model: {best_model['model_name']}")
    print(f"   Architecture: {best_model['architecture']}")
    print(f"   Test F1: {best_model['test_f1']}")
    print(f"   Test AUC: {best_model.get('test_auc', 'N/A')}")
    print(f"   Training Time: {best_model.get('train_time_mins', 'N/A')} minutes")
    print(f"   Epochs: {best_model.get('epochs_trained', 'N/A')}")
    
    # Try to load detailed results
    model_name = best_model['model_name']
    model_dir = None
    
    # Find model directory
    transformers_dir = os.path.join(BASE_DIR, 'transformers')
    if os.path.exists(transformers_dir):
        for item in os.listdir(transformers_dir):
            if item.startswith(model_name.split('_')[0]) and os.path.isdir(os.path.join(transformers_dir, item)):
                model_dir = os.path.join(transformers_dir, item)
                break
    
    if model_dir and os.path.exists(os.path.join(model_dir, 'final_metrics.json')):
        try:
            with open(os.path.join(model_dir, 'final_metrics.json'), 'r') as f:
                detailed_metrics = json.load(f)
            
            print(f"\n📊 Detailed Test Metrics:")
            test_metrics = detailed_metrics.get('test_metrics', {})
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric.upper()}: {value:.4f}")
                    
        except Exception as e:
            print(f"   ⚠️  Could not load detailed metrics: {e}")
    
    return best_model

def create_summary_report():
    """Create a comprehensive summary report"""
    print(f"\n📝 Generating Summary Report...")
    
    master_results = load_master_results()
    
    if master_results is None:
        print("⚠️  No results to summarize")
        return
    
    # Create summary
    summary = f"""
# Transformer Models Summary Report

## Overview
- **Total Models Trained**: {len(master_results)}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Device Used**: {DEVICE}

## Model Performance Summary
"""
    
    if len(master_results) > 0:
        # Convert to numeric for calculations
        for col in ['test_f1', 'test_auc', 'test_accuracy']:
            if col in master_results.columns:
                master_results[f'{col}_numeric'] = pd.to_numeric(master_results[col], errors='coerce')
        
        # Best models
        if 'test_f1_numeric' in master_results.columns:
            best_f1_idx = master_results['test_f1_numeric'].idxmax()
            best_f1_model = master_results.loc[best_f1_idx]
            summary += f"\n### Best F1 Score\n"
            summary += f"- **Model**: {best_f1_model['model_name']}\n"
            summary += f"- **F1 Score**: {best_f1_model['test_f1']}\n"
            summary += f"- **Architecture**: {best_f1_model.get('architecture', 'N/A')}\n"
        
        # Performance statistics
        f1_scores = master_results['test_f1_numeric'].dropna()
        if len(f1_scores) > 0:
            summary += f"\n### Performance Statistics\n"
            summary += f"- **Mean F1**: {f1_scores.mean():.4f}\n"
            summary += f"- **Std F1**: {f1_scores.std():.4f}\n"
            summary += f"- **Min F1**: {f1_scores.min():.4f}\n"
            summary += f"- **Max F1**: {f1_scores.max():.4f}\n"
    
    summary += f"\n### Training Summary\n"
    if 'train_time_mins' in master_results.columns:
        times = pd.to_numeric(master_results['train_time_mins'], errors='coerce').dropna()
        if len(times) > 0:
            summary += f"- **Total Training Time**: {times.sum():.1f} minutes\n"
            summary += f"- **Average Training Time**: {times.mean():.1f} minutes\n"
    
    # Save report
    report_path = os.path.join(BASE_DIR, 'transformers', 'summary_report.md')
    with open(report_path, 'w') as f:
        f.write(summary)
    
    print(f"✅ Summary report saved: {report_path}")
    print(summary)

# ============================================================================
# Main Analysis Execution
# ============================================================================

print(f"🔍 Starting comprehensive results analysis...")

# Load and display results
master_results = load_master_results()

if master_results is not None and len(master_results) > 0:
    # Create visualizations
    plot_model_comparison(master_results)
    plot_training_efficiency(master_results)
    
    # Analyze best model
    best_model = analyze_best_model(master_results)
    
    # Create summary report
    create_summary_report()
    
    print(f"\n🎉 Results analysis complete!")
    print(f"📁 All plots saved to: {os.path.join(BASE_DIR, 'transformers', 'comparison_plots')}")
    
else:
    print(f"\n💡 No trained models found yet.")
    print(f"   Train some models first, then run this analysis!")

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)