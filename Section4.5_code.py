# ============================================================================
# SECTION 4.5: TABNET MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*80)
print("SECTION 4.5: TABNET MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("="*80)
print("🧠 Hybrid approach between traditional ML and transformers")
print("📊 Using combined feature set with attention-based feature selection")
print("🔍 Grid search for optimal hyperparameters to combat overfitting")
print("="*80)

# ============================================================================
# Import Required Libraries
# ============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import gc
import time
from datetime import datetime
import json
from itertools import product

# TabNet specific imports
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.metrics import Metric
    TABNET_AVAILABLE = True
    print("✅ pytorch-tabnet imported successfully")
except ImportError:
    TABNET_AVAILABLE = False
    print("❌ pytorch-tabnet not available. Install with: pip install pytorch-tabnet")
    raise ImportError("pytorch-tabnet is required for Section 4.5")

# ============================================================================
# Hyperparameter Grid Search Configuration
# ============================================================================

# Base configuration (fixed parameters)
BASE_CONFIG = {
    # Fixed architecture parameters (less critical for overfitting)
    'n_independent': 2,           # Keep standard
    'n_shared': 2,                # Keep standard
    'momentum': 0.02,             # TabNet standard
    'mask_type': 'entmax',        # Better than sparsemax
    
    # Fixed training parameters
    'max_epochs': 150,            # Reduced from 200 for faster tuning
    'patience': 15,               # Reduced for faster convergence
    'virtual_batch_size': 128,    # Fixed for consistency
    'drop_last': False,
    
    # Scheduler parameters
    'scheduler_fn': torch.optim.lr_scheduler.StepLR,
    'scheduler_params': {'step_size': 40, 'gamma': 0.8},
    
    # System parameters
    'seed': RANDOM_SEED,
    'device_name': 'auto',
    'verbose': 1
}

# Grid search parameters (most critical for performance)
GRID_SEARCH_PARAMS = {
    'lambda_sparse': [1e-2, 5e-2, 1e-1],     # Sparsity regularization (most critical)
    'n_steps': [2, 3, 4],                     # Model complexity (second most critical)
    'lr': [0.005, 0.01],                      # Learning rate (third most critical)
    'batch_size': [256, 512],                 # Batch size affects generalization
    'n_d_n_a': [16, 32]                       # Decision/attention width (combined)
}

# Generate all combinations and select best 8
all_combinations = list(product(
    GRID_SEARCH_PARAMS['lambda_sparse'],
    GRID_SEARCH_PARAMS['n_steps'], 
    GRID_SEARCH_PARAMS['lr'],
    GRID_SEARCH_PARAMS['batch_size'],
    GRID_SEARCH_PARAMS['n_d_n_a']
))

print(f"Total possible combinations: {len(all_combinations)}")

# Hand-picked best 8 combinations based on overfitting analysis
BEST_CONFIGS = [
    # Config 1: Strong regularization + small model
    {'lambda_sparse': 1e-1, 'n_steps': 2, 'lr': 0.005, 'batch_size': 256, 'n_d': 16, 'n_a': 16},
    
    # Config 2: Very strong regularization + moderate model
    {'lambda_sparse': 5e-2, 'n_steps': 3, 'lr': 0.01, 'batch_size': 256, 'n_d': 16, 'n_a': 16},
    
    # Config 3: Strong regularization + moderate model + larger batch
    {'lambda_sparse': 5e-2, 'n_steps': 3, 'lr': 0.005, 'batch_size': 512, 'n_d': 16, 'n_a': 16},
    
    # Config 4: Moderate regularization + small model + slow learning
    {'lambda_sparse': 1e-2, 'n_steps': 2, 'lr': 0.005, 'batch_size': 512, 'n_d': 16, 'n_a': 16},
    
    # Config 5: Very strong regularization + small model + larger capacity
    {'lambda_sparse': 1e-1, 'n_steps': 2, 'lr': 0.005, 'batch_size': 256, 'n_d': 32, 'n_a': 32},
    
    # Config 6: Strong regularization + larger model + slow learning
    {'lambda_sparse': 5e-2, 'n_steps': 4, 'lr': 0.005, 'batch_size': 256, 'n_d': 16, 'n_a': 16},
    
    # Config 7: Moderate regularization + moderate model + fast learning
    {'lambda_sparse': 1e-2, 'n_steps': 3, 'lr': 0.01, 'batch_size': 512, 'n_d': 32, 'n_a': 32},
    
    # Config 8: Balanced approach
    {'lambda_sparse': 5e-2, 'n_steps': 3, 'lr': 0.005, 'batch_size': 512, 'n_d': 32, 'n_a': 32}
]

# Add gamma parameter to each config (feature reusability)
for i, config in enumerate(BEST_CONFIGS):
    if config['lambda_sparse'] >= 5e-2:
        config['gamma'] = 1.0  # Less reuse with high sparsity
    else:
        config['gamma'] = 1.3  # More reuse with lower sparsity

print(f"✅ Selected {len(BEST_CONFIGS)} best configurations for grid search")
print("🎯 Optimization targets: F1 Score (primary) + Accuracy (secondary)")

# ============================================================================
# Load Required Data from Previous Sections
# ============================================================================

print("\n4.5.1 Loading Required Data from Previous Checkpoints")
print("-" * 60)

# Check memory before loading
initial_memory = progress_tracker.get_memory_usage()
print(f"Initial memory usage: {initial_memory['rss_mb']:.1f} MB")

# Load data from checkpoints (same approach as Section 4)
required_vars = ['y_train', 'y_val', 'y_test', 'train_indices', 'val_indices', 'test_indices', 'cv_folds']
missing_vars = [var for var in required_vars if var not in locals()]

if missing_vars or 'aac_features' not in locals():
    print("Loading required data from previous checkpoints...")
    
    # Load from Section 3 (data splitting) - exact same approach as Section 4
    checkpoint_data = progress_tracker.resume_from_checkpoint("data_splitting")
    if checkpoint_data:
        train_indices = checkpoint_data['train_indices']
        val_indices = checkpoint_data['val_indices'] 
        test_indices = checkpoint_data['test_indices']
        cv_folds = checkpoint_data['cv_folds']
        print("✅ Loaded split indices from Section 3")
    else:
        raise RuntimeError("Section 3 checkpoint not found. Please run data splitting first.")
    
    # Load from Section 2 (feature extraction) - get all feature matrices
    checkpoint_data = progress_tracker.resume_from_checkpoint("feature_extraction")
    if checkpoint_data:
        feature_matrices = checkpoint_data['feature_matrices']
        combined_features = feature_matrices['combined']
        aac_features = feature_matrices['aac']
        dpc_features = feature_matrices['dpc']
        tpc_features = feature_matrices['tpc']
        binary_features = feature_matrices['binary']
        physicochemical_features = feature_matrices['physicochemical']
        print("✅ Loaded all feature matrices from Section 2")
    else:
        raise RuntimeError("Section 2 checkpoint not found. Please run feature extraction first.")
    
    # Load from Section 1 (data loading) - get df_final to create targets
    checkpoint_data = progress_tracker.resume_from_checkpoint("data_loading")
    if checkpoint_data:
        df_final = checkpoint_data['df_final']
        print("✅ Loaded df_final from Section 1")
    else:
        raise RuntimeError("Section 1 checkpoint not found. Please run data loading first.")
    
    # Create target arrays from df_final and indices (same as Section 4)
    y_train = df_final.iloc[train_indices]['target'].values
    y_val = df_final.iloc[val_indices]['target'].values  
    y_test = df_final.iloc[test_indices]['target'].values
    print("✅ Created target arrays from df_final")

# Verify data loaded correctly
print(f"✅ Data verification:")
print(f"   📊 Features shape: {combined_features.shape}")
print(f"   🔢 Train samples: {len(train_indices)}")
print(f"   🔢 Val samples: {len(val_indices)}")
print(f"   🔢 Test samples: {len(test_indices)}")
print(f"   ⚖️  Class balance - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")

# ============================================================================
# Check if Section Already Completed
# ============================================================================

if progress_tracker.is_completed("tabnet_model") and not FORCE_RETRAIN:
    print("\n⚡ TabNet model already trained. Loading from checkpoint...")
    checkpoint_data = progress_tracker.resume_from_checkpoint("tabnet_model")
    if checkpoint_data:
        tabnet_results = checkpoint_data['tabnet_results']
        best_config = checkpoint_data['best_config']
        grid_search_results = checkpoint_data['grid_search_results']
        tabnet_interpretability = checkpoint_data['interpretability_data']
        
        print("✅ TabNet results loaded from checkpoint!")
        print(f"   🏆 Best Test F1: {tabnet_results['test_metrics']['f1']:.4f}")
        print(f"   🎯 Best Test AUC: {tabnet_results['test_metrics']['auc']:.4f}")
        print(f"   ⚙️  Best Config: {best_config['name']}")
        
        # Make results available for ensemble methods
        if 'tabnet_results' in locals():
            print("✅ TabNet model ready for ensemble integration")
    else:
        print("❌ Failed to load TabNet checkpoint")
else:
    # ============================================================================
    # Data Preparation for TabNet
    # ============================================================================
    
    print("\n4.5.2 Data Preparation for TabNet")
    print("-" * 60)
    
    # Prepare feature matrices for TabNet
    print("Preparing feature matrices...")
    
    # Get feature data for each split
    X_train = combined_features.iloc[train_indices].values.astype(np.float32)
    X_val = combined_features.iloc[val_indices].values.astype(np.float32)
    X_test = combined_features.iloc[test_indices].values.astype(np.float32)
    
    # Convert targets to numpy arrays
    y_train_np = y_train.astype(np.int64)
    y_val_np = y_val.astype(np.int64)
    y_test_np = y_test.astype(np.int64)
    
    print(f"✅ Data preparation completed:")
    print(f"   📈 X_train shape: {X_train.shape}, y_train shape: {y_train_np.shape}")
    print(f"   📊 X_val shape: {X_val.shape}, y_val shape: {y_val_np.shape}")
    print(f"   🧪 X_test shape: {X_test.shape}, y_test shape: {y_test_np.shape}")
    
    # Feature scaling (TabNet generally doesn't require it, but can help)
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Feature scaling completed")
    
    # Check for any data quality issues
    print("Checking data quality...")
    train_nan = np.isnan(X_train_scaled).sum()
    train_inf = np.isinf(X_train_scaled).sum()
    
    if train_nan > 0 or train_inf > 0:
        print(f"⚠️  Found {train_nan} NaN and {train_inf} infinite values - cleaning...")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("✅ Data quality check completed")
    
    # ============================================================================
    # Grid Search - TabNet Hyperparameter Optimization
    # ============================================================================
    
    print("\n4.5.3 TabNet Hyperparameter Grid Search")
    print("-" * 60)
    
    # Track overall grid search time
    grid_search_start_time = time.time()
    
    # Results storage
    grid_search_results = []
    best_f1_score = 0.0
    best_config = None
    best_model = None
    
    print(f"🔍 Starting grid search with {len(BEST_CONFIGS)} configurations...")
    print(f"🎯 Targeting ~4 hours total ({240/len(BEST_CONFIGS):.1f} minutes per config)")
    
    for config_idx, config in enumerate(BEST_CONFIGS):
        print(f"\n{'='*60}")
        print(f"CONFIG {config_idx + 1}/{len(BEST_CONFIGS)}")
        print(f"{'='*60}")
        
        # Create full configuration
        full_config = BASE_CONFIG.copy()
        full_config.update(config)
        
        # Display current configuration
        print(f"🔧 Configuration:")
        print(f"   lambda_sparse: {config['lambda_sparse']}")
        print(f"   n_steps: {config['n_steps']}")
        print(f"   lr: {config['lr']}")
        print(f"   batch_size: {config['batch_size']}")
        print(f"   n_d/n_a: {config['n_d']}/{config['n_a']}")
        print(f"   gamma: {config['gamma']}")
        
        # Set random seed for reproducibility
        torch.manual_seed(RANDOM_SEED + config_idx)
        np.random.seed(RANDOM_SEED + config_idx)
        
        # Track configuration training time
        config_start_time = time.time()
        
        try:
            # Initialize TabNet model with current configuration
            tabnet_model = TabNetClassifier(
                n_d=config['n_d'],
                n_a=config['n_a'],
                n_steps=config['n_steps'],
                gamma=config['gamma'],
                n_independent=full_config['n_independent'],
                n_shared=full_config['n_shared'],
                lambda_sparse=config['lambda_sparse'],
                momentum=full_config['momentum'],
                mask_type=full_config['mask_type'],
                scheduler_fn=full_config['scheduler_fn'],
                scheduler_params=full_config['scheduler_params'],
                verbose=full_config['verbose'],
                device_name=full_config['device_name'],
                seed=full_config['seed']
            )
            
            print(f"🚀 Training TabNet with config {config_idx + 1}...")
            
            # Train the model
            tabnet_model.fit(
                X_train=X_train_scaled,
                y_train=y_train_np,
                eval_set=[(X_val_scaled, y_val_np)],
                eval_name=['validation'],
                eval_metric=['auc', 'accuracy'],
                max_epochs=full_config['max_epochs'],
                patience=full_config['patience'],
                batch_size=config['batch_size'],
                virtual_batch_size=full_config['virtual_batch_size'],
                drop_last=full_config['drop_last']
            )
            
            config_end_time = time.time()
            config_training_time = config_end_time - config_start_time
            
            # Generate predictions
            val_pred_proba = tabnet_model.predict_proba(X_val_scaled)[:, 1]
            val_pred = (val_pred_proba > 0.5).astype(int)
            
            test_pred_proba = tabnet_model.predict_proba(X_test_scaled)[:, 1]
            test_pred = (test_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            val_f1 = f1_score(y_val_np, val_pred)
            val_accuracy = accuracy_score(y_val_np, val_pred)
            val_auc = roc_auc_score(y_val_np, val_pred_proba)
            
            test_f1 = f1_score(y_test_np, test_pred)
            test_accuracy = accuracy_score(y_test_np, test_pred)
            test_auc = roc_auc_score(y_test_np, test_pred_proba)
            test_mcc = matthews_corrcoef(y_test_np, test_pred)
            
            # Composite score (F1 primary, Accuracy secondary)
            composite_score = 0.7 * val_f1 + 0.3 * val_accuracy
            
            # Store results
            config_results = {
                'config_idx': config_idx + 1,
                'config': config.copy(),
                'full_config': full_config.copy(),
                'training_time': config_training_time,
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'val_auc': val_auc,
                'test_f1': test_f1,
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'test_mcc': test_mcc,
                'composite_score': composite_score
            }
            
            grid_search_results.append(config_results)
            
            # Check if this is the best configuration
            if composite_score > best_f1_score:
                best_f1_score = composite_score
                best_config = config_results
                best_model = tabnet_model
                print(f"🏆 NEW BEST CONFIG! Composite Score: {composite_score:.4f}")
            
            # Print results
            print(f"📊 Results for Config {config_idx + 1}:")
            print(f"   Val F1: {val_f1:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
            print(f"   Test F1: {test_f1:.4f}, Test Acc: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
            print(f"   Composite Score: {composite_score:.4f}")
            print(f"   Training Time: {config_training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"❌ Error in config {config_idx + 1}: {str(e)}")
            # Try with reduced batch size if memory error
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"🔄 Retrying config {config_idx + 1} with reduced batch size...")
                try:
                    config['batch_size'] = min(config['batch_size'] // 2, 128)
                    full_config['virtual_batch_size'] = min(full_config['virtual_batch_size'] // 2, 64)
                    
                    tabnet_model = TabNetClassifier(
                        n_d=config['n_d'],
                        n_a=config['n_a'],
                        n_steps=config['n_steps'],
                        gamma=config['gamma'],
                        n_independent=full_config['n_independent'],
                        n_shared=full_config['n_shared'],
                        lambda_sparse=config['lambda_sparse'],
                        momentum=full_config['momentum'],
                        mask_type=full_config['mask_type'],
                        scheduler_fn=full_config['scheduler_fn'],
                        scheduler_params=full_config['scheduler_params'],
                        verbose=full_config['verbose'],
                        device_name=full_config['device_name'],
                        seed=full_config['seed']
                    )
                    
                    tabnet_model.fit(
                        X_train=X_train_scaled,
                        y_train=y_train_np,
                        eval_set=[(X_val_scaled, y_val_np)],
                        eval_name=['validation'],
                        eval_metric=['auc', 'accuracy'],
                        max_epochs=full_config['max_epochs'],
                        patience=full_config['patience'],
                        batch_size=config['batch_size'],
                        virtual_batch_size=full_config['virtual_batch_size'],
                        drop_last=full_config['drop_last']
                    )
                    
                    print(f"✅ Config {config_idx + 1} completed with reduced batch size")
                    
                except Exception as e2:
                    print(f"❌ Config {config_idx + 1} failed even with reduced batch size: {str(e2)}")
                    continue
            else:
                continue
        
        # Memory cleanup after each config
        if 'tabnet_model' in locals() and tabnet_model != best_model:
            del tabnet_model
        gc.collect()
        
        # Progress update
        elapsed_time = (time.time() - grid_search_start_time) / 60
        avg_time_per_config = elapsed_time / (config_idx + 1)
        estimated_total_time = avg_time_per_config * len(BEST_CONFIGS)
        remaining_time = estimated_total_time - elapsed_time
        
        print(f"⏱️  Progress: {config_idx + 1}/{len(BEST_CONFIGS)} configs completed")
        print(f"   Elapsed: {elapsed_time:.1f}min, Est. Remaining: {remaining_time:.1f}min")
        
    # Grid search completed
    grid_search_end_time = time.time()
    total_grid_search_time = grid_search_end_time - grid_search_start_time
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_grid_search_time/60:.1f} minutes")
    print(f"🏆 Best configuration: Config {best_config['config_idx']}")
    print(f"   Best Composite Score: {best_config['composite_score']:.4f}")
    print(f"   Best Test F1: {best_config['test_f1']:.4f}")
    print(f"   Best Test Accuracy: {best_config['test_accuracy']:.4f}")
    print(f"   Best Test AUC: {best_config['test_auc']:.4f}")
    
    # ============================================================================
    # Best Model Evaluation and Interpretability
    # ============================================================================
    
    print("\n4.5.4 Best Model Evaluation and Interpretability")
    print("-" * 60)
    
    # Generate final predictions with best model
    print("Generating final predictions with best model...")
    
    # Training predictions (for analysis)
    train_pred_proba = best_model.predict_proba(X_train_scaled)[:, 1]
    train_pred = (train_pred_proba > 0.5).astype(int)
    
    # Validation predictions
    val_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_pred_proba > 0.5).astype(int)
    
    # Test predictions
    test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_pred_proba > 0.5).astype(int)
    
    print("✅ Final predictions generated")
    
    # Calculate comprehensive metrics for best model
    def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, split_name):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        print(f"📊 {split_name} Metrics (Best Model):")
        for metric_name, value in metrics.items():
            print(f"   {metric_name.upper()}: {value:.4f}")
        
        return metrics
    
    # Evaluate best model on all splits
    train_metrics = calculate_comprehensive_metrics(y_train_np, train_pred, train_pred_proba, "Training")
    val_metrics = calculate_comprehensive_metrics(y_val_np, val_pred, val_pred_proba, "Validation") 
    test_metrics = calculate_comprehensive_metrics(y_test_np, test_pred, test_pred_proba, "Test")
    
    # ============================================================================
    # Interpretability Data Extraction
    # ============================================================================
    
    print("\n4.5.5 Extracting Interpretability Data")
    print("-" * 60)
    
    # Extract feature importances
    print("Extracting feature importances...")
    feature_importances = best_model.feature_importances_
    
    # Get feature names (assuming they're available from combined_features)
    if hasattr(combined_features, 'columns'):
        feature_names = combined_features.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(combined_features.shape[1])]
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    print(f"✅ Feature importances extracted for {len(feature_names)} features")
    print(f"   🏆 Top 5 features:")
    for i, (_, row) in enumerate(importance_df.head().iterrows()):
        print(f"      {i+1}. {row['feature_name']}: {row['importance']:.4f}")
    
    # Extract attention masks (if available)
    try:
        print("Extracting attention masks...")
        # Get masks for a sample of test data (first 100 samples)
        sample_size = min(100, len(X_test_scaled))
        explain_matrix, masks = best_model.explain(X_test_scaled[:sample_size])
        
        attention_data = {
            'explain_matrix': explain_matrix,
            'masks': masks,
            'sample_indices': list(range(sample_size))
        }
        print(f"✅ Attention masks extracted for {sample_size} samples")
        
    except Exception as e:
        print(f"⚠️  Could not extract attention masks: {str(e)}")
        attention_data = None
    
    # ============================================================================
    # Results Compilation and Saving
    # ============================================================================
    
    print("\n4.5.6 Compiling and Saving Results")
    print("-" * 60)
    
    # Compile comprehensive results
    tabnet_results = {
        'model_info': {
            'name': f'TabNet_GridSearch_Config{best_config["config_idx"]}',
            'type': 'hybrid_neural_network',
            'features': 'combined',
            'feature_count': combined_features.shape[1],
            'best_config_idx': best_config['config_idx'],
            'grid_search_time': total_grid_search_time,
            'total_configs_tested': len(grid_search_results),
            'timestamp': datetime.now().isoformat()
        },
        'training_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train_pred': train_pred,
            'train_pred_proba': train_pred_proba,
            'val_pred': val_pred,
            'val_pred_proba': val_pred_proba,
            'test_pred': test_pred,
            'test_pred_proba': test_pred_proba
        },
        'data_info': {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'feature_dims': X_train_scaled.shape[1]
        }
    }
    
    # Interpretability data (for future analysis) - convert DataFrame to dict
    interpretability_data = {
        'feature_importances': importance_df.to_dict('records'),
        'attention_data': attention_data,
        'feature_names': feature_names,
        'best_config_used': {
            'config_idx': best_config['config_idx'],
            'lambda_sparse': best_config['config']['lambda_sparse'],
            'n_steps': best_config['config']['n_steps'],
            'lr': best_config['config']['lr'],
            'batch_size': best_config['config']['batch_size'],
            'n_d': best_config['config']['n_d'],
            'n_a': best_config['config']['n_a'],
            'gamma': best_config['config']['gamma']
        },
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    # Save results for ensemble integration (same format as Section 4)
    model_predictions_for_ensemble = {
        'TabNet_combined': {
            'val_pred': val_pred_proba,
            'test_pred': test_pred_proba,
            'model_type': 'hybrid_neural_network',
            'features': 'combined',
            'metrics': test_metrics
        }
    }
    
    print("💾 Saving results...")
    
    # Create checkpoint data (clean version without nested objects)
    checkpoint_data = {
        'tabnet_results': tabnet_results,
        'best_config': {
            'config_idx': best_config['config_idx'],
            'config': best_config['config'],
            'composite_score': best_config['composite_score'],
            'test_f1': best_config['test_f1'],
            'test_accuracy': best_config['test_accuracy'],
            'test_auc': best_config['test_auc'],
            'training_time': best_config['training_time']
        },
        'grid_search_results': [{
            'config_idx': r['config_idx'],
            'config': r['config'],
            'composite_score': r['composite_score'],
            'test_f1': r['test_f1'],
            'test_accuracy': r['test_accuracy'],
            'test_auc': r['test_auc'],
            'training_time': r['training_time']
        } for r in grid_search_results],
        'interpretability_data': interpretability_data,
        'model_predictions': model_predictions_for_ensemble,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save checkpoint
    progress_tracker.mark_completed("tabnet_model", 
                                   metadata={'test_f1': test_metrics['f1'], 
                                            'test_auc': test_metrics['auc'],
                                            'best_config': best_config['config_idx'],
                                            'grid_search_time': total_grid_search_time},
                                   checkpoint_data=checkpoint_data)
    
    # Save additional files for detailed analysis
    results_dir = os.path.join(BASE_DIR, 'results', 'tabnet')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save feature importance
    importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
    
    # Save grid search results
    grid_df = pd.DataFrame(grid_search_results)
    grid_df.to_csv(os.path.join(results_dir, 'grid_search_results.csv'), index=False)
    
    # Save detailed results (JSON serializable version)
    def make_json_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-friendly format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj
    
    with open(os.path.join(results_dir, 'tabnet_results.json'), 'w') as f:
        # Convert entire results to JSON serializable format
        results_for_json = make_json_serializable(tabnet_results)
        json.dump(results_for_json, f, indent=2)
    
    # Save best configuration
    with open(os.path.join(results_dir, 'best_config.json'), 'w') as f:
        best_config_clean = {
            'config_idx': best_config['config_idx'],
            'config': best_config['config'],
            'composite_score': float(best_config['composite_score']),
            'test_f1': float(best_config['test_f1']),
            'test_accuracy': float(best_config['test_accuracy']),
            'test_auc': float(best_config['test_auc']),
            'training_time': float(best_config['training_time'])
        }
        json.dump(best_config_clean, f, indent=2)
    
    print(f"✅ Results saved to {results_dir}")
    
    # ============================================================================
    # Grid Search Analysis and Visualization
    # ============================================================================
    
    print("\n4.5.7 Grid Search Analysis")
    print("-" * 60)
    
    # Create analysis of grid search results
    if len(grid_search_results) > 1:
        print("📊 Grid Search Analysis:")
        
        # Sort by composite score
        sorted_results = sorted(grid_search_results, key=lambda x: x['composite_score'], reverse=True)
        
        print("🏆 Top 3 Configurations:")
        for i, result in enumerate(sorted_results[:3]):
            print(f"   {i+1}. Config {result['config_idx']}: Score {result['composite_score']:.4f}")
            print(f"      F1: {result['test_f1']:.4f}, Acc: {result['test_accuracy']:.4f}, AUC: {result['test_auc']:.4f}")
            print(f"      Params: λ={result['config']['lambda_sparse']}, steps={result['config']['n_steps']}, lr={result['config']['lr']}")
        
        # Parameter impact analysis
        print("\n📈 Parameter Impact Analysis:")
        
        # Lambda sparse impact
        lambda_values = list(set([r['config']['lambda_sparse'] for r in grid_search_results]))
        for lam in sorted(lambda_values):
            avg_score = np.mean([r['composite_score'] for r in grid_search_results if r['config']['lambda_sparse'] == lam])
            print(f"   λ_sparse {lam}: avg score {avg_score:.4f}")
        
        # N_steps impact
        steps_values = list(set([r['config']['n_steps'] for r in grid_search_results]))
        for steps in sorted(steps_values):
            avg_score = np.mean([r['composite_score'] for r in grid_search_results if r['config']['n_steps'] == steps])
            print(f"   n_steps {steps}: avg score {avg_score:.4f}")
        
        # Learning rate impact
        lr_values = list(set([r['config']['lr'] for r in grid_search_results]))
        for lr in sorted(lr_values):
            avg_score = np.mean([r['composite_score'] for r in grid_search_results if r['config']['lr'] == lr])
            print(f"   lr {lr}: avg score {avg_score:.4f}")
    
    # ============================================================================
    # Memory Cleanup
    # ============================================================================
    
    print("\n4.5.8 Memory Cleanup")
    print("-" * 60)
    
    # Clean up large variables
    del X_train, X_val, X_test
    del X_train_scaled, X_val_scaled, X_test_scaled
    del train_pred, val_pred, test_pred
    if 'attention_data' in locals() and attention_data is not None:
        del explain_matrix, masks
    if 'best_model' in locals():
        del best_model
    
    # Force garbage collection
    gc.collect()
    
    final_memory = progress_tracker.get_memory_usage()
    print(f"🧹 Memory cleanup completed")
    print(f"   📊 Memory usage: {initial_memory['rss_mb']:.1f} MB → {final_memory['rss_mb']:.1f} MB")

# ============================================================================
# Section Summary and Integration Check
# ============================================================================

print("\n" + "="*80)
print("SECTION 4.5 SUMMARY - TABNET HYPERPARAMETER TUNING")
print("="*80)

if 'tabnet_results' in locals() and 'best_config' in locals():
    print(f"🏷️  Model: TabNet (Hybrid Neural Network) - Grid Search Optimized")
    print(f"📊 Features: Combined feature set ({tabnet_results['model_info']['feature_count']} features)")
    print(f"🔍 Grid Search: {tabnet_results['model_info']['total_configs_tested']} configurations tested")
    print(f"⏱️  Total Time: {tabnet_results['model_info']['grid_search_time']/60:.1f} minutes")
    
    print(f"\n🏆 Best Configuration (Config {best_config['config_idx']}):")
    print(f"   • lambda_sparse: {best_config['config']['lambda_sparse']}")
    print(f"   • n_steps: {best_config['config']['n_steps']}")
    print(f"   • learning_rate: {best_config['config']['lr']}")
    print(f"   • batch_size: {best_config['config']['batch_size']}")
    print(f"   • n_d/n_a: {best_config['config']['n_d']}/{best_config['config']['n_a']}")
    
    print(f"\n🎯 Best Performance:")
    print(f"   • Test F1 Score: {tabnet_results['test_metrics']['f1']:.4f}")
    print(f"   • Test Accuracy: {tabnet_results['test_metrics']['accuracy']:.4f}")
    print(f"   • Test AUC: {tabnet_results['test_metrics']['auc']:.4f}")
    print(f"   • Test MCC: {tabnet_results['test_metrics']['mcc']:.4f}")
    print(f"   • Composite Score: {best_config['composite_score']:.4f}")
    
    # Check for overfitting
    train_f1 = tabnet_results['training_metrics']['f1']
    test_f1 = tabnet_results['test_metrics']['f1']
    overfitting_gap = train_f1 - test_f1
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   • Train F1: {train_f1:.4f}")
    print(f"   • Test F1: {test_f1:.4f}")
    print(f"   • Gap: {overfitting_gap:.4f} {'✅ (Good)' if overfitting_gap < 0.1 else '⚠️ (High)' if overfitting_gap < 0.2 else '❌ (Severe)'}")
    
    # Performance improvement check
    baseline_f1 = 0.7356  # Previous TabNet result without tuning
    improvement = test_f1 - baseline_f1
    improvement_pct = (improvement / baseline_f1) * 100
    
    print(f"\n📈 Improvement Analysis:")
    print(f"   • Baseline F1 (no tuning): {baseline_f1:.4f}")
    print(f"   • Tuned F1: {test_f1:.4f}")
    print(f"   • Improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)")
    print(f"   • Status: {'✅ Significant' if improvement_pct >= 3.0 else '⚠️ Moderate' if improvement_pct >= 1.0 else '❌ Minimal'}")
    
    # Integration status
    print(f"\n🔗 Integration Status:")
    print(f"   ✅ Ready for Section 6 (Error Analysis)")
    print(f"   ✅ Ready for Section 7 (Ensemble Methods)")
    print(f"   ✅ Predictions saved in ensemble format")
    print(f"   ✅ Interpretability data saved for future analysis")
    print(f"   ✅ Grid search results saved for methodology reporting")
    
    # Next steps recommendation
    print(f"\n💡 Recommendations:")
    if improvement_pct >= 3.0:
        print(f"   🎉 Significant improvement achieved! Include in ensemble methods.")
        print(f"   🔍 Consider interpretability analysis for insights.")
    elif improvement_pct >= 1.0:
        print(f"   👍 Moderate improvement. Include in ensemble for diversity.")
        print(f"   🔄 Consider further hyperparameter exploration if time permits.")
    else:
        print(f"   📊 Limited improvement. May not be competitive with other models.")
        print(f"   🤔 Consider whether to include in ensemble methods.")
    
    print(f"   📋 Compare with Section 4 ML models and Section 5 transformers")
    print(f"   🔬 Proceed to ensemble analysis to leverage model diversity")

else:
    print("❌ TabNet results not available (loaded from checkpoint)")
    if 'best_config' in locals():
        print(f"✅ Best configuration loaded: Config {best_config['config_idx']}")

print("="*80)
print("✅ SECTION 4.5 COMPLETED SUCCESSFULLY")
print("🚀 Ready to proceed with ensemble analysis or model comparison")
print("="*80)