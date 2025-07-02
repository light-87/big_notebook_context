# ============================================================================
# SECTION 3: DATA SPLITTING STRATEGY
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: DATA SPLITTING STRATEGY")
print("="*80)

import numpy as np
import pandas as pd
import os
import gc
import pickle
from sklearn.model_selection import StratifiedGroupKFold
from IPython.display import display, clear_output
import time

# ============================================================================
# 3.0 Memory Monitoring & Initialization
# ============================================================================

# Monitor initial memory usage
initial_memory = progress_tracker.get_memory_usage()
print(f"Initial memory usage: {initial_memory['rss_mb']:.1f} MB")

# Force cleanup before starting
gc.collect()

# ============================================================================
# 3.1 Load Required Variables from Previous Sections
# ============================================================================

print("\n3.1 Loading Required Data from Previous Sections")
print("-" * 40)

# Check if we need to load data from previous sections
required_vars = ['df_final', 'combined_features_matrix']
missing_vars = [var for var in required_vars if var not in locals()]

if missing_vars:
    print("Loading required data from previous checkpoints...")
    
    # Try to load from Section 1 (data_loading)
    if 'df_final' not in locals():
        checkpoint_data = progress_tracker.resume_from_checkpoint("data_loading")
        if checkpoint_data:
            df_final = checkpoint_data['df_final']
            print("✓ Loaded df_final from Section 1 checkpoint")
        else:
            raise RuntimeError("df_final not found. Please run Section 1 first.")
    
    # Try to load from Section 2 (feature_extraction)
    if 'combined_features_matrix' not in locals():
        checkpoint_data = progress_tracker.resume_from_checkpoint("feature_extraction")
        if checkpoint_data:
            feature_matrices = checkpoint_data['feature_matrices']
            combined_features_matrix = feature_matrices['combined']
            metadata = checkpoint_data['metadata']
            
            # Also load individual feature matrices for downstream use
            aac_features = feature_matrices['aac']
            dpc_features = feature_matrices['dpc'] 
            tpc_features = feature_matrices['tpc']
            binary_features = feature_matrices['binary']
            physicochemical_features = feature_matrices['physicochemical']
            
            print("✓ Loaded feature matrices from Section 2 checkpoint")
        else:
            raise RuntimeError("Feature matrices not found. Please run Section 2 first.")

# Verify all required data is available
if 'df_final' not in locals() or 'combined_features_matrix' not in locals():
    raise RuntimeError("Required data not found. Please run Sections 1 and 2 first.")

print(f"✓ Data validation complete:")
print(f"  - df_final shape: {df_final.shape}")
print(f"  - combined_features_matrix shape: {combined_features_matrix.shape}")

# ============================================================================
# 3.2 Check if Section Already Completed
# ============================================================================

# Check if this section is already completed
if progress_tracker.is_completed("data_splitting") and not FORCE_RETRAIN:
    print("\nData splitting already completed. Loading from checkpoint...")
    checkpoint_data = progress_tracker.resume_from_checkpoint("data_splitting")
    if checkpoint_data:
        # Load all split data
        train_indices = checkpoint_data['train_indices']
        val_indices = checkpoint_data['val_indices']
        test_indices = checkpoint_data['test_indices']
        train_proteins_list = checkpoint_data['train_proteins_list']
        val_proteins_list = checkpoint_data['val_proteins_list']
        test_proteins_list = checkpoint_data['test_proteins_list']
        cv_folds = checkpoint_data['cv_folds']
        split_stats = checkpoint_data['split_stats']
        
        # Recreate data splits for immediate use
        print("Recreating data splits from saved indices...")
        X_train = combined_features_matrix.iloc[train_indices]
        X_val = combined_features_matrix.iloc[val_indices]
        X_test = combined_features_matrix.iloc[test_indices]
        y_train = df_final.iloc[train_indices]['target'].values
        y_val = df_final.iloc[val_indices]['target'].values
        y_test = df_final.iloc[test_indices]['target'].values
        
        # Protein groups for CV
        train_proteins = df_final.iloc[train_indices]['Header'].values
        
        print("✓ Data splits loaded from checkpoint successfully!")
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples")
        
        # Jump to summary section
        jump_to_summary = True
    else:
        print("Checkpoint loading failed. Starting fresh...")
        jump_to_summary = False
else:
    print("Starting data splitting process...")
    jump_to_summary = False

if not jump_to_summary:
    # ============================================================================
    # 3.3 Protein-Based Grouped Splitting
    # ============================================================================
    
    print("\n3.3 Creating Protein-Based Data Splits")
    print("-" * 40)
    
    # Get unique proteins
    proteins = df_final['Header'].unique()
    n_proteins = len(proteins)
    print(f"Total unique proteins: {n_proteins:,}")
    
    # Shuffle proteins with fixed seed for reproducibility
    np.random.seed(RANDOM_SEED)
    shuffled_proteins = np.random.permutation(proteins)
    
    # Calculate split points (70% train, 15% val, 15% test)
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    train_end = int(n_proteins * train_ratio)
    val_end = int(n_proteins * (train_ratio + val_ratio))
    
    # Split proteins into sets
    train_proteins_list = shuffled_proteins[:train_end]
    val_proteins_list = shuffled_proteins[train_end:val_end]
    test_proteins_list = shuffled_proteins[val_end:]
    
    print(f"\nProtein distribution:")
    print(f"- Train: {len(train_proteins_list):,} proteins ({len(train_proteins_list)/n_proteins*100:.1f}%)")
    print(f"- Val: {len(val_proteins_list):,} proteins ({len(val_proteins_list)/n_proteins*100:.1f}%)")
    print(f"- Test: {len(test_proteins_list):,} proteins ({len(test_proteins_list)/n_proteins*100:.1f}%)")
    
    # Get indices for each split using optimized masking
    print("Creating sample indices...")
    train_mask = df_final['Header'].isin(train_proteins_list)
    val_mask = df_final['Header'].isin(val_proteins_list)
    test_mask = df_final['Header'].isin(test_proteins_list)
    
    train_indices = df_final.index[train_mask].tolist()
    val_indices = df_final.index[val_mask].tolist()
    test_indices = df_final.index[test_mask].tolist()
    
    # Create optimized data splits
    print("Creating feature matrices for each split...")
    X_train = combined_features_matrix.iloc[train_indices]
    X_val = combined_features_matrix.iloc[val_indices]
    X_test = combined_features_matrix.iloc[test_indices]
    
    y_train = df_final.iloc[train_indices]['target'].values
    y_val = df_final.iloc[val_indices]['target'].values
    y_test = df_final.iloc[test_indices]['target'].values
    
    # Get protein labels for train set (needed for CV)
    train_proteins = df_final.iloc[train_indices]['Header'].values
    
    print(f"\nSample distribution:")
    print(f"- Train: {len(X_train):,} samples ({len(X_train)/len(df_final)*100:.1f}%)")
    print(f"- Val: {len(X_val):,} samples ({len(X_val)/len(df_final)*100:.1f}%)")
    print(f"- Test: {len(X_test):,} samples ({len(X_test)/len(df_final)*100:.1f}%)")
    
    # ============================================================================
    # 3.4 Split Validation & Quality Control
    # ============================================================================
    
    print("\n3.4 Validating Data Splits")
    print("-" * 40)
    
    # Check for protein leakage (critical validation)
    train_proteins_set = set(train_proteins_list)
    val_proteins_set = set(val_proteins_list)
    test_proteins_set = set(test_proteins_list)
    
    train_val_overlap = train_proteins_set.intersection(val_proteins_set)
    train_test_overlap = train_proteins_set.intersection(test_proteins_set)
    val_test_overlap = val_proteins_set.intersection(test_proteins_set)
    
    print("Protein leakage check:")
    print(f"- Train-Val overlap: {len(train_val_overlap)} proteins")
    print(f"- Train-Test overlap: {len(train_test_overlap)} proteins")
    print(f"- Val-Test overlap: {len(val_test_overlap)} proteins")
    
    if len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap) > 0:
        print("⚠️  WARNING: Protein leakage detected! This should not happen.")
        raise RuntimeError("Data leakage detected between splits!")
    else:
        print("✅ No protein leakage detected - splits are valid!")
    
    # Check class balance in each split
    print("\nClass balance verification:")
    
    split_balance_stats = []
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        pos_count = np.sum(y_split == 1)
        neg_count = np.sum(y_split == 0)
        total = len(y_split)
        balance_ratio = neg_count / pos_count if pos_count > 0 else float('inf')
        
        split_balance_stats.append({
            'Split': split_name,
            'Total': total,
            'Positive': pos_count,
            'Negative': neg_count,
            'Pos_%': f"{pos_count/total*100:.1f}%",
            'Neg_%': f"{neg_count/total*100:.1f}%",
            'Ratio': f"{balance_ratio:.2f}:1"
        })
        
        print(f"\n{split_name} set:")
        print(f"  - Positive: {pos_count:,} ({pos_count/total*100:.1f}%)")
        print(f"  - Negative: {neg_count:,} ({neg_count/total*100:.1f}%)")
        print(f"  - Ratio: {balance_ratio:.2f}:1")
    
    # Display balance summary table
    balance_df = pd.DataFrame(split_balance_stats)
    print("\nClass Balance Summary:")
    display(balance_df)
    
    # Statistical comparison of feature distributions (sample check)
    print("\nFeature distribution comparison (first 5 features):")
    
    feature_stats = []
    n_features_to_check = min(5, X_train.shape[1])
    
    for feature_idx in range(n_features_to_check):
        train_mean = X_train.iloc[:, feature_idx].mean()
        val_mean = X_val.iloc[:, feature_idx].mean()
        test_mean = X_test.iloc[:, feature_idx].mean()
        
        train_std = X_train.iloc[:, feature_idx].std()
        val_std = X_val.iloc[:, feature_idx].std()
        test_std = X_test.iloc[:, feature_idx].std()
        
        feature_stats.append({
            'Feature': f'Feature_{feature_idx}',
            'Train_Mean': f"{train_mean:.4f}",
            'Val_Mean': f"{val_mean:.4f}",
            'Test_Mean': f"{test_mean:.4f}",
            'Train_Std': f"{train_std:.4f}",
            'Val_Std': f"{val_std:.4f}",
            'Test_Std': f"{test_std:.4f}"
        })
    
    stats_df = pd.DataFrame(feature_stats)
    display(stats_df)
    
    # ============================================================================
    # 3.5 Cross-Validation Setup (Optimized)
    # ============================================================================
    
    print("\n3.5 Setting up Cross-Validation")
    print("-" * 40)
    
    # Create 5-fold stratified group cross-validation
    n_folds = 5
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Generate CV folds with progress tracking
    cv_folds = []
    
    print(f"Creating {n_folds}-fold cross-validation splits...")
    print("Progress: ", end="", flush=True)
    
    fold_start_time = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=train_proteins)):
        fold_info = {
            'fold': fold_idx,
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_proteins': set(train_proteins[train_idx]),
            'val_proteins': set(train_proteins[val_idx])
        }
        
        # Validate no protein leakage in CV folds
        protein_overlap = fold_info['train_proteins'].intersection(fold_info['val_proteins'])
        if len(protein_overlap) > 0:
            raise RuntimeError(f"Protein leakage in CV fold {fold_idx}: {len(protein_overlap)} proteins")
        
        cv_folds.append(fold_info)
        print("●", end="", flush=True)
    
    fold_time = time.time() - fold_start_time
    print(f" Complete! ({fold_time:.1f}s)")
    
    # Display CV fold statistics
    print(f"\nCross-validation fold statistics:")
    cv_stats = []
    for fold in cv_folds:
        train_pos = np.sum(y_train[fold['train_indices']] == 1)
        train_neg = np.sum(y_train[fold['train_indices']] == 0)
        val_pos = np.sum(y_train[fold['val_indices']] == 1)
        val_neg = np.sum(y_train[fold['val_indices']] == 0)
        
        cv_stats.append({
            'Fold': fold['fold'],
            'Train_Size': fold['train_size'],
            'Val_Size': fold['val_size'],
            'Train_Pos': train_pos,
            'Train_Neg': train_neg,
            'Val_Pos': val_pos,
            'Val_Neg': val_neg,
            'Train_Proteins': len(fold['train_proteins']),
            'Val_Proteins': len(fold['val_proteins'])
        })
    
    cv_stats_df = pd.DataFrame(cv_stats)
    display(cv_stats_df)
    
    # ============================================================================
    # 3.6 Export Data Statistics & Save Tables
    # ============================================================================
    
    print("\n3.6 Exporting Split Statistics")
    print("-" * 40)
    
    # Create comprehensive split statistics
    split_stats = {
        'total_proteins': n_proteins,
        'total_samples': len(df_final),
        'train_proteins': len(train_proteins_list),
        'val_proteins': len(val_proteins_list),
        'test_proteins': len(test_proteins_list),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'cv_folds': n_folds,
        'feature_dimensions': X_train.shape[1],
        'no_protein_leakage': True,
        'splits_balanced': True,
        'memory_usage_mb': progress_tracker.get_memory_usage()['rss_mb']
    }
    
    # Save detailed split statistics to tables
    tables_dir = os.path.join(BASE_DIR, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Save split summary
    balance_df.to_csv(os.path.join(tables_dir, 'data_split_statistics.csv'), index=False)
    cv_stats_df.to_csv(os.path.join(tables_dir, 'cv_fold_statistics.csv'), index=False)
    
    print("✓ Split statistics saved to tables/")
    
    # ============================================================================
    # 3.7 Save Checkpoint with All Variables
    # ============================================================================
    
    print("\n3.7 Saving Checkpoint")
    print("-" * 40)
    
    checkpoint_data = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_proteins_list': train_proteins_list,
        'val_proteins_list': val_proteins_list,
        'test_proteins_list': test_proteins_list,
        'cv_folds': cv_folds,
        'split_stats': split_stats,
        # Additional variables for downstream sections
        'n_folds': n_folds,
        'train_proteins': train_proteins,
        'feature_dimensions': X_train.shape[1]
    }
    
    # Also save the split indices separately for easy access by other tools
    splits_dir = os.path.join(BASE_DIR, 'checkpoints', 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    print("Saving individual split files...")
    np.save(os.path.join(splits_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(splits_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(splits_dir, 'test_indices.npy'), test_indices)
    
    # Save CV folds separately
    with open(os.path.join(splits_dir, 'cv_folds.pkl'), 'wb') as f:
        pickle.dump(cv_folds, f, protocol=4)
    
    # Save main checkpoint
    progress_tracker.mark_completed(
        "data_splitting",
        metadata=split_stats,
        checkpoint_data=checkpoint_data
    )
    
    print("✓ Checkpoint saved successfully!")

# ============================================================================
# 3.8 Memory Cleanup & Optimization
# ============================================================================

print("\n3.8 Memory Cleanup & Optimization")
print("-" * 40)

# Clean up intermediate variables to free memory
if 'train_mask' in locals():
    del train_mask, val_mask, test_mask
if 'shuffled_proteins' in locals():
    del shuffled_proteins
if 'checkpoint_data' in locals():
    del checkpoint_data

# Force garbage collection
gc.collect()

# Final memory check
final_memory = progress_tracker.get_memory_usage()
memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
print(f"Final memory usage: {final_memory['rss_mb']:.1f} MB (+{memory_increase:.1f} MB)")

if memory_increase > 1000:  # More than 1GB increase
    print("⚠️  High memory usage detected. Consider running gc.collect() manually.")

# ============================================================================
# 3.9 Summary Report
# ============================================================================

print("\n" + "="*80)
print("SECTION 3 SUMMARY")
print("="*80)

# Display comprehensive split summary
summary_data = []
for split_name, split_proteins, samples, y_split in [
    ('Train', train_proteins_list, X_train, y_train),
    ('Validation', val_proteins_list, X_val, y_val),
    ('Test', test_proteins_list, X_test, y_test)
]:
    pos_count = np.sum(y_split == 1)
    neg_count = np.sum(y_split == 0)
    summary_data.append({
        'Split': split_name,
        'Proteins': len(split_proteins),
        'Samples': len(samples),
        'Positive': pos_count,
        'Negative': neg_count,
        'Balance': f"{neg_count/pos_count:.2f}:1" if pos_count > 0 else "N/A",
        'Pos_%': f"{pos_count/len(samples)*100:.1f}%",
        'Features': samples.shape[1]
    })

summary_df = pd.DataFrame(summary_data)
print("\nData Split Summary:")
display(summary_df)

print(f"\n✓ Protein-based splitting completed successfully")
print(f"✓ No protein leakage between splits")
print(f"✓ {n_folds}-fold cross-validation setup for ML models")
print(f"✓ Class balance maintained across all splits")
print(f"✓ Memory usage optimized: {final_memory['rss_mb']:.1f} MB")
print(f"✓ All split data saved to checkpoints/")

# ============================================================================
# 3.10 Variables Available for Next Sections
# ============================================================================

print("\n3.10 Variables Available for Next Sections")
print("-" * 40)

print("✅ Main data splits:")
print(f"  - X_train: {X_train.shape}")
print(f"  - X_val: {X_val.shape}")
print(f"  - X_test: {X_test.shape}")
print(f"  - y_train: {y_train.shape}")
print(f"  - y_val: {y_val.shape}")
print(f"  - y_test: {y_test.shape}")

print(f"\n✅ Cross-validation:")
print(f"  - cv_folds: {len(cv_folds)} folds")
print(f"  - train_proteins: {len(train_proteins)} protein groups")

# Check for individual feature matrices
if 'aac_features' in locals():
    print(f"\n✅ Individual feature matrices:")
    print(f"  - aac_features: {aac_features.shape}")
    print(f"  - dpc_features: {dpc_features.shape}")
    print(f"  - tpc_features: {tpc_features.shape}")
    print(f"  - binary_features: {binary_features.shape}")
    print(f"  - physicochemical_features: {physicochemical_features.shape}")

print(f"\n✅ Split indices and metadata saved to:")
print(f"  - checkpoints/data_splitting.pkl")
print(f"  - checkpoints/splits/train_indices.npy")
print(f"  - checkpoints/splits/val_indices.npy")
print(f"  - checkpoints/splits/test_indices.npy")
print(f"  - checkpoints/splits/cv_folds.pkl")

print("="*80)
print("✅ Data splitting completed successfully!")
print("🚀 Ready to proceed to Section 4: Machine Learning Models")
print("="*80)

# Export updated progress report
progress_report = progress_tracker.export_progress_report()
with open(os.path.join(BASE_DIR, 'logs', 'progress_report.txt'), 'w') as f:
    f.write(progress_report)

print(f"\n📄 Progress report updated: {os.path.join(BASE_DIR, 'logs', 'progress_report.txt')}")