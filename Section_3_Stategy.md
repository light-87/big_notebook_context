# Section 3: Data Splitting Strategy - Complete Documentation

## 🎯 **Overview**

**Purpose:** Create protein-based data splits to prevent data leakage, ensure fair evaluation, and set up cross-validation infrastructure for machine learning and transformer model training.

**Key Principle:** **Protein-level grouping** ensures that no protein appears in multiple splits, preventing data leakage that would artificially inflate model performance.

**Duration:** 2-5 minutes  
**Memory Impact:** Low (+100-500 MB)  
**Dependencies:** Sections 0, 1, 2  

---

## 📋 **Section Objectives**

### **Primary Goals**
1. **Create robust train/validation/test splits** with 70%/15%/15% ratios
2. **Prevent data leakage** through protein-based grouping
3. **Maintain class balance** across all splits
4. **Set up cross-validation infrastructure** for ML model evaluation
5. **Validate split quality** through comprehensive checks
6. **Export split statistics** for methodology reporting

### **Quality Assurance**
- ✅ **Zero protein leakage** between train/val/test sets
- ✅ **Balanced class distribution** in all splits
- ✅ **Reproducible splits** with fixed random seeds
- ✅ **Comprehensive validation** with statistical checks
- ✅ **Independent execution** capability with checkpointing

---

## 🔄 **Section Independence & Checkpointing**

### **Input Dependencies**
```python
# From Section 1 (data_loading checkpoint)
df_final                    # Main dataset with 62K+ balanced samples
Header, Position, target    # Protein IDs, positions, and labels

# From Section 2 (feature_extraction checkpoint)  
combined_features_matrix    # Complete feature matrix (2,696+ features)
aac_features               # Individual feature matrices
dpc_features               # (loaded for downstream use)
tpc_features
binary_features
physicochemical_features
```

### **Output Variables for Downstream Sections**
```python
# Main data splits
X_train, X_val, X_test     # Feature matrices for each split
y_train, y_val, y_test     # Target arrays for each split

# Cross-validation infrastructure
cv_folds                   # 5-fold CV definitions with protein grouping
train_proteins             # Protein groups for CV stratification

# Split metadata
train_indices, val_indices, test_indices    # Sample indices
train_proteins_list, val_proteins_list, test_proteins_list  # Protein lists
split_stats                # Comprehensive statistics
```

### **Checkpoint Structure**
```python
checkpoint_data = {
    'train_indices': train_indices,
    'val_indices': val_indices, 
    'test_indices': test_indices,
    'train_proteins_list': train_proteins_list,
    'val_proteins_list': val_proteins_list,
    'test_proteins_list': test_proteins_list,
    'cv_folds': cv_folds,
    'split_stats': split_stats,
    'n_folds': n_folds,
    'train_proteins': train_proteins,
    'feature_dimensions': feature_dimensions
}
```

---

## ⚙️ **Detailed Execution Flow**

### **Phase 1: Initialization & Data Loading (3.1-3.2)**
```python
# Memory monitoring and cleanup
initial_memory = progress_tracker.get_memory_usage()
gc.collect()

# Load required variables from previous sections
# - Automatic checkpoint detection and loading
# - Fallback error handling for missing dependencies
# - Data validation and shape verification
```

**Key Operations:**
- Load `df_final` from Section 1 checkpoint
- Load `combined_features_matrix` and individual feature matrices from Section 2
- Validate data integrity and shapes
- Handle missing dependencies with clear error messages

### **Phase 2: Protein-Based Splitting (3.3)**
```python
# Get unique proteins and shuffle with fixed seed
proteins = df_final['Header'].unique()
np.random.seed(RANDOM_SEED)
shuffled_proteins = np.random.permutation(proteins)

# Calculate split boundaries (70/15/15)
train_end = int(n_proteins * 0.70)
val_end = int(n_proteins * 0.85)

# Create protein lists for each split
train_proteins_list = shuffled_proteins[:train_end]
val_proteins_list = shuffled_proteins[train_end:val_end]
test_proteins_list = shuffled_proteins[val_end:]
```

**Critical Design Decisions:**
- **Protein-level splitting** prevents any protein from appearing in multiple splits
- **Fixed random seed** ensures reproducible splits across runs
- **70/15/15 ratio** provides adequate training data while maintaining robust validation

### **Phase 3: Sample Index Generation (3.3 continued)**
```python
# Create boolean masks for efficient indexing
train_mask = df_final['Header'].isin(train_proteins_list)
val_mask = df_final['Header'].isin(val_proteins_list)
test_mask = df_final['Header'].isin(test_proteins_list)

# Extract indices and create data splits
train_indices = df_final.index[train_mask].tolist()
val_indices = df_final.index[val_mask].tolist()
test_indices = df_final.index[test_mask].tolist()

# Create feature matrices and target arrays
X_train = combined_features_matrix.iloc[train_indices]
X_val = combined_features_matrix.iloc[val_indices]
X_test = combined_features_matrix.iloc[test_indices]
y_train = df_final.iloc[train_indices]['target'].values
y_val = df_final.iloc[val_indices]['target'].values
y_test = df_final.iloc[test_indices]['target'].values
```

### **Phase 4: Comprehensive Validation (3.4)**

#### **Protein Leakage Detection**
```python
# Check for overlapping proteins between splits
train_proteins_set = set(train_proteins_list)
val_proteins_set = set(val_proteins_list)
test_proteins_set = set(test_proteins_list)

train_val_overlap = train_proteins_set.intersection(val_proteins_set)
train_test_overlap = train_proteins_set.intersection(test_proteins_set)
val_test_overlap = val_proteins_set.intersection(test_proteins_set)

# Raise error if any leakage detected
if len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap) > 0:
    raise RuntimeError("Data leakage detected between splits!")
```

#### **Class Balance Verification**
```python
# Calculate class distributions for each split
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    pos_count = np.sum(y_split == 1)
    neg_count = np.sum(y_split == 0)
    balance_ratio = neg_count / pos_count
    # Generate detailed balance statistics
```

#### **Feature Distribution Analysis**
```python
# Compare feature distributions across splits (sample check)
for feature_idx in range(min(5, X_train.shape[1])):
    train_mean = X_train.iloc[:, feature_idx].mean()
    val_mean = X_val.iloc[:, feature_idx].mean()
    test_mean = X_test.iloc[:, feature_idx].mean()
    # Statistical comparison to ensure similar distributions
```

### **Phase 5: Cross-Validation Setup (3.5)**
```python
# Create 5-fold stratified group cross-validation
n_folds = 5
cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

# Generate CV folds with progress tracking
cv_folds = []
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
    
    # Validate no protein leakage within CV folds
    protein_overlap = fold_info['train_proteins'].intersection(fold_info['val_proteins'])
    if len(protein_overlap) > 0:
        raise RuntimeError(f"Protein leakage in CV fold {fold_idx}")
    
    cv_folds.append(fold_info)
```

**Cross-Validation Features:**
- **StratifiedGroupKFold** maintains class balance while respecting protein groups
- **5-fold setup** provides robust evaluation with adequate sample sizes
- **Protein group validation** ensures no leakage within CV folds
- **Comprehensive fold statistics** for evaluation transparency

### **Phase 6: Data Export & Checkpointing (3.6-3.7)**
```python
# Export comprehensive statistics
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
    'memory_usage_mb': memory_usage
}

# Save to multiple formats
balance_df.to_csv('tables/data_split_statistics.csv')
cv_stats_df.to_csv('tables/cv_fold_statistics.csv')
np.save('checkpoints/splits/train_indices.npy', train_indices)
# ... additional exports

# Save main checkpoint
progress_tracker.mark_completed("data_splitting", 
                               metadata=split_stats, 
                               checkpoint_data=checkpoint_data)
```

---

## 📊 **Expected Outputs & Validation**

### **Split Size Validation**
- **Training set:** ~70% of samples (~43,500 samples)
- **Validation set:** ~15% of samples (~9,300 samples)  
- **Test set:** ~15% of samples (~9,300 samples)
- **Total proteins:** Distributed proportionally across splits

### **Class Balance Verification**
```
Expected Balance (1:1 input ratio):
- Train: ~21,750 positive, ~21,750 negative (ratio: ~1.00:1)
- Val: ~4,650 positive, ~4,650 negative (ratio: ~1.00:1) 
- Test: ~4,650 positive, ~4,650 negative (ratio: ~1.00:1)
```

### **Feature Matrix Shapes**
```
Expected Shapes:
- X_train: (~43,500, 2,696+) features
- X_val: (~9,300, 2,696+) features
- X_test: (~9,300, 2,696+) features
- y_train: (~43,500,) labels
- y_val: (~9,300,) labels
- y_test: (~9,300,) labels
```

### **Cross-Validation Structure**
```
CV Folds: 5 folds
Each fold:
- ~80% of training data for training (~34,800 samples)
- ~20% of training data for validation (~8,700 samples)
- No protein overlap between fold train/val sets
- Maintained class stratification
```

---

## 🗂️ **Files Generated**

### **Checkpoint Files**
```
checkpoints/
├── data_splitting.pkl              # Main checkpoint with all variables
└── splits/
    ├── train_indices.npy           # Training sample indices
    ├── val_indices.npy             # Validation sample indices  
    ├── test_indices.npy            # Test sample indices
    └── cv_folds.pkl                # Cross-validation fold definitions
```

### **Statistics Tables**
```
tables/
├── data_split_statistics.csv      # Split summary with class balance
└── cv_fold_statistics.csv         # CV fold details and statistics
```

### **Progress Reports**
```
logs/
└── progress_report.txt             # Updated progress tracking
```

---

## 🚨 **Error Handling & Troubleshooting**

### **Common Issues & Solutions**

#### **1. Missing Dependencies**
- **Symptom:** "Required data not found" error
- **Cause:** Sections 1 or 2 not completed or checkpoints corrupted
- **Solution:** Re-run previous sections or check checkpoint integrity
- **Command:** `progress_tracker.resume_from_checkpoint("data_loading")`

#### **2. Protein Leakage Detection**
- **Symptom:** "Data leakage detected between splits!" error
- **Cause:** Bug in protein splitting logic (should never happen)
- **Solution:** Check random seed consistency and protein list generation
- **Prevention:** Fixed seed ensures reproducible, valid splits

#### **3. Severe Class Imbalance**
- **Symptom:** Warning about extreme class ratios in splits
- **Cause:** Uneven distribution of positive/negative samples across proteins
- **Solution:** Check input data balance from Section 1
- **Monitoring:** Automatic balance validation with ratio reporting

#### **4. Memory Issues**
- **Symptom:** Out of memory during split creation
- **Cause:** Large feature matrices or insufficient RAM
- **Solution:** Reduce batch processing or increase available memory
- **Prevention:** Progressive memory cleanup and monitoring

#### **5. CV Fold Validation Failures**
- **Symptom:** "Protein leakage in CV fold X" error
- **Cause:** StratifiedGroupKFold implementation issue
- **Solution:** Verify sklearn version and GroupKFold parameters
- **Debug:** Check protein group consistency in training data

### **Manual Interventions**

#### **Force Retrain**
```python
# Force Section 3 to retrain from scratch
progress_tracker.force_retrain("data_splitting")
```

#### **Memory Cleanup**
```python
# Manual memory cleanup if needed
import gc
gc.collect()
progress_tracker.trigger_cleanup()
```

#### **Checkpoint Validation**
```python
# Validate checkpoint integrity
checkpoint_data = progress_tracker.resume_from_checkpoint("data_splitting")
if checkpoint_data:
    print("Checkpoint valid")
else:
    print("Checkpoint corrupted - retrain required")
```

---

## 🔗 **Integration with Other Sections**

### **Dependencies (Input Requirements)**
- **Section 0:** Configuration, progress tracking, global variables
- **Section 1:** `df_final` with balanced dataset and protein information
- **Section 2:** `combined_features_matrix` and individual feature matrices

### **Provides to Downstream Sections**
- **Section 4 (ML Models):** Train/val/test splits, CV folds for model training
- **Section 5 (Transformers):** Data splits for transformer training and evaluation  
- **Section 6 (Error Analysis):** Test data and predictions for error analysis
- **Section 7 (Ensemble):** Split data for ensemble model combination
- **Section 8 (Final Evaluation):** Test data for final performance assessment

### **Critical Data Flow**
```python
Section 1 (df_final) → Section 3 (protein-based splits) → Section 4+ (model training)
Section 2 (features) → Section 3 (feature splits) → Section 4+ (model input)
Section 3 (CV folds) → Section 4 (ML training) → Section 6 (error analysis)
Section 3 (test data) → Section 8 (final evaluation) → Section 9 (reporting)
```

---

## 📈 **Performance Metrics & Optimization**

### **Processing Time Benchmarks**
- **Data loading:** ~5-10 seconds (checkpoint dependent)
- **Protein splitting:** ~2-5 seconds (depends on protein count)
- **Index generation:** ~3-8 seconds (depends on dataset size)
- **Validation checks:** ~2-5 seconds (comprehensive testing)
- **CV setup:** ~5-15 seconds (5 folds with validation)
- **Total time:** ~2-5 minutes (typical execution)

### **Memory Usage Patterns**
- **Initial memory:** Baseline from previous sections
- **Peak memory:** +100-500 MB during split creation
- **Final memory:** Minimal increase (<100 MB retained)
- **Cleanup efficiency:** >90% of temporary memory released

### **Optimization Features**
- **Efficient indexing:** Uses `.iloc` for optimal pandas performance
- **Progressive cleanup:** Removes intermediate variables immediately
- **Batch validation:** Validates features in batches to control memory
- **Smart caching:** Reuses computed splits when loading from checkpoint

---

## 🛡️ **Quality Assurance Checklist**

### **Pre-Execution Validation**
- ✅ Sections 0, 1, 2 completed successfully
- ✅ Required checkpoints exist and are valid
- ✅ Sufficient memory available (recommend 4GB+ free)
- ✅ Output directories exist and are writable

### **During Execution Monitoring**
- ✅ Memory usage stays within reasonable bounds
- ✅ Progress indicators show expected timing
- ✅ No error messages or warnings
- ✅ Class balance maintained throughout

### **Post-Execution Verification**
- ✅ No protein leakage detected between any splits
- ✅ Class balance ratios within expected range (0.8:1 to 1.2:1)
- ✅ Split sizes match expected proportions (±2%)
- ✅ CV folds have proper protein grouping
- ✅ All output files generated successfully
- ✅ Checkpoint saves without errors

### **Statistical Validation**
- ✅ Feature distributions similar across splits (sample check)
- ✅ No extreme outliers in split statistics
- ✅ Protein distribution roughly proportional
- ✅ CV fold balance within acceptable variance

---

## 📝 **Best Practices & Recommendations**

### **Before Running Section 3**
1. ✅ **Verify data quality** from Section 1 (balanced classes, clean data)
2. ✅ **Confirm feature extraction** completed successfully in Section 2
3. ✅ **Check available memory** (recommend 4GB+ free for safety)
4. ✅ **Backup critical checkpoints** before proceeding

### **During Execution**
1. 📊 **Monitor memory usage** in console output
2. ⏱️ **Watch for reasonable timing** (should complete in 2-5 minutes)
3. 🔍 **Review validation results** for any warnings
4. 💾 **Ensure checkpoint saving** completes successfully

### **After Completion**
1. ✅ **Verify split statistics** match expected distributions
2. ✅ **Check output files** were generated correctly
3. ✅ **Review memory cleanup** efficiency
4. ✅ **Validate checkpoint integrity** for future resumption

### **Troubleshooting Tips**
- **If memory issues:** Reduce feature matrix size or increase available RAM
- **If timing issues:** Check for competing processes or disk I/O bottlenecks
- **If validation fails:** Review input data quality from previous sections
- **If checkpoint fails:** Check disk space and write permissions

---

## 🚀 **Advanced Configuration Options**

### **Adjustable Parameters**
```python
# Split ratios (must sum to 1.0)
train_ratio = 0.70    # 70% for training
val_ratio = 0.15      # 15% for validation  
test_ratio = 0.15     # 15% for testing

# Cross-validation settings
n_folds = 5           # Number of CV folds
cv_shuffle = True     # Shuffle data in CV
cv_random_state = RANDOM_SEED  # CV reproducibility

# Memory management
cleanup_interval = 5  # Cleanup every N operations
memory_threshold = 0.8  # Cleanup when 80% memory used

# Validation settings
n_features_check = 5  # Number of features to validate
balance_tolerance = 0.2  # Acceptable balance deviation
```

### **Expert Mode Options**
```python
# For advanced users - modify split strategy
CUSTOM_SPLIT_RATIOS = [0.75, 0.125, 0.125]  # Custom train/val/test
STRATIFY_BY_AA = True  # Additional stratification by amino acid type
MINIMUM_PROTEIN_SAMPLES = 5  # Minimum samples per protein for inclusion
FORCE_EXACT_BALANCE = True  # Force exact 1:1 balance in all splits
```

---

## 📚 **Technical Implementation Details**

### **Algorithm Complexity**
- **Protein sorting:** O(p log p) where p = number of proteins
- **Index generation:** O(n) where n = number of samples
- **Split creation:** O(n × f) where f = number of features
- **Validation checks:** O(n + p) for comprehensive validation
- **CV fold generation:** O(n × k) where k = number of folds

### **Memory Complexity**
- **Split indices:** O(n) for storing sample indices
- **Feature matrices:** O(n × f) for split feature matrices
- **CV structure:** O(n × k) for fold definitions
- **Temporary variables:** O(p) for protein lists and masks

### **Statistical Foundations**
- **Stratified sampling** ensures representative class distributions
- **Group-based splitting** prevents data leakage through protein isolation
- **Cross-validation design** provides robust performance estimation
- **Random seed control** ensures experimental reproducibility

---

This comprehensive documentation provides complete guidance for Section 3 execution, troubleshooting, and integration with the broader pipeline. The section is designed to be robust, efficient, and independently executable while maintaining the highest standards for preventing data leakage and ensuring fair model evaluation.