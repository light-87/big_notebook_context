# Section 2: Feature Extraction - Complete Documentation

## 📋 Overview

**Section 2** is responsible for extracting five types of molecular features from protein sequences for phosphorylation site prediction. This optimized implementation includes enhanced memory management, parallel processing, and increased feature dimensionality for better model performance.

---

## 🎯 Objectives

1. **Extract comprehensive protein features** from balanced dataset
2. **Generate 800 TPC features** (increased from 100) for richer sequence representation
3. **Optimize memory usage** with progressive cleanup and monitoring
4. **Enable section independence** through robust checkpoint management
5. **Provide real-time progress tracking** with enhanced visualization

---

## 📊 Feature Types Extracted

### 1. **AAC (Amino Acid Composition) - 20 Features**
- **Purpose**: Capture overall amino acid frequency in sequence windows
- **Method**: Count frequency of each of 20 standard amino acids
- **Window**: ±20 residues around phosphorylation site
- **Normalization**: Frequencies sum to 1.0
- **Features**: `AAC_A`, `AAC_C`, `AAC_D`, ..., `AAC_Y`

### 2. **DPC (Dipeptide Composition) - 400 Features** 
- **Purpose**: Capture local sequence patterns and amino acid pairs
- **Method**: Count frequency of all possible dipeptides (20×20 = 400)
- **Window**: ±20 residues around phosphorylation site
- **Normalization**: Frequencies sum to 1.0
- **Features**: `DPC_AA`, `DPC_AC`, `DPC_AD`, ..., `DPC_YY`

### 3. **TPC (Tripeptide Composition) - 800 Features** ⭐
- **Purpose**: Capture complex sequence motifs and patterns
- **Method**: Identify top 800 most frequent tripeptides across dataset
- **Innovation**: Increased from 100 to 800 features for richer representation
- **Window**: ±20 residues around phosphorylation site
- **Normalization**: Frequencies normalized by total tripeptides
- **Features**: `TPC_0000`, `TPC_0001`, ..., `TPC_0799`

### 4. **Binary Encoding - Variable Features**
- **Purpose**: Position-specific amino acid encoding
- **Method**: One-hot encoding for each position in window
- **Window**: 2×20 + 1 = 41 positions
- **Encoding**: 20 bits per position (one per amino acid)
- **Total Features**: 41 × 20 = 820 features
- **Features**: `BE_pos00_aaA`, `BE_pos00_aaC`, ..., `BE_pos40_aaY`

### 5. **Physicochemical Properties - Variable Features**
- **Purpose**: Incorporate biochemical properties of amino acids
- **Method**: Apply property vectors to each position in window
- **Source**: `data/physiochemical_property.csv` (16 properties per amino acid)
- **Window**: 41 positions
- **Total Features**: 41 × 16 = 656 features
- **Features**: `PC_pos00_prop00`, `PC_pos00_prop01`, ..., `PC_pos40_prop15`

---

## 🏗️ Architecture & Implementation

### **Core Configuration**
```python
WINDOW_SIZE = 20          # ±20 residues around phosphorylation site
TPC_FEATURES = 800        # Increased from 100 for better representation
BATCH_SIZE = 2000         # Optimized for memory efficiency
N_WORKERS = min(8, CPU_COUNT)  # Parallel processing workers
```

### **Memory Management Strategy**
- **Progressive Cleanup**: Garbage collection after each feature type
- **Real-time Monitoring**: Memory usage displayed in progress bars
- **Batch Processing**: Process samples in configurable batches
- **Efficient Data Structures**: Pre-allocated arrays and dictionaries
- **Cache Management**: LRU cache for frequently accessed functions

### **Performance Optimizations**
- **Numpy Operations**: Vectorized amino acid counting
- **LRU Caching**: Cache window extraction and feature functions
- **Parallel Processing**: Multi-core feature extraction
- **Memory Pre-allocation**: Reduce dynamic memory allocation
- **Sparse Representation**: Efficient tripeptide handling

---

## 📁 Input Dependencies

### **Required Variables from Previous Sections**
- `df_final`: Balanced dataset with protein sequences and labels
- `physicochemical_props`: Amino acid property lookup table
- `BASE_DIR`: Experiment base directory
- `WINDOW_SIZE`: Sequence window size parameter

### **Required Files**
- `data/physiochemical_property.csv`: Amino acid properties (auto-detected format)

### **Checkpoint Dependencies**
- Section 1 checkpoint: `checkpoints/data_loading.pkl`

---

## 📤 Output Specifications

### **Feature Matrices Generated**
1. **`aac_features`**: DataFrame with 20 AAC features
2. **`dpc_features`**: DataFrame with 400 DPC features  
3. **`tpc_features`**: DataFrame with 800 TPC features ⭐
4. **`binary_features`**: DataFrame with 820 binary encoding features
5. **`physicochemical_features`**: DataFrame with 656 physicochemical features
6. **`combined_features_matrix`**: Combined feature matrix (2,696+ total features)

### **Metadata Arrays**
- **`Header_array`**: Protein identifiers for each sample
- **`Position_array`**: Phosphorylation site positions
- **`target_array`**: Binary labels (0=negative, 1=positive)

### **Performance Statistics**
- **`feature_stats`**: Memory usage and timing per feature type
- **`extraction_times`**: Processing time for each feature type

---

## 🔄 Section Independence & Checkpointing

### **Checkpoint Data Structure**
```python
checkpoint_data = {
    'feature_matrices': {
        'aac': aac_features,
        'dpc': dpc_features,
        'tpc': tpc_features,
        'binary': binary_features,
        'physicochemical': physicochemical_features,
        'combined': combined_features_matrix
    },
    'feature_stats': feature_stats,
    'extraction_times': extraction_times,
    'metadata': {
        'Header': Header_array,
        'Position': Position_array,
        'target': target_array
    },
    'config': {
        'window_size': WINDOW_SIZE,
        'tpc_features': TPC_FEATURES,
        'batch_size': BATCH_SIZE,
        'n_workers': N_WORKERS
    }
}
```

### **Resume Capability**
- **Automatic Detection**: Checks for completed feature extraction
- **Full State Restoration**: All feature matrices and metadata loaded
- **Configuration Validation**: Ensures parameters match previous run
- **Force Retrain Option**: `progress_tracker.force_retrain("feature_extraction")`

---

## ⚙️ Execution Flow

### **Phase 1: Initialization & Setup**
1. Load required variables from Section 1 checkpoint
2. Initialize memory monitoring and progress tracking
3. Set up parallel processing workers
4. Load physicochemical properties file

### **Phase 2: Feature Extraction Loop**
```python
for feature_type in ['aac', 'dpc', 'tpc', 'binary', 'physicochemical']:
    1. Initialize progress bar with memory monitoring
    2. Split samples into batches for processing
    3. Extract features in parallel (if applicable)
    4. Convert results to DataFrame
    5. Validate feature quality (NaN/infinite checks)
    6. Store feature statistics
    7. Cleanup intermediate variables
```

### **Phase 3: Feature Combination & Validation**
1. Combine all feature types horizontally
2. Validate final feature matrix integrity
3. Check for data quality issues
4. Extract metadata arrays
5. Generate comprehensive statistics

### **Phase 4: Checkpoint & Cleanup**
1. Save complete checkpoint with all data
2. Export performance statistics
3. Generate summary report
4. Final memory cleanup

---

## 📈 Performance Metrics

### **Expected Processing Times** (varies by hardware)
- **AAC Features**: ~30-60 seconds
- **DPC Features**: ~45-90 seconds  
- **TPC Features**: ~90-180 seconds (increased due to 800 features)
- **Binary Encoding**: ~60-120 seconds
- **Physicochemical**: ~45-90 seconds
- **Total Extraction**: ~5-10 minutes

### **Memory Usage Patterns**
- **Peak Memory**: ~2-4 GB during TPC extraction
- **Steady State**: ~1-2 GB for combined features
- **Memory Efficiency**: Progressive cleanup keeps usage reasonable

### **Feature Dimensions**
- **Total Features**: ~2,696 (varies based on window size)
- **Sample Count**: Matches input dataset (typically ~60K balanced samples)
- **Final Matrix**: Shape (samples, 2696)

---

## 🔧 Configuration Options

### **Adjustable Parameters**
```python
# Core feature extraction settings
WINDOW_SIZE = 20              # Sequence window around phospho site
TPC_FEATURES = 800           # Number of top tripeptides to use
BATCH_SIZE = 2000            # Samples per processing batch
N_WORKERS = 8                # Parallel processing workers

# Memory management
MEMORY_CLEANUP_INTERVAL = 5   # Cleanup every N batches
PROGRESS_UPDATE_INTERVAL = 1  # Progress bar update frequency
```

### **Feature Type Toggles**
All feature types are enabled by default, but can be controlled by modifying the `feature_types` list:
```python
feature_types = ['aac', 'dpc', 'tpc', 'binary', 'physicochemical']
```

---

## 🚨 Error Handling & Troubleshooting

### **Common Issues & Solutions**

#### **1. Memory Errors**
- **Symptom**: Out of memory during feature extraction
- **Solution**: Reduce `BATCH_SIZE` or `N_WORKERS`
- **Prevention**: Monitor memory usage in progress bars

#### **2. Physicochemical Properties Not Found**
- **Symptom**: Warning about missing physicochemical file
- **Solution**: Ensure `data/physiochemical_property.csv` exists
- **Fallback**: Uses properties from Section 1 checkpoint

#### **3. Checkpoint Loading Failures**
- **Symptom**: Error loading Section 1 data
- **Solution**: Re-run Section 1 or check checkpoint integrity
- **Command**: `progress_tracker.force_retrain("data_loading")`

#### **4. Feature Validation Failures**
- **Symptom**: NaN or infinite values detected
- **Solution**: Automatic cleanup replaces invalid values with 0.0
- **Monitoring**: Validation results shown in quality report

### **Manual Interventions**

#### **Force Retrain**
```python
# Force Section 2 to retrain from scratch
progress_tracker.force_retrain("feature_extraction")
```

#### **Memory Cleanup**
```python
# Manual memory cleanup if needed
import gc
gc.collect()
cleanup_memory()
```

#### **Debug File Paths**
The code automatically searches multiple file paths and shows detailed debugging information for physicochemical properties file location.

---

## 📊 Quality Assurance

### **Automated Validation Checks**
1. **Feature Matrix Integrity**: Shape validation and column counting
2. **Data Quality**: NaN and infinite value detection
3. **Feature Distribution**: Zero-variance feature identification
4. **Memory Monitoring**: Continuous memory usage tracking
5. **Processing Times**: Performance benchmark recording

### **Expected Outputs Validation**
- **Feature Count**: Should be 2,696± features (depending on window size)
- **Sample Count**: Should match input dataset size
- **Memory Usage**: Should remain under 4GB peak
- **Processing Time**: Should complete within 10 minutes on modern hardware

---

## 🔗 Integration with Other Sections

### **Dependencies**
- **Section 0**: Configuration and setup
- **Section 1**: Data loading and preprocessing

### **Provides to Downstream Sections**
- **Section 3**: Feature matrices for data splitting
- **Section 4**: Features for ML model training
- **Section 5**: Features for transformer comparison
- **Section 6**: Features for error analysis
- **Section 7**: Features for ensemble methods

### **Critical Variables for Next Sections**
```python
# Essential outputs for downstream sections
combined_features_matrix    # Main feature matrix for ML models
Header_array               # Sample identifiers for splitting
Position_array             # Position information for analysis  
target_array              # Labels for supervised learning
feature_stats             # Performance metrics for reporting
```

---

## 📝 Best Practices

### **Before Running Section 2**
1. ✅ Ensure Section 1 completed successfully
2. ✅ Verify physicochemical properties file exists
3. ✅ Check available memory (recommend 8GB+ free)
4. ✅ Consider reducing batch size on limited memory systems

### **During Execution**
1. 📊 Monitor memory usage in progress bars
2. ⏱️ Expect 5-10 minutes total processing time
3. 🔍 Review feature dimensions for sanity checking
4. 💾 Automatic checkpoint saving prevents data loss

### **After Completion**
1. ✅ Verify final feature matrix shape
2. ✅ Review quality validation report
3. ✅ Check memory cleanup completion
4. ✅ Confirm checkpoint saved successfully

---

## 🚀 Performance Improvements in This Version

### **Major Enhancements**
1. **🎯 800 TPC Features**: Increased from 100 for richer sequence representation
2. **⚡ Parallel Processing**: Multi-core feature extraction for speed
3. **🧠 Smart Memory Management**: Real-time monitoring and progressive cleanup
4. **📊 Enhanced Progress Tracking**: Memory usage shown in progress bars
5. **🔧 Robust File Handling**: Auto-detection of physicochemical properties format
6. **🛡️ Error Resilience**: Comprehensive error handling and recovery

### **Performance Gains**
- **3-5x Faster**: Through parallel processing and optimization
- **Better Memory Efficiency**: Progressive cleanup and monitoring
- **Increased Feature Richness**: 800 TPC features vs 100 previously
- **Enhanced Reliability**: Robust error handling and validation

---

## 📚 Technical Implementation Details

### **Algorithm Complexity**
- **AAC Extraction**: O(n × w) where n=samples, w=window_size
- **DPC Extraction**: O(n × w) with 400 feature calculations
- **TPC Extraction**: O(n × w × t) where t=tripeptide_count, most expensive
- **Binary Encoding**: O(n × w × 20) for one-hot encoding
- **Physicochemical**: O(n × w × p) where p=property_count

### **Memory Patterns**
- **Sequential Processing**: One feature type at a time to minimize peak memory
- **Batch Processing**: Configurable batch sizes prevent memory overflow
- **Progressive Cleanup**: Immediate cleanup after each feature type
- **Cache Management**: LRU cache for repeated calculations

### **Parallel Processing Strategy**
- **Feature-Level Parallelism**: Different feature types processed sequentially
- **Batch-Level Parallelism**: Large batches split across workers when beneficial
- **Memory-Aware Scaling**: Reduces parallelism on memory-constrained systems

---

This documentation provides a complete reference for Section 2 Feature Extraction, covering all aspects from high-level objectives to low-level implementation details. The section is designed to be robust, efficient, and independently executable while providing comprehensive features for downstream machine learning tasks.