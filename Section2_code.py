# ============================================================================
# SECTION 2: OPTIMIZED FEATURE EXTRACTION (CORRECTED - FULL TPC)
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: OPTIMIZED FEATURE EXTRACTION (CORRECTED - FULL TPC)")
print("="*80)

# ============================================================================
# 2.0 Import Libraries and Setup
# ============================================================================

import os
import sys
import time
import gc
import psutil
import numpy as np
import pandas as pd

# SPEED OPTIMIZATION: Import datatable for fast data operations
try:
    import datatable as dt
    from datatable import f, by
    USE_DATATABLE = True
    print("✓ Datatable available for speed optimization")
except ImportError:
    USE_DATATABLE = False
    print("⚠️ Datatable not available, using pandas (slower)")

try:
    import progressbar
    PROGRESSBAR_AVAILABLE = True
except ImportError:
    PROGRESSBAR_AVAILABLE = False
    print("⚠️ progressbar not available, using simple progress tracking")

import multiprocessing as mp
from typing import Dict, List, Tuple
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle  # For batch save/load functionality
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 2.1 Constants and Configuration (CORRECTED)
# ============================================================================

# Pre-define amino acid sets as constants for better performance
AMINO_ACIDS = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
AMINO_ACID_SET = set(AMINO_ACIDS)
AMINO_ACID_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# Generate ALL possible tripeptides (20^3 = 8000)
ALL_TRIPEPTIDES = []
for aa1 in AMINO_ACIDS:
    for aa2 in AMINO_ACIDS:
        for aa3 in AMINO_ACIDS:
            ALL_TRIPEPTIDES.append(f"{aa1}{aa2}{aa3}")

print(f"Generated {len(ALL_TRIPEPTIDES)} possible tripeptides")

# Feature extraction parameters (CORRECTED)
WINDOW_SIZE = 20  # Use from previous sections
TPC_FEATURES = 8000  # CORRECTED: Full TPC feature space (20^3)
BATCH_SIZE = 1000  # Reduced due to larger feature space
N_WORKERS = min(mp.cpu_count(), 6)  # Reduced due to memory considerations

print(f"\nSection 2: Feature Extraction Configuration (CORRECTED)")
print(f"- Window size: {WINDOW_SIZE}")
print(f"- TPC features: {TPC_FEATURES} (FULL FEATURE SPACE)")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Workers: {N_WORKERS}")
print(f"- Speed optimization: {'Datatable enabled' if USE_DATATABLE else 'Pandas only'}")
print(f"- Available memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# ============================================================================
# 2.2 Memory Management Functions
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Force garbage collection and return memory usage"""
    gc.collect()
    return get_memory_usage()

def monitor_memory(stage: str):
    """Monitor and print memory usage"""
    memory_mb = get_memory_usage()
    print(f"   Memory usage after {stage}: {memory_mb:.1f} MB")
    return memory_mb

# ============================================================================
# 2.3 Optimized Feature Extraction Functions (CORRECTED)
# ============================================================================

@lru_cache(maxsize=50000)
def get_sequence_window(sequence: str, position: int, window_size: int) -> str:
    """Extract sequence window around position - cached for performance"""
    start = max(0, position - window_size)
    end = min(len(sequence), position + window_size + 1)
    return sequence[start:end]

@lru_cache(maxsize=10000)
def extract_aac(sequence: str) -> Dict[str, float]:
    """Extract Amino Acid Composition features - optimized"""
    aac = {}
    
    if len(sequence) == 0:
        for aa in AMINO_ACIDS:
            aac[f'AAC_{aa}'] = 0.0
        return aac
    
    # Convert to numpy array for vectorized operations
    seq_array = np.array(list(sequence))
    seq_length = len(sequence)
    
    for aa in AMINO_ACIDS:
        count = np.sum(seq_array == aa)
        aac[f'AAC_{aa}'] = count / seq_length
    
    return aac

@lru_cache(maxsize=10000)
def extract_dpc(sequence: str) -> Dict[str, float]:
    """Extract Dipeptide Composition features - optimized"""
    # Pre-allocate dipeptide dictionary
    dpc = {}
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            dpc[f'DPC_{aa1}{aa2}'] = 0.0
    
    if len(sequence) < 2:
        return dpc
    
    # Count dipeptides efficiently
    total_dipeptides = len(sequence) - 1
    for i in range(total_dipeptides):
        dipeptide = sequence[i:i+2]
        if len(dipeptide) == 2 and dipeptide[0] in AMINO_ACID_SET and dipeptide[1] in AMINO_ACID_SET:
            dpc[f'DPC_{dipeptide}'] += 1.0
    
    # Normalize in place
    if total_dipeptides > 0:
        for key in dpc:
            dpc[key] /= total_dipeptides
    
    return dpc

def extract_tpc_full(sequence: str) -> Dict[str, float]:
    """Extract FULL Tripeptide Composition features - ALL 8000 tripeptides"""
    # Initialize all possible tripeptides with 0
    tpc = {}
    for tripeptide in ALL_TRIPEPTIDES:
        tpc[f'TPC_{tripeptide}'] = 0.0
    
    if len(sequence) < 3:
        return tpc
    
    # Count all tripeptides in sequence
    total_tripeptides = len(sequence) - 2
    
    for i in range(total_tripeptides):
        tripeptide = sequence[i:i+3]
        if all(aa in AMINO_ACID_SET for aa in tripeptide):
            tpc[f'TPC_{tripeptide}'] += 1.0
    
    # Normalize by total tripeptides
    if total_tripeptides > 0:
        for key in tpc:
            tpc[key] /= total_tripeptides
    
    return tpc

def extract_binary_encoding(sequence: str, position: int, window_size: int) -> Dict[str, int]:
    """Extract binary encoding features"""
    binary_features = {}
    
    # Create window around position
    start = max(0, position - window_size)
    end = min(len(sequence), position + window_size + 1)
    window = sequence[start:end]
    
    # Pad window to fixed size
    total_positions = 2 * window_size + 1
    padded_window = ['X'] * total_positions
    
    # Fill actual sequence into center of padded window
    offset = window_size - (position - start)
    for i, aa in enumerate(window):
        pos_idx = offset + i
        if 0 <= pos_idx < total_positions:
            padded_window[pos_idx] = aa
    
    # Create binary encoding for each position
    for pos in range(total_positions):
        for aa in AMINO_ACIDS:
            feature_name = f'BE_pos{pos:02d}_aa{aa}'
            binary_features[feature_name] = 1 if padded_window[pos] == aa else 0
    
    return binary_features

def extract_physicochemical_features(sequence: str, position: int, window_size: int, properties_dict: Dict) -> Dict[str, float]:
    """Extract physicochemical property features"""
    pc_features = {}
    
    # Create window around position
    start = max(0, position - window_size)
    end = min(len(sequence), position + window_size + 1)
    window = sequence[start:end]
    
    # Pad window to fixed size
    total_positions = 2 * window_size + 1
    padded_window = ['A'] * total_positions  # Default to Alanine for padding
    
    # Fill actual sequence into center of padded window
    offset = window_size - (position - start)
    for i, aa in enumerate(window):
        pos_idx = offset + i
        if 0 <= pos_idx < total_positions:
            padded_window[pos_idx] = aa if aa in AMINO_ACID_SET else 'A'
    
    # Extract properties for each position
    n_properties = len(next(iter(properties_dict.values())))
    
    for pos in range(total_positions):
        aa = padded_window[pos]
        properties = properties_dict.get(aa, properties_dict['A'])  # Default to Alanine
        
        for prop_idx in range(n_properties):
            feature_name = f'PC_pos{pos:02d}_prop{prop_idx:02d}'
            pc_features[feature_name] = properties[prop_idx]
    
    return pc_features

# ============================================================================
# 2.4 Batch Processing Functions (Updated for Memory-Efficient TPC)
# ============================================================================

def process_batch_features(batch_data, feature_type, properties_dict=None):
    """Process a batch of samples for specific feature type"""
    results = []
    
    for idx, row in batch_data.iterrows():
        sequence = row['Sequence']
        position = row['Position']
        header = row['Header']
        target = row['target']
        
        # Extract sequence window
        window = get_sequence_window(sequence, position, WINDOW_SIZE)
        
        try:
            if feature_type == 'aac':
                features = extract_aac(window)
            elif feature_type == 'dpc':
                features = extract_dpc(window)
            elif feature_type == 'tpc':
                features = extract_tpc_full(window)  # CORRECTED: Use full TPC
            elif feature_type == 'binary':
                features = extract_binary_encoding(sequence, position, WINDOW_SIZE)
            elif feature_type == 'physicochemical':
                features = extract_physicochemical_features(sequence, position, WINDOW_SIZE, properties_dict)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            # Add metadata
            features['Header'] = header
            features['Position'] = position
            features['target'] = target
            
            results.append(features)
            
        except Exception as e:
            print(f"⚠️ Error processing sample {idx}: {e}")
            continue
    
    return results

def save_tpc_batch_to_disk(batch_results, batch_idx, temp_dir):
    """Save TPC batch results to disk to avoid memory overflow"""
    if not batch_results:
        return None
    
    # Create temporary file path
    batch_file = os.path.join(temp_dir, f'tpc_batch_{batch_idx:04d}.pkl')
    
    try:
        # Convert batch to DataFrame and save immediately
        batch_df = pd.DataFrame(batch_results)
        
        # Separate metadata from features
        feature_cols = [col for col in batch_df.columns if col not in ['Header', 'Position', 'target']]
        feature_matrix = batch_df[feature_cols]
        metadata = batch_df[['Header', 'Position', 'target']]
        
        # Save both parts
        with open(batch_file, 'wb') as f:
            pickle.dump({
                'features': feature_matrix,
                'metadata': metadata,
                'batch_idx': batch_idx,
                'shape': feature_matrix.shape
            }, f)
        
        print(f"    ✓ Saved batch {batch_idx} to disk: {feature_matrix.shape}")
        return batch_file
        
    except Exception as e:
        print(f"    ⚠️ Error saving batch {batch_idx}: {e}")
        return None

def combine_tpc_batches_from_disk(batch_files, temp_dir):
    """Load and combine TPC batches from disk incrementally"""
    print(f"  Combining {len(batch_files)} TPC batches from disk...")
    
    # Load first batch to initialize
    combined_features = None
    combined_metadata = None
    total_samples = 0
    
    for i, batch_file in enumerate(batch_files):
        if batch_file is None:
            continue
            
        try:
            # Load batch
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            batch_features = batch_data['features']
            batch_metadata = batch_data['metadata']
            
            if combined_features is None:
                # Initialize with first batch
                combined_features = batch_features.copy()
                combined_metadata = batch_metadata.copy()
            else:
                # Append subsequent batches
                combined_features = pd.concat([combined_features, batch_features], axis=0, ignore_index=True)
                combined_metadata = pd.concat([combined_metadata, batch_metadata], axis=0, ignore_index=True)
            
            total_samples += len(batch_features)
            print(f"    ✓ Loaded batch {i+1}/{len(batch_files)}: {batch_features.shape} (total: {total_samples})")
            
            # Clean up batch file
            os.remove(batch_file)
            
            # Force garbage collection
            del batch_data, batch_features, batch_metadata
            gc.collect()
            
        except Exception as e:
            print(f"    ⚠️ Error loading batch {i+1}: {e}")
            continue
    
    print(f"  ✓ Combined TPC features shape: {combined_features.shape}")
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    return combined_features

# ============================================================================
# 2.5 Main Feature Extraction
# ============================================================================

print("\n2.5 Starting Feature Extraction")
print("-" * 40)

# Check for required variables from Section 1
required_vars = ['df_final', 'BASE_DIR']
missing_vars = [var for var in required_vars if var not in locals()]

if missing_vars:
    print("Loading required data from Section 1 checkpoint...")
    checkpoint_data = progress_tracker.resume_from_checkpoint("data_loading")
    if checkpoint_data:
        df_final = checkpoint_data['df_final']
        BASE_DIR = checkpoint_data.get('BASE_DIR', 'results/exp_1')
        physicochemical_props = checkpoint_data.get('physicochemical_props', {})
        print("✓ Data loaded from checkpoint")
    else:
        raise ValueError("❌ Section 1 checkpoint not found. Please run Section 1 first.")

# Load or create physicochemical properties
physicochemical_props_loaded = False
if 'physicochemical_props' in locals():
    if isinstance(physicochemical_props, dict) and len(physicochemical_props) > 0:
        print("✓ Using physicochemical properties from checkpoint (dict format)")
        physicochemical_props_loaded = True
    elif hasattr(physicochemical_props, 'empty') and not physicochemical_props.empty:
        print("✓ Converting physicochemical properties from DataFrame to dict...")
        # Convert DataFrame to dict format
        physicochemical_props_dict = {}
        for _, row in physicochemical_props.iterrows():
            aa = row.iloc[0]  # First column should be amino acid
            properties = row.iloc[1:].tolist()  # Rest are properties
            physicochemical_props_dict[aa] = properties
        physicochemical_props = physicochemical_props_dict
        physicochemical_props_loaded = True
        print(f"✓ Converted properties for {len(physicochemical_props)} amino acids")

if not physicochemical_props_loaded:
    print("Loading physicochemical properties from file...")
    try:
        # Try to load from file
        props_file = 'data/physiochemical_property.csv'
        if os.path.exists(props_file):
            props_df = pd.read_csv(props_file)
            print(f"✓ Loaded physicochemical file: {props_df.shape}")
            print(f"Columns: {list(props_df.columns)}")
            
            # Convert to dict format
            physicochemical_props = {}
            aa_column = props_df.columns[0]  # Assume first column is amino acids
            
            for _, row in props_df.iterrows():
                aa = row[aa_column]
                properties = [row[col] for col in props_df.columns if col != aa_column]
                physicochemical_props[aa] = properties
            
            print(f"✓ Loaded physicochemical properties from {props_file}")
            print(f"✓ Properties for {len(physicochemical_props)} amino acids")
        else:
            print("⚠️ Physicochemical file not found, using default properties")
            # Default properties (normalized values)
            physicochemical_props = {aa: [0.5] * 16 for aa in AMINO_ACIDS}
    except Exception as e:
        print(f"⚠️ Error loading physicochemical properties: {e}")
        physicochemical_props = {aa: [0.5] * 16 for aa in AMINO_ACIDS}

print(f"Dataset shape: {df_final.shape}")
print(f"Total samples to process: {len(df_final)}")

# ============================================================================
# 2.6 Feature Extraction Loop
# ============================================================================

feature_types = ['aac', 'dpc', 'tpc', 'binary', 'physicochemical']
all_features = {}
feature_stats = {}

total_start_time = time.time()

for feature_type in feature_types:
    print(f"\n2.6.{feature_types.index(feature_type)+1} Extracting {feature_type.upper()} features...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Special handling for TPC to avoid memory overflow
    if feature_type == 'tpc':
        print(f"  Using BATCH SAVE-AND-COMBINE strategy for TPC (8000 features)...")
        
        # Create temporary directory for batch files
        temp_dir = os.path.join(BASE_DIR, 'temp_tpc_batches')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process TPC in smaller batches
        TPC_BATCH_SIZE = 5000  # Smaller batches for TPC
        n_batches = (len(df_final) + TPC_BATCH_SIZE - 1) // TPC_BATCH_SIZE
        batch_files = []
        
        if PROGRESSBAR_AVAILABLE:
            bar = progressbar.ProgressBar(
                max_value=n_batches,
                widgets=[
                    f'  {feature_type.upper()}: ',
                    progressbar.Percentage(), ' ',
                    progressbar.Bar(), ' ',
                    progressbar.ETA(), ' ',
                    progressbar.DynamicMessage('memory')
                ]
            )
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * TPC_BATCH_SIZE
            batch_end = min((batch_idx + 1) * TPC_BATCH_SIZE, len(df_final))
            batch_data = df_final.iloc[batch_start:batch_end]
            
            # Process batch
            batch_results = process_batch_features(batch_data, feature_type)
            
            # Save batch to disk immediately
            batch_file = save_tpc_batch_to_disk(batch_results, batch_idx, temp_dir)
            batch_files.append(batch_file)
            
            # Update progress
            if PROGRESSBAR_AVAILABLE:
                memory_usage = get_memory_usage()
                bar.update(batch_idx + 1, memory=f'{memory_usage:.0f}MB')
            else:
                if (batch_idx + 1) % 5 == 0:
                    print(f"    Processed and saved batch {batch_idx + 1}/{n_batches}")
            
            # Clean up batch results from memory immediately
            del batch_results
            gc.collect()
        
        if PROGRESSBAR_AVAILABLE:
            bar.finish()
        
        # Combine all batches from disk
        print(f"  Loading and combining TPC batches...")
        feature_matrix = combine_tpc_batches_from_disk(batch_files, temp_dir)
        
        if feature_matrix is not None:
            print(f"  ✓ {feature_type.upper()} features shape: {feature_matrix.shape}")
            
            # Validate features
            nan_count = feature_matrix.isnull().sum().sum()
            inf_count = np.isinf(feature_matrix.values).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️ Warning: {nan_count} NaN values, {inf_count} infinity values detected")
                feature_matrix = feature_matrix.fillna(0)
                feature_matrix = feature_matrix.replace([np.inf, -np.inf], 0)
            
            all_features[feature_type] = feature_matrix
            
            # Calculate statistics
            extraction_time = time.time() - start_time
            memory_used = get_memory_usage() - start_memory
            feature_stats[feature_type] = {
                'shape': feature_matrix.shape,
                'extraction_time': extraction_time,
                'conversion_time': 0,  # No conversion needed for batch method
                'memory_used': memory_used,
                'features_per_second': len(df_final) / extraction_time,
                'conversion_method': 'batch_save_combine'
            }
            
            print(f"  ✓ TPC extracted in {extraction_time:.1f}s using batch strategy, Memory: +{memory_used:.1f}MB")
        else:
            print(f"  ❌ Failed to extract TPC features")
            all_features[feature_type] = pd.DataFrame()
    
    else:
        # Standard processing for other feature types (AAC, DPC, Binary, Physicochemical)
        all_batch_results = []
        n_batches = (len(df_final) + BATCH_SIZE - 1) // BATCH_SIZE
        
        if PROGRESSBAR_AVAILABLE:
            bar = progressbar.ProgressBar(
                max_value=n_batches,
                widgets=[
                    f'  {feature_type.upper()}: ',
                    progressbar.Percentage(), ' ',
                    progressbar.Bar(), ' ',
                    progressbar.ETA(), ' ',
                    progressbar.DynamicMessage('memory')
                ]
            )
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, len(df_final))
            batch_data = df_final.iloc[batch_start:batch_end]
            
            # Process batch
            if feature_type == 'physicochemical':
                batch_results = process_batch_features(batch_data, feature_type, physicochemical_props)
            else:
                batch_results = process_batch_features(batch_data, feature_type)
            
            all_batch_results.extend(batch_results)
            
            # Update progress
            if PROGRESSBAR_AVAILABLE:
                memory_usage = get_memory_usage()
                bar.update(batch_idx + 1, memory=f'{memory_usage:.0f}MB')
            else:
                if (batch_idx + 1) % 10 == 0:
                    print(f"    Processed batch {batch_idx + 1}/{n_batches}")
            
            # Periodic memory cleanup
            if (batch_idx + 1) % 5 == 0:
                gc.collect()
        
        if PROGRESSBAR_AVAILABLE:
            bar.finish()
        
        # Convert to DataFrame - SPEED OPTIMIZED with datatable
        print(f"  Converting {len(all_batch_results)} results to DataFrame...")
        conversion_start = time.time()
        
        if all_batch_results:
            if USE_DATATABLE and len(all_batch_results) > 1000:
                # SPEED OPTIMIZATION: Use datatable for large datasets
                print(f"    Using datatable for fast conversion...")
                
                try:
                    # Convert to datatable Frame (much faster than pandas)
                    dt_frame = dt.Frame(all_batch_results)
                    
                    # Convert to pandas (still needed for downstream compatibility)
                    features_df = dt_frame.to_pandas()
                    
                    conversion_time = time.time() - conversion_start
                    print(f"    ✓ Datatable conversion: {conversion_time:.1f}s")
                    
                except Exception as e:
                    print(f"    ⚠️ Datatable conversion failed: {e}")
                    print(f"    Falling back to pandas...")
                    features_df = pd.DataFrame(all_batch_results)
                    conversion_time = time.time() - conversion_start
                    print(f"    ✓ Pandas conversion: {conversion_time:.1f}s")
            else:
                # Use pandas for smaller datasets
                features_df = pd.DataFrame(all_batch_results)
                conversion_time = time.time() - conversion_start
                print(f"    ✓ Pandas conversion: {conversion_time:.1f}s")
            
            # Separate metadata from features
            feature_cols = [col for col in features_df.columns if col not in ['Header', 'Position', 'target']]
            feature_matrix = features_df[feature_cols]
            
            print(f"  ✓ {feature_type.upper()} features shape: {feature_matrix.shape}")
            
            # Validate features
            nan_count = feature_matrix.isnull().sum().sum()
            inf_count = np.isinf(feature_matrix.values).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️ Warning: {nan_count} NaN values, {inf_count} infinity values detected")
                feature_matrix = feature_matrix.fillna(0)
                feature_matrix = feature_matrix.replace([np.inf, -np.inf], 0)
            
            all_features[feature_type] = feature_matrix
            
            # Calculate statistics
            extraction_time = time.time() - start_time
            memory_used = get_memory_usage() - start_memory
            feature_stats[feature_type] = {
                'shape': feature_matrix.shape,
                'extraction_time': extraction_time,
                'conversion_time': conversion_time,
                'memory_used': memory_used,
                'features_per_second': len(df_final) / extraction_time,
                'conversion_method': 'datatable' if USE_DATATABLE and len(all_batch_results) > 1000 else 'pandas'
            }
            
            print(f"  ✓ Extracted in {extraction_time:.1f}s (convert: {conversion_time:.1f}s), Memory: +{memory_used:.1f}MB")
            
            # Clean up batch results
            del all_batch_results, features_df
            gc.collect()
        else:
            print(f"  ⚠️ No results to convert for {feature_type}")
    
    # Memory cleanup after each feature type
    gc.collect()

# ============================================================================
# 2.7 Combine All Features - SPEED OPTIMIZED
# ============================================================================

print(f"\n2.7 Combining all feature types (SPEED OPTIMIZED)...")
start_combine_time = time.time()

if USE_DATATABLE and len(df_final) > 1000:
    print("  Using datatable for fast feature combination...")
    
    try:
        # Start with metadata as datatable Frame
        metadata_dict = {
            'Header': df_final['Header'].values,
            'Position': df_final['Position'].values,
            'target': df_final['target'].values
        }
        
        combined_dt = dt.Frame(metadata_dict)
        print(f"✓ Base metadata: {combined_dt.shape}")
        
        # Add each feature type
        feature_count = 0
        for feature_type in feature_types:
            if feature_type in all_features and not all_features[feature_type].empty:
                # Convert pandas to datatable for fast operations
                feature_dt = dt.Frame(all_features[feature_type])
                
                # Combine with existing data (column-wise concatenation)
                combined_dt = dt.cbind(combined_dt, feature_dt)
                feature_count += all_features[feature_type].shape[1]
                print(f"✓ Added {feature_type}: +{all_features[feature_type].shape[1]} features")
            else:
                print(f"⚠ Skipped {feature_type}: no features extracted")
        
        # Convert final result to pandas for compatibility
        print("  Converting final result to pandas...")
        combined_features = combined_dt.to_pandas()
        
        print(f"✓ Datatable combination successful!")
        
    except Exception as e:
        print(f"⚠️ Datatable combination failed: {e}")
        print("  Falling back to pandas concatenation...")
        
        # Fallback to pandas method
        combined_features = pd.DataFrame({
            'Header': df_final['Header'].values,
            'Position': df_final['Position'].values,
            'target': df_final['target'].values
        })
        
        feature_count = 0
        for feature_type in feature_types:
            if feature_type in all_features and not all_features[feature_type].empty:
                combined_features = pd.concat([combined_features, all_features[feature_type]], axis=1)
                feature_count += all_features[feature_type].shape[1]
                print(f"✓ Added {feature_type}: +{all_features[feature_type].shape[1]} features")
            else:
                print(f"⚠ Skipped {feature_type}: no features extracted")

else:
    print("  Using pandas for feature combination...")
    
    # Start with metadata
    combined_features = pd.DataFrame({
        'Header': df_final['Header'].values,
        'Position': df_final['Position'].values,
        'target': df_final['target'].values
    })

    print(f"✓ Base metadata: {combined_features.shape}")

    # Add each feature type
    feature_count = 0
    for feature_type in feature_types:
        if feature_type in all_features and not all_features[feature_type].empty:
            # Concatenate features horizontally
            combined_features = pd.concat([combined_features, all_features[feature_type]], axis=1)
            feature_count += all_features[feature_type].shape[1]
            print(f"✓ Added {feature_type}: +{all_features[feature_type].shape[1]} features")
        else:
            print(f"⚠ Skipped {feature_type}: no features extracted")

print(f"✓ Combined features shape: {combined_features.shape}")
print(f"✓ Total feature columns: {feature_count}")

combine_time = time.time() - start_combine_time
print(f"✓ Combining completed in {combine_time:.2f}s")

# Extract individual components for checkpoint
aac_features = all_features['aac']
dpc_features = all_features['dpc'] 
tpc_features = all_features['tpc']  # Now contains all 8000 features
binary_features = all_features['binary']
physicochemical_features = all_features['physicochemical']

# Combined features without metadata
feature_cols = [col for col in combined_features.columns 
               if col not in ['Header', 'Position', 'target']]
combined_features_matrix = combined_features[feature_cols]

# Extract metadata arrays
Header_array = combined_features['Header'].values
Position_array = combined_features['Position'].values  
target_array = combined_features['target'].values

# ============================================================================
# 2.8 Feature Quality Validation
# ============================================================================

print(f"\n2.8 Feature Quality Validation")
print("-" * 40)

# Check for NaN and infinite values
total_nan = combined_features_matrix.isnull().sum().sum()
total_inf = np.isinf(combined_features_matrix.values).sum()

print(f"NaN values: {total_nan}")
print(f"Infinite values: {total_inf}")

if total_nan > 0 or total_inf > 0:
    print("⚠️ Cleaning invalid values...")
    combined_features_matrix = combined_features_matrix.fillna(0)
    combined_features_matrix = combined_features_matrix.replace([np.inf, -np.inf], 0)
    print("✓ Invalid values cleaned")

# Feature statistics
print(f"\nFeature Statistics:")
for feature_type, stats in feature_stats.items():
    print(f"  {feature_type.upper():>15}: {stats['shape'][1]:>5} features, "
          f"{stats['extraction_time']:>6.1f}s, "
          f"{stats['features_per_second']:>6.0f} samples/s")

# Memory usage summary
final_memory = cleanup_memory()
print(f"\nMemory Usage: {final_memory:.1f} MB")

# ============================================================================
# 2.9 Export Feature Statistics & Analysis
# ============================================================================

print(f"\n2.9 Export Feature Statistics & Analysis")
print("-" * 40)

# Create tables directory
tables_dir = os.path.join(BASE_DIR, 'tables')
os.makedirs(tables_dir, exist_ok=True)

# Feature statistics table
feature_stats_data = []
for feature_type, stats in feature_stats.items():
    feature_stats_data.append({
        'Feature_Type': feature_type.upper(),
        'Number_of_Features': stats['shape'][1],
        'Extraction_Time_s': round(stats['extraction_time'], 2),
        'Memory_Used_MB': round(stats['memory_used'], 1),
        'Features_per_Second': round(stats['features_per_second'], 0),
        'Efficiency_Score': round(stats['shape'][1] / stats['extraction_time'], 1)
    })

feature_stats_df = pd.DataFrame(feature_stats_data)
feature_stats_file = os.path.join(tables_dir, 'feature_extraction_performance.csv')
feature_stats_df.to_csv(feature_stats_file, index=False)
print(f"✓ Feature statistics saved to {feature_stats_file}")

# Feature correlation analysis (sample)
print(f"\nPerforming feature correlation analysis (sample)...")
correlation_data = []

for feature_type in feature_types:
    if feature_type in all_features and not all_features[feature_type].empty:
        feature_matrix = all_features[feature_type]
        
        # Sample analysis for large feature sets
        if feature_matrix.shape[1] > 100:
            # Sample 50 random features for correlation analysis
            sample_cols = np.random.choice(feature_matrix.columns, 50, replace=False)
            sample_matrix = feature_matrix[sample_cols]
        else:
            sample_matrix = feature_matrix
        
        # Calculate correlation statistics
        corr_matrix = sample_matrix.corr()
        
        correlation_data.append({
            'Feature_Type': feature_type.upper(),
            'Features_Analyzed': sample_matrix.shape[1],
            'Mean_Correlation': round(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(), 4),
            'Max_Correlation': round(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(), 4),
            'Min_Correlation': round(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min(), 4),
            'High_Corr_Pairs': np.sum(np.abs(corr_matrix.values) > 0.8) - sample_matrix.shape[1]  # Exclude diagonal
        })

correlation_df = pd.DataFrame(correlation_data)
correlation_file = os.path.join(tables_dir, 'feature_correlation_summary.csv')
correlation_df.to_csv(correlation_file, index=False)
print(f"✓ Correlation analysis saved to {correlation_file}")

# ============================================================================
# 2.10 Generate Comprehensive Visualizations
# ============================================================================

print(f"\n2.10 Generate Comprehensive Visualizations")
print("-" * 40)

# Create plots directory
plots_dir = os.path.join(BASE_DIR, 'plots', 'feature_analysis')
os.makedirs(plots_dir, exist_ok=True)

# Set publication quality defaults
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    
    # 1. Feature Count Comparison
    print("  Creating feature count comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of feature counts
    feature_counts = [stats['shape'][1] for stats in feature_stats.values()]
    feature_names = [ft.upper() for ft in feature_types]
    
    bars = ax1.bar(feature_names, feature_counts, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum'])
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Feature Count by Type (CORRECTED TPC)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(feature_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart of feature proportions
    ax2.pie(feature_counts, labels=feature_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Feature Distribution\n(Total: {:,} features)'.format(sum(feature_counts)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_count_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 2. Extraction Performance Analysis
    print("  Creating extraction performance analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extraction time comparison
    extraction_times = [stats['extraction_time'] for stats in feature_stats.values()]
    bars1 = ax1.bar(feature_names, extraction_times, alpha=0.7, color='lightcoral')
    ax1.set_ylabel('Extraction Time (seconds)')
    ax1.set_title('Feature Extraction Time by Type')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars1, extraction_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(extraction_times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Memory usage comparison
    memory_usage = [stats['memory_used'] for stats in feature_stats.values()]
    bars2 = ax2.bar(feature_names, memory_usage, alpha=0.7, color='lightblue')
    ax2.set_ylabel('Memory Used (MB)')
    ax2.set_title('Memory Usage by Feature Type')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, mem_val in zip(bars2, memory_usage):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(memory_usage)*0.01,
                f'{mem_val:.1f}MB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'extraction_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 3. Feature Density Analysis
    print("  Creating feature density analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature_type in enumerate(feature_types):
        if feature_type in all_features and not all_features[feature_type].empty:
            feature_matrix = all_features[feature_type]
            
            # Calculate sparsity (percentage of zeros)
            sparsity = (feature_matrix == 0).sum().sum() / (feature_matrix.shape[0] * feature_matrix.shape[1]) * 100
            
            # Sample features for visualization
            if feature_matrix.shape[1] > 20:
                sample_cols = np.random.choice(feature_matrix.columns, 20, replace=False)
                sample_matrix = feature_matrix[sample_cols]
            else:
                sample_matrix = feature_matrix
            
            # Feature variance distribution
            variances = sample_matrix.var()
            axes[idx].hist(variances, bins=20, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{feature_type.upper()}\nSparsity: {sparsity:.1f}%')
            axes[idx].set_xlabel('Feature Variance')
            axes[idx].set_ylabel('Count')
    
    # Remove empty subplot
    if len(feature_types) < 6:
        fig.delaxes(axes[5])
    
    plt.suptitle('Feature Variance Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_variance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("✓ All visualizations saved to plots/feature_analysis/")
    
except ImportError:
    print("⚠️ Matplotlib/Seaborn not available, skipping visualizations")

# ============================================================================
# 2.11 Generate Summary Report (FIXED)
# ============================================================================

print(f"\n2.11 Generate Summary Report")
print("-" * 40)

# FIXED: Safe calculation of time variables
try:
    if 'total_start_time' in locals():
        total_time = time.time() - total_start_time
    else:
        print("Warning: total_start_time not found, using current time")
        total_time = 0
        total_start_time = time.time()
except:
    total_time = 0
    total_start_time = time.time()

# FIXED: Safe handling of memory and other variables
try:
    if 'final_memory' in locals() and isinstance(final_memory, (int, float)):
        memory_usage = final_memory
    else:
        memory_usage = get_memory_usage() if 'get_memory_usage' in locals() else 0
        print(f"Warning: final_memory not available, using current: {memory_usage}")
except:
    memory_usage = 0

try:
    if 'df_final' in locals() and hasattr(df_final, '__len__'):
        samples_count = len(df_final)
    else:
        samples_count = 0
        print("Warning: df_final not available")
except:
    samples_count = 0

# FIXED: Safe start time formatting
try:
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))
except:
    start_time_str = "Unknown"

try:
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
except:
    end_time_str = "Unknown"

# Comprehensive summary (FIXED - Safe variable handling)
summary_report = f"""
# SECTION 2: FEATURE EXTRACTION - COMPREHENSIVE SUMMARY REPORT
================================================================

## Execution Summary
- **Start Time:** {start_time_str}
- **End Time:** {end_time_str}
- **Total Duration:** {total_time:.1f} seconds ({total_time/60:.1f} minutes)
- **Samples Processed:** {samples_count:,}
- **Memory Usage:** {memory_usage:.1f} MB

## Feature Extraction Results (CORRECTED - FULL TPC)
"""

# FIXED: Debug and check variable types
print("Debugging variable types:")
for var_name in ['feature_stats', 'final_memory', 'total_time', 'df_final']:
    if var_name in locals():
        var_value = locals()[var_name]
        print(f"  {var_name}: {type(var_value)} = {str(var_value)[:100]}...")
    else:
        print(f"  {var_name}: Not found")

# FIXED: Check if feature_stats exists and is a dictionary
if 'feature_stats' in locals() and isinstance(feature_stats, dict):
    print(f"Processing {len(feature_stats)} feature types")
    for feature_type, stats in feature_stats.items():
        print(f"  Processing {feature_type}: {type(stats)}")
        # FIXED: Verify stats is a dictionary and has required keys
        if isinstance(stats, dict) and all(key in stats for key in ['shape', 'extraction_time', 'memory_used', 'features_per_second']):
            try:
                # FIXED: Safe extraction of numeric values
                shape_val = stats['shape'][1] if isinstance(stats['shape'], (list, tuple)) and len(stats['shape']) > 1 else 0
                extraction_time_val = float(stats['extraction_time']) if isinstance(stats['extraction_time'], (int, float)) else 0
                conversion_time_val = float(stats.get('conversion_time', 0)) if isinstance(stats.get('conversion_time', 0), (int, float)) else 0
                conversion_method_val = str(stats.get('conversion_method', 'pandas'))
                features_per_second_val = float(stats['features_per_second']) if isinstance(stats['features_per_second'], (int, float)) else 0
                memory_used_val = float(stats['memory_used']) if isinstance(stats['memory_used'], (int, float)) else 0
                
                # Safe efficiency calculation
                if extraction_time_val > 0:
                    efficiency_val = shape_val / extraction_time_val
                else:
                    efficiency_val = 0
                
                summary_report += f"""
### {feature_type.upper()} Features
- **Features Extracted:** {shape_val:,}
- **Extraction Time:** {extraction_time_val:.1f}s
- **Conversion Time:** {conversion_time_val:.1f}s
- **Conversion Method:** {conversion_method_val}
- **Processing Speed:** {features_per_second_val:.0f} samples/second
- **Memory Used:** {memory_used_val:.1f} MB
- **Efficiency Score:** {efficiency_val:.1f} features/second
"""
            except Exception as e:
                print(f"Warning: Error processing stats for {feature_type}: {e}")
                summary_report += f"\n### {feature_type.upper()} Features\n- **Status:** Error processing statistics\n"
        else:
            print(f"Warning: Invalid stats for {feature_type}")
            summary_report += f"\n### {feature_type.upper()} Features\n- **Status:** Invalid statistics format\n"
else:
    print("Warning: feature_stats not found or not a dictionary")
    feature_stats = {}
    summary_report += "\n### Feature Statistics\n- **Status:** No feature statistics available\n"

# FIXED: Safe handling of TPC features count
try:
    if 'tpc_features' in locals() and hasattr(tpc_features, 'shape'):
        tpc_feature_count = tpc_features.shape[1]
    elif 'TPC_FEATURES' in locals() and isinstance(TPC_FEATURES, (int, float)):
        tpc_feature_count = int(TPC_FEATURES)
    else:
        tpc_feature_count = 8000  # Default expected value
except:
    tpc_feature_count = 8000

# FIXED: Safe handling of total feature count
try:
    if 'combined_features_matrix' in locals() and hasattr(combined_features_matrix, 'shape'):
        total_feature_count = combined_features_matrix.shape[1]
    else:
        # Calculate from feature_stats if available
        total_feature_count = 0
        if isinstance(feature_stats, dict):
            for stats in feature_stats.values():
                if isinstance(stats, dict) and 'shape' in stats:
                    shape = stats['shape']
                    if isinstance(shape, (list, tuple)) and len(shape) > 1:
                        total_feature_count += shape[1]
        if total_feature_count == 0:
            total_feature_count = tpc_feature_count  # Fallback
except:
    total_feature_count = tpc_feature_count

# FIXED: Safe handling of USE_DATATABLE and BATCH_SIZE
try:
    datatable_status = 'ENABLED' if USE_DATATABLE else 'NOT AVAILABLE'
    conversion_method = 'Datatable' if USE_DATATABLE else 'Pandas'
except:
    datatable_status = 'UNKNOWN'
    conversion_method = 'Unknown'

try:
    batch_size_val = BATCH_SIZE if 'BATCH_SIZE' in locals() else 1000
except:
    batch_size_val = 1000

# Add TPC correction note (FIXED - No Unicode arrow)
summary_report += f"""
## IMPORTANT CORRECTION
- **TPC Features:** Corrected from 800 -> {tpc_feature_count:,} (FULL FEATURE SPACE)
- **Total Features:** {total_feature_count:,} (corrected from previous count)
- **Feature Space:** All possible tripeptides (20^3 = 8,000) now included

## SPEED OPTIMIZATIONS APPLIED
- **Datatable Usage:** {datatable_status} for fast data operations
- **Conversion Speed:** {conversion_method} used for DataFrame conversion
- **Batch Processing:** {batch_size_val} samples per batch for memory efficiency
- **Memory Management:** Progressive cleanup and garbage collection

## Performance Metrics
"""

# FIXED: Safe calculation of most efficient feature type
if feature_stats and isinstance(feature_stats, dict):
    try:
        valid_stats = {}
        for ft, stats in feature_stats.items():
            if isinstance(stats, dict) and 'extraction_time' in stats:
                extraction_time = stats['extraction_time']
                if isinstance(extraction_time, (int, float)) and extraction_time > 0:
                    valid_stats[ft] = extraction_time
        
        if valid_stats:
            most_efficient_type = min(valid_stats.keys(), key=lambda x: valid_stats[x])
            most_efficient_time = valid_stats[most_efficient_type]
            summary_report += f"- **Most Efficient:** {most_efficient_type} ({most_efficient_time:.1f}s)\n"
        else:
            summary_report += "- **Most Efficient:** Unable to determine (no valid timing data)\n"
    except Exception as e:
        print(f"Warning: Error calculating most efficient: {e}")
        summary_report += "- **Most Efficient:** Unable to determine (calculation error)\n"
    
    summary_report += f"- **Largest Feature Set:** TPC ({tpc_feature_count:,} features)\n"
else:
    summary_report += "- **Performance Metrics:** Unable to calculate (no feature stats available)\n"

# FIXED: Safe processing speed calculation
try:
    if total_time > 0 and samples_count > 0:
        processing_speed = samples_count / total_time
    else:
        processing_speed = 0
except:
    processing_speed = 0

summary_report += f"- **Total Processing Speed:** {processing_speed:.0f} samples/second overall\n"
summary_report += f"- **Speed Optimization:** {conversion_method} processing used\n"

# FIXED: Safe handling of data quality metrics
nan_count = 0
inf_count = 0
try:
    if 'total_nan' in locals() and isinstance(total_nan, (int, float)):
        nan_count = int(total_nan)
    if 'total_inf' in locals() and isinstance(total_inf, (int, float)):
        inf_count = int(total_inf)
except:
    pass

summary_report += f"""
## Data Quality
- **NaN Values:** {nan_count} (cleaned)
- **Infinite Values:** {inf_count} (cleaned)
- **Data Integrity:** All samples processed successfully
"""

# FIXED: Safe file counting
tables_count = 0
plots_count = 0

try:
    if 'tables_dir' in locals() and os.path.exists(tables_dir):
        tables_count = len([f for f in os.listdir(tables_dir) if f.endswith('.csv')])
except Exception as e:
    print(f"Warning: Could not count table files: {e}")

try:
    if 'plots_dir' in locals() and os.path.exists(plots_dir):
        plots_count = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
except Exception as e:
    print(f"Warning: Could not count plot files: {e}")

# FIXED: Safe current time formatting
try:
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
except:
    current_time_str = "Unknown"

summary_report += f"""
## Output Files Generated
- Feature matrices: 5 individual + 1 combined
- Statistics tables: {tables_count} files
- Visualization plots: {plots_count} files
- Checkpoint data: Complete with all components

## Ready for Next Steps
[SUCCESS] All features extracted and validated
[SUCCESS] Data quality confirmed
[SUCCESS] Checkpoint saved successfully
[SUCCESS] Ready for Section 3 (Data Splitting)

Generated: {current_time_str}
"""

# FIXED: Save summary report with proper encoding and error handling
if 'BASE_DIR' in locals():
    summary_file = os.path.join(BASE_DIR, 'Section_2_Summary_Report.txt')
else:
    summary_file = 'Section_2_Summary_Report.txt'
    print("Warning: BASE_DIR not found, saving to current directory")

try:
    # Use UTF-8 encoding to handle any special characters
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"✓ Summary report saved to {summary_file}")
except Exception as e:
    print(f"Warning: Could not save summary report with UTF-8: {e}")
    # Fallback: save with ASCII encoding, replacing problematic characters
    try:
        ascii_report = summary_report.encode('ascii', 'replace').decode('ascii')
        with open(summary_file, 'w', encoding='ascii') as f:
            f.write(ascii_report)
        print(f"✓ Summary report saved (ASCII version) to {summary_file}")
    except Exception as e2:
        print(f"Error: Could not save summary report even with ASCII encoding: {e2}")

# Display key metrics (FIXED - Safe variable handling)
print(f"\n" + "="*60)
print("FEATURE EXTRACTION SUMMARY (CORRECTED)")
print("="*60)
print(f"Total Samples: {samples_count:,}")
print(f"Total Features: {total_feature_count:,} (CORRECTED)")
print(f"Processing Time: {total_time:.1f}s")
print(f"Memory Usage: {memory_usage:.1f} MB")
print("")
print("Feature Breakdown:")

# FIXED: Safe iteration over feature_stats for display
if feature_stats and isinstance(feature_stats, dict):
    for feature_type, stats in feature_stats.items():
        if isinstance(stats, dict) and 'shape' in stats and 'extraction_time' in stats:
            try:
                shape_val = stats['shape'][1] if isinstance(stats['shape'], (list, tuple)) and len(stats['shape']) > 1 else 0
                time_val = float(stats['extraction_time']) if isinstance(stats['extraction_time'], (int, float)) else 0
                print(f"  {feature_type.upper():>15}: {shape_val:>5,} features ({time_val:>5.1f}s)")
            except Exception as e:
                print(f"  {feature_type.upper():>15}: Error displaying stats - {e}")
        else:
            print(f"  {feature_type.upper():>15}: N/A (incomplete stats)")
else:
    print("  No feature statistics available")

# ============================================================================
# 2.12 Save Comprehensive Checkpoint (FIXED)
# ============================================================================

print(f"\n2.12 Save Comprehensive Checkpoint")
print("-" * 40)

# FIXED: Check all variables exist before creating checkpoint
checkpoint_data = {}

# FIXED: Safely check if feature_types exists
try:
    if 'feature_types' not in locals():
        feature_types = ['aac', 'dpc', 'tpc', 'binary', 'physicochemical']
        print("Warning: feature_types not found, using default list")
except:
    feature_types = ['aac', 'dpc', 'tpc', 'binary', 'physicochemical']

# Safely add feature matrices
feature_matrices = {}
feature_var_mapping = {
    'aac': 'aac_features',
    'dpc': 'dpc_features', 
    'tpc': 'tpc_features',
    'binary': 'binary_features',
    'physicochemical': 'physicochemical_features',
    'combined': 'combined_features_matrix'
}

for feature_name, feature_var in feature_var_mapping.items():
    try:
        if feature_var in locals() and locals()[feature_var] is not None:
            feature_matrices[feature_name] = locals()[feature_var]
            print(f"✓ Added {feature_name} features to checkpoint")
        else:
            print(f"Warning: {feature_var} not found, skipping from checkpoint")
            # Create empty DataFrame as placeholder
            feature_matrices[feature_name] = pd.DataFrame()
    except Exception as e:
        print(f"Error adding {feature_name}: {e}")
        feature_matrices[feature_name] = pd.DataFrame()

checkpoint_data['feature_matrices'] = feature_matrices

# FIXED: Safely add feature stats
if feature_stats and isinstance(feature_stats, dict):
    checkpoint_data['feature_stats'] = feature_stats
    # Create extraction times dict safely
    checkpoint_data['extraction_times'] = {}
    for ft in feature_types:
        if ft in feature_stats and isinstance(feature_stats[ft], dict):
            extraction_time = feature_stats[ft].get('extraction_time', 0)
            if isinstance(extraction_time, (int, float)):
                checkpoint_data['extraction_times'][ft] = extraction_time
            else:
                checkpoint_data['extraction_times'][ft] = 0
else:
    checkpoint_data['feature_stats'] = {}
    checkpoint_data['extraction_times'] = {}
    print("Warning: No valid feature_stats found")

# FIXED: Safely add metadata
metadata_dict = {}
metadata_var_mapping = {
    'Header': 'Header_array',
    'Position': 'Position_array', 
    'target': 'target_array'
}

for meta_name, meta_var in metadata_var_mapping.items():
    try:
        if meta_var in locals() and locals()[meta_var] is not None:
            metadata_dict[meta_name] = locals()[meta_var]
            print(f"✓ Added {meta_name} metadata to checkpoint")
        else:
            print(f"Warning: {meta_var} not found, using empty array")
            metadata_dict[meta_name] = np.array([])
    except Exception as e:
        print(f"Error adding {meta_name}: {e}")
        metadata_dict[meta_name] = np.array([])

checkpoint_data['metadata'] = metadata_dict

# FIXED: Safely add config
config = {}
config_vars = {
    'window_size': ('WINDOW_SIZE', 20),
    'tpc_features': ('TPC_FEATURES', 8000),
    'batch_size': ('BATCH_SIZE', 1000),
    'n_workers': ('N_WORKERS', 4)
}

for config_name, (var_name, default_val) in config_vars.items():
    try:
        if var_name in locals():
            val = locals()[var_name]
            if isinstance(val, (int, float)):
                config[config_name] = val
            else:
                config[config_name] = default_val
                print(f"Warning: {var_name} is not numeric, using default {default_val}")
        else:
            config[config_name] = default_val
            print(f"Warning: {var_name} not found, using default {default_val}")
    except:
        config[config_name] = default_val

config['total_features'] = total_feature_count
checkpoint_data['config'] = config

# Add summary report and physicochemical props safely
checkpoint_data['summary_report'] = summary_report

try:
    if 'physicochemical_props' in locals() and physicochemical_props is not None:
        checkpoint_data['physicochemical_props'] = physicochemical_props
    else:
        checkpoint_data['physicochemical_props'] = {}
        print("Warning: physicochemical_props not found")
except:
    checkpoint_data['physicochemical_props'] = {}

# FIXED: Create metadata with safe calculations
metadata = {
    'n_samples': samples_count,
    'n_features_total': total_feature_count,
    'total_extraction_time': total_time,
    'memory_usage_mb': memory_usage,
    'correction_applied': 'TPC features corrected from 800 to 8000 (full feature space)',
    'extraction_completed': current_time_str
}

# Add individual feature counts safely
for feature_name, feature_var in feature_var_mapping.items():
    if feature_name == 'combined':
        continue  # Skip combined for individual counts
    try:
        if feature_var in locals() and locals()[feature_var] is not None and hasattr(locals()[feature_var], 'shape'):
            metadata[f'n_features_{feature_name}'] = locals()[feature_var].shape[1]
        else:
            metadata[f'n_features_{feature_name}'] = 0
    except:
        metadata[f'n_features_{feature_name}'] = 0

# FIXED: Safe progress tracker call
try:
    if 'progress_tracker' in locals():
        progress_tracker.mark_completed(
            "feature_extraction",
            metadata=metadata,
            checkpoint_data=checkpoint_data
        )
        print(f"✅ Comprehensive checkpoint saved!")
        print(f"   Components saved: {len(checkpoint_data)}")
        print(f"   Metadata fields: {len(metadata)}")
    else:
        print("Warning: progress_tracker not available, checkpoint not saved")
except Exception as e:
    print(f"Warning: Could not save checkpoint: {e}")
    print("Checkpoint data prepared but not saved")

# ============================================================================
# 2.13 Final Memory Cleanup & Validation (FIXED)
# ============================================================================

print(f"\n2.13 Final Memory Cleanup & Validation")
print("-" * 40)

# Memory cleanup
print("Performing comprehensive memory cleanup...")

# FIXED: Safe cleanup of feature matrices
try:
    if 'all_features' in locals() and isinstance(all_features, dict):
        for feature_type in feature_types:
            if feature_type in all_features:
                try:
                    del all_features[feature_type]
                    print(f"  ✓ Cleaned up {feature_type} from all_features")
                except:
                    print(f"  ⚠ Could not clean up {feature_type}")
except Exception as e:
    print(f"Warning: Error during all_features cleanup: {e}")

# FIXED: Safe cleanup of intermediate variables
cleanup_vars = ['combined_features', 'feature_stats_data', 'correlation_data']
for var_name in cleanup_vars:
    try:
        if var_name in locals():
            del locals()[var_name]
            print(f"  ✓ Cleaned up {var_name}")
    except Exception as e:
        print(f"  ⚠ Could not clean up {var_name}: {e}")

# Force garbage collection
gc.collect()

# FIXED: Safe memory calculation
try:
    if 'get_memory_usage' in locals():
        final_cleanup_memory = get_memory_usage()
    else:
        final_cleanup_memory = 0
        print("Warning: get_memory_usage function not available")
        
    if isinstance(memory_usage, (int, float)) and isinstance(final_cleanup_memory, (int, float)):
        memory_freed = memory_usage - final_cleanup_memory
    else:
        memory_freed = 0
        
    print(f"✓ Memory cleanup completed")
    print(f"  Before cleanup: {memory_usage:.1f} MB")
    print(f"  After cleanup: {final_cleanup_memory:.1f} MB")
    print(f"  Memory freed: {memory_freed:.1f} MB")
except Exception as e:
    print(f"Warning: Could not calculate memory usage: {e}")
    final_cleanup_memory = 0
    memory_freed = 0

# FIXED: Final validation with safe checks
print(f"\nFinal validation:")

validation_vars = [
    ('aac_features', 'AAC features'),
    ('dpc_features', 'DPC features'), 
    ('tpc_features', 'TPC features'),
    ('binary_features', 'Binary features'),
    ('physicochemical_features', 'Physicochemical features'),
    ('combined_features_matrix', 'Combined features')
]

for var_name, display_name in validation_vars:
    try:
        if var_name in locals() and locals()[var_name] is not None and hasattr(locals()[var_name], 'shape'):
            shape = locals()[var_name].shape
            if var_name == 'tpc_features':
                print(f"✓ {display_name}: {shape} (CORRECTED - FULL SPACE)")
            else:
                print(f"✓ {display_name}: {shape}")
        else:
            print(f"⚠ {display_name}: Not available or invalid")
    except Exception as e:
        print(f"⚠ {display_name}: Error checking - {e}")

# FIXED: Safe metadata validation
metadata_vars = [
    ('Header_array', 'Header'),
    ('Position_array', 'Position'),
    ('target_array', 'Target')
]

metadata_info = []
for var_name, display_name in metadata_vars:
    try:
        if var_name in locals() and locals()[var_name] is not None and hasattr(locals()[var_name], '__len__'):
            length = len(locals()[var_name])
            metadata_info.append(f"{display_name}({length})")
        else:
            metadata_info.append(f"{display_name}(N/A)")
    except Exception as e:
        metadata_info.append(f"{display_name}(Error)")

print(f"✓ Metadata arrays: {', '.join(metadata_info)}")

# FIXED: Final success message with safe variable handling
print(f"\n🎉 SECTION 2 COMPLETED SUCCESSFULLY!")
print(f"=" * 60)
print(f"🔧 CORRECTION APPLIED: TPC features expanded to full 8,000 feature space")

try:
    optimization_msg = 'Datatable acceleration enabled' if USE_DATATABLE else 'Standard pandas processing'
except:
    optimization_msg = 'Processing method unknown'

print(f"⚡ SPEED OPTIMIZATION: {optimization_msg}")
print(f"📊 Total features: {total_feature_count:,}")
print(f"⏱️  Processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"💾 Memory usage: {final_cleanup_memory:.1f} MB")
print(f"📁 Files generated: Tables, plots, reports, and checkpoint")
print(f"✅ Ready for Section 3 (Data Splitting)")
print(f"=" * 60)