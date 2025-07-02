# ============================================================================
# SECTION 1: DATA LOADING & EXPLORATION - FIXED VERSION
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: DATA LOADING & EXPLORATION")
print("="*80)

# Check if this section is already completed
if progress_tracker.is_completed("data_loading") and not FORCE_RETRAIN:
    print("Data loading already completed. Loading from checkpoint...")
    checkpoint_data = progress_tracker.load_checkpoint("data_loading")
    
    # Load all variables from checkpoint
    df_seq = checkpoint_data['df_seq']
    df_labels = checkpoint_data['df_labels']
    df_merged = checkpoint_data['df_merged']
    df_final = checkpoint_data['df_final']
    physicochemical_props = checkpoint_data['physicochemical_props']
    class_distribution = checkpoint_data['class_distribution']
    stats_dict = checkpoint_data['stats_dict']
    aa_distribution = checkpoint_data['aa_distribution']
    
    print(f"✓ Loaded dataset with {len(df_final)} samples")
    print(f"✓ Class distribution: {class_distribution.to_dict()}")
else:
    # ============================================================================
    # 1.1 Data Loading
    # ============================================================================
    
    print("\n1.1 Loading Data Files")
    print("-" * 40)
    
    # Memory tracking
    initial_memory = progress_tracker.get_memory_usage()
    print(f"Initial memory usage: {initial_memory['rss_mb']:.1f} MB")
    
    def load_sequences_optimized(file_path: str) -> pd.DataFrame:
        """Optimized FASTA sequence loader with validation"""
        logger.info(f"Loading sequences from {file_path}")
        
        sequences = []
        headers = []
        current_seq = ""
        current_header = ""
        
        with open(file_path, 'r') as file:
            for line in tqdm(file, desc="Reading sequences"):
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_header and current_seq:
                        # Apply sequence length filter
                        if len(current_seq) <= MAX_SEQUENCE_LENGTH:
                            headers.append(current_header)
                            sequences.append(current_seq)
                        else:
                            logger.warning(f"Filtered sequence {current_header}: length {len(current_seq)} > {MAX_SEQUENCE_LENGTH}")
                    
                    # Extract header ID (middle part between |) - ORIGINAL PATTERN
                    full_header = line[1:]
                    parts = full_header.split("|")
                    current_header = parts[1] if len(parts) > 1 else full_header
                    current_seq = ""
                else:
                    current_seq += line
        
        # Add the last sequence
        if current_header and current_seq:
            if len(current_seq) <= MAX_SEQUENCE_LENGTH:
                headers.append(current_header)
                sequences.append(current_seq)
        
        df = pd.DataFrame({
            'Header': headers,
            'Sequence': sequences
        })
        
        # Add sequence length for analysis
        df['SeqLength'] = df['Sequence'].str.len()
        
        logger.info(f"Loaded {len(df)} sequences (filtered: length <= {MAX_SEQUENCE_LENGTH})")
        return df
    
    def load_labels_robust(file_path: str) -> pd.DataFrame:
        """Robust label loader with error handling"""
        logger.info(f"Loading labels from {file_path}")
        
        try:
            # Try different Excel reading methods
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            raise
        
        # Validate required columns - CORRECT ORDER
        required_columns = ['UniProt ID', 'AA', 'Position']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Keep original column names for merging (don't rename to Header here)
        
        # Add target column (all positive samples)
        df['target'] = 1
        
        logger.info(f"Loaded {len(df)} phosphorylation sites")
        return df
    
    def load_physicochemical_properties(file_path: str) -> pd.DataFrame:
        """Load amino acid physicochemical properties"""
        logger.info(f"Loading physicochemical properties from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate amino acid coverage
        expected_aas = set('ACDEFGHIKLMNPQRSTVWY')
        available_aas = set(df['AA'].unique()) if 'AA' in df.columns else set()
        
        if not expected_aas.issubset(available_aas):
            missing_aas = expected_aas - available_aas
            logger.warning(f"Missing physicochemical data for amino acids: {missing_aas}")
        
        logger.info(f"Loaded physicochemical properties for {len(df)} amino acids")
        return df
    
    # Load all data files
    print("Loading protein sequences...")
    df_seq = load_sequences_optimized('data/Sequence_data.txt')
    
    print("Loading phosphorylation labels...")
    df_labels = load_labels_robust('data/labels.xlsx')
    
    print("Loading physicochemical properties...")
    physicochemical_props = load_physicochemical_properties('data/physiochemical_property.csv')
    
    # Memory check after loading
    post_load_memory = progress_tracker.get_memory_usage()
    print(f"Memory after loading: {post_load_memory['rss_mb']:.1f} MB (+{post_load_memory['rss_mb'] - initial_memory['rss_mb']:.1f} MB)")
    
    # ============================================================================
    # 1.2 Data Merging and Validation
    # ============================================================================
    
    print("\n1.2 Data Merging and Validation")
    print("-" * 40)
    
    # Merge sequences with labels - USE CORRECT COLUMN NAMES
    df_merged = pd.merge(
        df_seq,
        df_labels,
        left_on="Header",
        right_on="UniProt ID",
        how="inner"
    )
    df_merged["target"] = 1  # All these are positive examples
    
    print("Data merging statistics:")
    print(f"- Original sequences: {len(df_seq)}")
    print(f"- Original labels: {len(df_labels)}")
    print(f"- Successfully merged: {len(df_merged)}")
    print(f"- Proteins with phospho data: {df_merged['Header'].nunique()}")
    
    # Data validation
    print("\nData validation checks:")
    
    # Check for missing values
    missing_values = df_merged.isnull().sum().sum()
    print(f"- Missing values: {missing_values}")
    
    # Check amino acid validity at phosphorylation sites
    invalid_aas = ~df_merged['AA'].isin(['S', 'T', 'Y'])
    print(f"- Invalid amino acids at sites: {invalid_aas.sum()}")
    
    # Check position bounds
    position_errors = df_merged.apply(
        lambda x: x['Position'] > len(x['Sequence']) or x['Position'] < 1, axis=1
    ).sum()
    print(f"- Position out of bounds: {position_errors}")
    
    # Check for duplicates
    duplicates = df_merged.duplicated(['Header', 'Position']).sum()
    print(f"- Duplicate entries: {duplicates}")
    
    if missing_values > 0 or invalid_aas.sum() > 0 or position_errors > 0:
        logger.warning("Data quality issues detected - cleaning data")
        # Remove problematic entries
        df_merged = df_merged.dropna()
        df_merged = df_merged[df_merged['AA'].isin(['S', 'T', 'Y'])]
        df_merged = df_merged[
            (df_merged['Position'] >= 1) & 
            (df_merged['Position'] <= df_merged['Sequence'].str.len())
        ].copy()
        print(f"- Cleaned dataset size: {len(df_merged)}")
    
    # ============================================================================
    # 1.3 Data Exploration and Analysis
    # ============================================================================
    
    print("\n1.3 Data Exploration and Analysis")
    print("-" * 40)
    
    # Create plots directory
    plot_dir = os.path.join(BASE_DIR, 'plots', 'data_exploration')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Dataset statistics
    print("Generating dataset statistics...")
    
    # Amino acid distribution analysis
    aa_distribution = df_merged['AA'].value_counts()
    
    # Sequence length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_seq['SeqLength'].values, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('Number of Proteins')
    plt.title('Distribution of Protein Sequence Lengths')
    plt.axvline(df_seq['SeqLength'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_seq["SeqLength"].mean():.0f}')
    plt.axvline(df_seq['SeqLength'].median(), color='orange', linestyle='--', 
                label=f'Median: {df_seq["SeqLength"].median():.0f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Amino acid distribution at phosphorylation sites
    plt.figure(figsize=(8, 6))
    aa_distribution.plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.xlabel('Amino Acid')
    plt.ylabel('Count')
    plt.title('Amino Acid Distribution at Phosphorylation Sites')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'amino_acid_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Phosphorylation sites per protein
    sites_per_protein = df_merged.groupby('Header').size()
    plt.figure(figsize=(10, 6))
    plt.hist(sites_per_protein.values, bins=30, edgecolor='black', alpha=0.7, color='goldenrod')
    plt.xlabel('Number of Phosphorylation Sites')
    plt.ylabel('Number of Proteins')
    plt.title('Distribution of Phosphorylation Sites per Protein')
    plt.axvline(sites_per_protein.mean(), color='red', linestyle='--', 
                label=f'Mean: {sites_per_protein.mean():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, 'phosphorylation_site_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Data exploration visualizations saved!")
    
    # ============================================================================
    # 1.4 Balanced Negative Sample Generation - OPTIMIZED
    # ============================================================================
    
    print("\n1.4 Generating Balanced Negative Samples")
    print("-" * 40)
    
    def generate_negative_samples_optimized(df_merged: pd.DataFrame) -> pd.DataFrame:
        """Optimized negative sample generation with better memory management"""
        logger.info("Generating negative samples...")
        
        all_rows = []
        groups = list(df_merged.groupby('Header'))
        
        # Use multiprocessing for large datasets if available
        n_cores = min(mp.cpu_count(), 8)  # Limit to 8 cores max
        print(f"Processing {len(groups)} proteins using {n_cores} cores...")
        
        # Process in batches to manage memory
        batch_size = 100
        for i in tqdm(range(0, len(groups), batch_size), desc="Processing protein batches"):
            batch_groups = groups[i:i+batch_size]
            
            for header, group in batch_groups:
                seq = group['Sequence'].iloc[0]
                positive_positions = set(group['Position'].astype(int).tolist())
                
                # Find all S/T/Y positions more efficiently
                sty_positions = [pos+1 for pos, aa in enumerate(seq) if aa in "STY"]
                negative_candidates = [pos for pos in sty_positions if pos not in positive_positions]
                
                n_pos = len(positive_positions)
                sample_size = min(n_pos, len(negative_candidates))
                
                if sample_size > 0:
                    # Use consistent random seed for reproducibility
                    random.seed(RANDOM_SEED + hash(header) % 10000)
                    sampled_negatives = random.sample(negative_candidates, sample_size)
                    
                    # Keep all positives
                    all_rows.append(group.copy())
                    
                    # Add negatives more efficiently
                    if sampled_negatives:
                        negative_rows = []
                        base_row = group.iloc[0].copy()
                        
                        for neg_pos in sampled_negatives:
                            new_row = base_row.copy()
                            new_row['AA'] = seq[neg_pos - 1]
                            new_row['Position'] = neg_pos
                            new_row['target'] = 0
                            negative_rows.append(new_row)
                        
                        if negative_rows:
                            all_rows.append(pd.DataFrame(negative_rows))
            
            # Periodic garbage collection for large datasets
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        df_final = pd.concat(all_rows, ignore_index=True)
        logger.info(f"Generated dataset with {len(df_final)} rows (positives + negatives)")
        
        # Clean up memory
        del all_rows
        gc.collect()
        
        return df_final
    
    # Generate negative samples
    df_final = generate_negative_samples_optimized(df_merged)
    
    # Verify class balance
    class_distribution = df_final['target'].value_counts().sort_index()
    
    print("\nClass Distribution:")
    print(f"- Negative samples (0): {class_distribution.get(0, 0)}")
    print(f"- Positive samples (1): {class_distribution.get(1, 0)}")
    
    if len(class_distribution) == 2:
        balance_ratio = class_distribution.get(0, 0) / class_distribution.get(1, 0)
        print(f"- Balance ratio (neg/pos): {balance_ratio:.3f}")
        
        # Validate balance is close to 1:1
        if abs(balance_ratio - 1.0) > 0.1:
            logger.warning(f"Class imbalance detected: ratio = {balance_ratio:.3f}")
    
    # Class balance verification plot
    plt.figure(figsize=(8, 6))
    class_distribution.plot(kind='bar', color=['lightcoral', 'lightgreen'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution After Balancing')
    plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'], rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'class_balance_verification.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # 1.5 Generate Comprehensive Statistics - FIXED
    # ============================================================================
    
    print("\n1.5 Generating Comprehensive Statistics")
    print("-" * 40)
    
    # Create comprehensive statistics dictionary
    stats_dict = {
        'Total_Proteins': df_seq['Header'].nunique(),
        'Total_Phosphorylation_Sites': len(df_labels),
        'Proteins_With_Phospho_Data': df_merged['Header'].nunique(),
        'Average_Sites_Per_Protein': len(df_merged) / df_merged['Header'].nunique(),
        'Mean_Sequence_Length': df_seq['SeqLength'].mean(),
        'Median_Sequence_Length': df_seq['SeqLength'].median(),
        'Min_Sequence_Length': df_seq['SeqLength'].min(),
        'Max_Sequence_Length': df_seq['SeqLength'].max(),
        'Final_Positive_Samples': class_distribution.get(1, 0),
        'Final_Negative_Samples': class_distribution.get(0, 0),
        'Balance_Ratio': class_distribution.get(0, 0) / class_distribution.get(1, 0) if class_distribution.get(1, 0) > 0 else 0,
        'Total_Final_Samples': len(df_final),
        'S_Sites': aa_distribution.get('S', 0),
        'T_Sites': aa_distribution.get('T', 0),
        'Y_Sites': aa_distribution.get('Y', 0)
    }
    
    # ============================================================================
    # 1.6 Save Results and Tables
    # ============================================================================
    
    print("\n1.6 Saving Results and Tables")
    print("-" * 40)
    
    # Create tables directory
    tables_dir = os.path.join(BASE_DIR, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Save dataset statistics
    stats_df = pd.DataFrame([stats_dict]).T
    stats_df.columns = ['Value']
    stats_df.to_csv(os.path.join(tables_dir, 'dataset_statistics.csv'))
    
    # Save amino acid distribution
    aa_distribution.to_frame('Count').to_csv(
        os.path.join(tables_dir, 'amino_acid_distribution.csv')
    )
    
    # Save sequence length statistics
    seq_length_stats = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'Value': [
            df_seq['SeqLength'].count(),
            df_seq['SeqLength'].mean(),
            df_seq['SeqLength'].std(),
            df_seq['SeqLength'].min(),
            df_seq['SeqLength'].quantile(0.25),
            df_seq['SeqLength'].quantile(0.50),
            df_seq['SeqLength'].quantile(0.75),
            df_seq['SeqLength'].max()
        ]
    })
    seq_length_stats.to_csv(
        os.path.join(tables_dir, 'sequence_length_stats.csv'), 
        index=False
    )
    
    print("✓ All tables saved!")
    
    # ============================================================================
    # 1.7 Memory Cleanup and Optimization
    # ============================================================================
    
    print("\n1.7 Memory Cleanup and Optimization")
    print("-" * 40)
    
    # Remove unnecessary columns to save memory
    if 'SeqLength' in df_seq.columns:
        del df_seq['SeqLength']
    
    # Clean up intermediate variables that won't be needed
    # Keep df_merged for potential analysis in later sections
    
    # Force garbage collection
    gc.collect()
    
    # Memory check
    final_memory = progress_tracker.get_memory_usage()
    memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
    print(f"Final memory usage: {final_memory['rss_mb']:.1f} MB (+{memory_increase:.1f} MB)")
    
    # ============================================================================
    # 1.8 Save Checkpoint - FIXED WITH ALL VARIABLES
    # ============================================================================
    
    print("\n1.8 Saving Checkpoint")
    print("-" * 40)
    
    # CRITICAL FIX: Include ALL necessary variables in checkpoint
    checkpoint_data = {
        'df_seq': df_seq,
        'df_labels': df_labels,
        'df_merged': df_merged,
        'df_final': df_final,                      # ✅ CRITICAL - Main dataset
        'physicochemical_props': physicochemical_props,  # ✅ CRITICAL - For features
        'class_distribution': class_distribution,   # ✅ FIXED - Was missing
        'stats_dict': stats_dict,                  # ✅ FIXED - Was missing  
        'aa_distribution': aa_distribution         # ✅ ADDED - For analysis
    }
    
    # Validate checkpoint data before saving
    print("Validating checkpoint data...")
    
    # Data shape validation
    assert len(df_final) > 0, "df_final is empty"
    assert 'target' in df_final.columns, "target column missing"
    assert len(class_distribution) == 2, "class_distribution should have 2 classes"
    
    # Class balance validation  
    balance_ratio = class_distribution.get(0, 0) / class_distribution.get(1, 0)
    assert 0.5 <= balance_ratio <= 2.0, f"Extreme class imbalance: {balance_ratio}"
    
    print("✓ Checkpoint data validation passed")
    
    # Save with metadata
    progress_tracker.mark_completed(
        "data_loading",
        metadata={
            'total_proteins': df_seq['Header'].nunique(),
            'total_sites': len(df_labels),
            'final_samples': len(df_final),
            'positive_samples': class_distribution.get(1, 0),
            'negative_samples': class_distribution.get(0, 0),
            'balance_ratio': balance_ratio,
            'memory_usage_mb': final_memory['rss_mb']
        },
        checkpoint_data=checkpoint_data
    )
    
    print("✓ Section 1 checkpoint saved successfully!")

# ============================================================================
# 1.9 Final Data Quality Report
# ============================================================================

print("\n1.9 Final Data Quality Report")
print("-" * 40)

print("Data Quality Validation Results:")
print(f"✓ Total samples: {len(df_final):,}")
print(f"✓ Missing values: {df_final.isnull().sum().sum()}")
print(f"✓ Duplicate entries: {df_final.duplicated(['Header', 'Position']).sum()}")
print(f"✓ Invalid amino acids: {(~df_final['AA'].isin(['S', 'T', 'Y'])).sum()}")

# Position validation
position_errors = df_final.apply(
    lambda x: x['Position'] > len(x['Sequence']) or x['Position'] < 1, axis=1
).sum()
print(f"✓ Position errors: {position_errors}")

# Class distribution summary
print(f"\nClass Distribution Summary:")
for class_val, count in class_distribution.items():
    class_name = "Positive" if class_val == 1 else "Negative"
    percentage = (count / len(df_final)) * 100
    print(f"  {class_name} (class {class_val}): {count:,} samples ({percentage:.1f}%)")

print(f"\nDataset Summary:")
print(f"  Proteins: {df_final['Header'].nunique():,}")
print(f"  Average sites per protein: {len(df_final) / df_final['Header'].nunique():.1f}")
print(f"  Memory usage: {progress_tracker.get_memory_usage()['rss_mb']:.1f} MB")

# ============================================================================
# Summary Report
# ============================================================================

print("\n" + "="*80)
print("SECTION 1 SUMMARY - COMPLETED SUCCESSFULLY")
print("="*80)
print(f"✓ Loaded {len(df_seq)} protein sequences")
print(f"✓ Loaded {len(df_labels)} phosphorylation sites")
print(f"✓ Generated {class_distribution.get(0, 0)} negative samples")
print(f"✓ Final balanced dataset: {len(df_final):,} samples")
print(f"✓ Saved 4 visualizations and 3 data tables")
print(f"✓ All critical variables stored in checkpoint")
print("="*80)

# Display sample data
print("\nSample of final dataset:")
display(df_final.head(10))

print(f"\nDataset shape: {df_final.shape}")
print(f"Columns: {list(df_final.columns)}")

print("\n✅ Data loading and exploration completed successfully!")
print("✅ Ready for Section 2: Data Splitting Strategy")