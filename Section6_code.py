# ============================================================================
# SECTION 6: COMPREHENSIVE ERROR ANALYSIS - COMPLETE CODE (exp_3)
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: COMPREHENSIVE ERROR ANALYSIS (exp_3)")
print("="*80)

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 6.0 Setup and Configuration
# ============================================================================

# Ensure we're using exp_3 configuration
EXPERIMENT_NAME = "phosphorylation_prediction_exp_3"
BASE_DIR = "results/exp_3"
FORCE_RETRAIN = False

# Create directories for error analysis outputs
error_analysis_dirs = [
    os.path.join(BASE_DIR, 'tables', 'error_analysis'),
    os.path.join(BASE_DIR, 'plots', 'error_analysis'),
    os.path.join(BASE_DIR, 'reports', 'error_analysis')
]

for dir_path in error_analysis_dirs:
    os.makedirs(dir_path, exist_ok=True)

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Base Directory: {BASE_DIR}")
print(f"Force Retrain: {FORCE_RETRAIN}")

# ============================================================================
# 6.1 Load Required Data from Previous Sections
# ============================================================================

print("\n6.1 Loading Required Data from Previous Sections")
print("-" * 50)

# Check if error analysis is already completed
if progress_tracker.is_completed("error_analysis") and not FORCE_RETRAIN:
    print("Error analysis already completed. Loading from checkpoint...")
    error_checkpoint = progress_tracker.resume_from_checkpoint("error_analysis")
    if error_checkpoint:
        error_analysis_results = error_checkpoint['error_analysis_results']
        model_predictions = error_checkpoint['model_predictions']
        consensus_analysis = error_checkpoint['consensus_analysis']
        diversity_metrics = error_checkpoint['diversity_metrics']
        
        print("✓ Error analysis loaded from checkpoint!")
        
        # Skip to summary if already completed
        skip_to_summary = True
    else:
        skip_to_summary = False
else:
    skip_to_summary = False

if not skip_to_summary:
    # Load necessary data from previous sections (exp_3 specific)
    required_checkpoints = ['data_loading', 'data_splitting', 'ml_models_enhanced']
    loaded_data = {}

    print("Loading core checkpoints...")
    for checkpoint_name in required_checkpoints:
        try:
            checkpoint_data = progress_tracker.resume_from_checkpoint(checkpoint_name)
            if checkpoint_data:
                loaded_data[checkpoint_name] = checkpoint_data
                print(f"✓ Loaded {checkpoint_name} checkpoint")
                
                # Debug: show structure for ml_models_enhanced
                if checkpoint_name == 'ml_models_enhanced':
                    print(f"  Available keys: {list(checkpoint_data.keys())}")
            else:
                raise RuntimeError(f"{checkpoint_name} checkpoint not found")
        except Exception as e:
            print(f"❌ Error loading {checkpoint_name}: {e}")
            raise

    # Extract core variables
    df_final = loaded_data['data_loading']['df_final']
    test_indices = loaded_data['data_splitting']['test_indices']
    
    # Section 4 Enhanced uses different key names
    specialized_models = loaded_data['ml_models_enhanced']['specialized_models']
    final_test_results = loaded_data['ml_models_enhanced']['final_test_results']
    test_predictions_saved = loaded_data['ml_models_enhanced']['test_predictions']
    
    print("✓ Section 4 Enhanced checkpoint structure:")
    print(f"  - Specialized models: {list(specialized_models.keys())}")
    print(f"  - Test results: {list(final_test_results.keys())}")
    print(f"  - Test predictions: {list(test_predictions_saved.keys())}")

    # Get test data
    test_df = df_final.iloc[test_indices].copy()
    y_test = test_df['target'].values

    print(f"✓ Test dataset: {len(test_df)} samples")
    print(f"✓ Test labels - Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")

    # Load TabNet model (optional)
    tabnet_data = None
    try:
        tabnet_checkpoint = progress_tracker.resume_from_checkpoint("tabnet_model")
        if tabnet_checkpoint:
            tabnet_data = tabnet_checkpoint
            print("✓ Loaded TabNet model checkpoint")
    except:
        print("⚠️  TabNet model checkpoint not found (optional)")

    # Load transformer results from individual files
    print("\nLoading transformer results from individual files...")
    transformer_results = {}
    
    # Check if master results file exists
    master_results_path = os.path.join(BASE_DIR, 'transformers', 'master_results.csv')
    if os.path.exists(master_results_path):
        master_results_df = pd.read_csv(master_results_path)
        print(f"✓ Loaded master results: {len(master_results_df)} transformer models")
        
        # Debug: Show available columns
        print(f"  Available columns: {list(master_results_df.columns)}")
        
        # Check if we have the expected columns
        expected_cols = ['model_name', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
        missing_cols = [col for col in expected_cols if col not in master_results_df.columns]
        if missing_cols:
            print(f"  ⚠️  Missing expected columns: {missing_cols}")
        
        # Show first few rows for debugging
        if len(master_results_df) > 0:
            print(f"  Sample data:")
            print(f"    Model names: {list(master_results_df['model_name'].head())}")
            print(f"    Test F1 values: {list(master_results_df['test_f1'].head())}")
        
        # Load individual transformer predictions
        for _, row in master_results_df.iterrows():
            model_name = row['model_name']  # Correct column name
            
            # Extract base model name (remove timestamp)
            base_model_name = model_name.split('_')[0] + '_' + model_name.split('_')[1]  # e.g., transformer_v1
            model_dir = os.path.join(BASE_DIR, 'transformers', model_name)
            
            # Try to load test predictions
            pred_file = os.path.join(model_dir, 'predictions', 'test_predictions.csv')
            if os.path.exists(pred_file):
                try:
                    pred_df = pd.read_csv(pred_file)
                    
                    # Extract predictions and probabilities using correct column names from Section 5
                    if 'prediction_binary' in pred_df.columns and 'prediction_prob' in pred_df.columns:
                        predictions = pred_df['prediction_binary'].values.astype(int)
                        probabilities = pred_df['prediction_prob'].values.astype(float)
                    elif 'predictions' in pred_df.columns and 'probabilities' in pred_df.columns:
                        # Fallback for different naming convention
                        predictions = pred_df['predictions'].values
                        probabilities = pred_df['probabilities'].values
                        
                        # Convert to numeric if they're strings
                        predictions = pd.to_numeric(predictions, errors='coerce').astype(int)
                        probabilities = pd.to_numeric(probabilities, errors='coerce').astype(float)
                    else:
                        # Last fallback: assume first columns are what we need
                        print(f"    Available columns: {list(pred_df.columns)}")
                        # Look for columns that might contain predictions
                        prob_col = None
                        binary_col = None
                        
                        for col in pred_df.columns:
                            if 'prob' in col.lower() or 'probability' in col.lower():
                                prob_col = col
                            elif 'binary' in col.lower() or 'prediction' in col.lower():
                                binary_col = col
                        
                        if prob_col and binary_col:
                            predictions = pd.to_numeric(pred_df[binary_col], errors='coerce').astype(int)
                            probabilities = pd.to_numeric(pred_df[prob_col], errors='coerce').astype(float)
                        else:
                            print(f"    ⚠️  Could not identify prediction columns, skipping {model_name}")
                            continue
                    
                    transformer_results[base_model_name] = {
                        'test_predictions': predictions,
                        'test_probabilities': probabilities,
                        'test_metrics': {
                            'accuracy': pd.to_numeric(row['test_accuracy'], errors='coerce'),
                            'precision': pd.to_numeric(row['test_precision'], errors='coerce'),
                            'recall': pd.to_numeric(row['test_recall'], errors='coerce'),
                            'f1': pd.to_numeric(row['test_f1'], errors='coerce'),
                            'auc': pd.to_numeric(row['test_auc'], errors='coerce')
                        }
                    }
                    print(f"  ✓ {base_model_name}: Loaded predictions")
                except Exception as e:
                    print(f"  ⚠️  {model_name}: Error loading predictions - {e}")
                    # Print more details for debugging
                    if os.path.exists(pred_file):
                        try:
                            debug_df = pd.read_csv(pred_file)
                            print(f"    File exists, columns: {list(debug_df.columns)}")
                            print(f"    Shape: {debug_df.shape}")
                        except:
                            print(f"    File exists but couldn't read it")
                    else:
                        print(f"    File doesn't exist: {pred_file}")
            else:
                print(f"  ⚠️  {model_name}: Prediction file not found at {pred_file}")
                
                # Try to load from final_metrics.json as fallback
                metrics_file = os.path.join(model_dir, 'final_metrics.json')
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                        
                        # Check if test predictions are in the metrics file
                        if 'test_predictions' in metrics_data and 'test_probabilities' in metrics_data:
                            predictions = pd.to_numeric(metrics_data['test_predictions'], errors='coerce').astype(int)
                            probabilities = pd.to_numeric(metrics_data['test_probabilities'], errors='coerce').astype(float)
                            
                            transformer_results[base_model_name] = {
                                'test_predictions': predictions,
                                'test_probabilities': probabilities,
                                'test_metrics': {
                                    'accuracy': pd.to_numeric(row['test_accuracy'], errors='coerce'),
                                    'precision': pd.to_numeric(row['test_precision'], errors='coerce'),
                                    'recall': pd.to_numeric(row['test_recall'], errors='coerce'),
                                    'f1': pd.to_numeric(row['test_f1'], errors='coerce'),
                                    'auc': pd.to_numeric(row['test_auc'], errors='coerce')
                                }
                            }
                            print(f"  ✓ {base_model_name}: Loaded from metrics file")
                        else:
                            print(f"  ⚠️  {model_name}: No predictions found in metrics file")
                            print(f"    Available keys: {list(metrics_data.keys())}")
                    except Exception as e:
                        print(f"  ⚠️  {model_name}: Error loading metrics - {e}")
    else:
        print("⚠️  Master results file not found")

    print(f"✓ Loaded {len(transformer_results)} transformer models with predictions")

    # ============================================================================
    # 6.2 Collect Model Predictions
    # ============================================================================

    print("\n6.2 Collecting Model Predictions")
    print("-" * 40)

    model_predictions = {}

    # Collect ML model predictions
    print("Collecting ML model predictions from saved predictions...")
    
    # Use the saved test predictions from Section 4
    for feature_type, pred_data in test_predictions_saved.items():
        try:
            # pred_data should contain 'pred_proba' and 'pred_binary'
            if 'pred_proba' in pred_data and 'pred_binary' in pred_data:
                predictions = pred_data['pred_binary']
                probabilities = pred_data['pred_proba']
                
                model_predictions[f'ml_{feature_type}'] = {
                    'predictions': predictions.astype(int),
                    'probabilities': probabilities,
                    'model_type': 'ml',
                    'feature_type': feature_type
                }
                
                # Get performance metrics from final_test_results
                if feature_type in final_test_results:
                    metrics = final_test_results[feature_type]
                    acc = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1', 0)
                    auc = metrics.get('auc', 0)
                    print(f"  ✓ ML {feature_type}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                
            else:
                print(f"  ⚠️  ML {feature_type}: Missing prediction data")
                
        except Exception as e:
            print(f"  ❌ Error with ML {feature_type}: {e}")

    # Add ensemble and combined predictions if available
    if 'ensemble_test_predictions' in loaded_data['ml_models_enhanced']:
        ensemble_preds = loaded_data['ml_models_enhanced']['ensemble_test_predictions']
        if 'pred_proba' in ensemble_preds and 'pred_binary' in ensemble_preds:
            model_predictions['ml_ensemble'] = {
                'predictions': ensemble_preds['pred_binary'].astype(int),
                'probabilities': ensemble_preds['pred_proba'],
                'model_type': 'ml',
                'feature_type': 'ensemble'
            }
            
            if 'ensemble' in final_test_results:
                metrics = final_test_results['ensemble']
                print(f"  ✓ ML ensemble: Acc={metrics.get('accuracy', 0):.4f}, F1={metrics.get('f1', 0):.4f}, AUC={metrics.get('auc', 0):.4f}")

    if 'combined_test_predictions' in loaded_data['ml_models_enhanced']:
        combined_preds = loaded_data['ml_models_enhanced']['combined_test_predictions']
        if 'pred_proba' in combined_preds and 'pred_binary' in combined_preds:
            model_predictions['ml_combined'] = {
                'predictions': combined_preds['pred_binary'].astype(int),
                'probabilities': combined_preds['pred_proba'],
                'model_type': 'ml',
                'feature_type': 'combined'
            }
            
            if 'combined' in final_test_results:
                metrics = final_test_results['combined']
                print(f"  ✓ ML combined: Acc={metrics.get('accuracy', 0):.4f}, F1={metrics.get('f1', 0):.4f}, AUC={metrics.get('auc', 0):.4f}")

    # Add TabNet predictions (if available)
    if tabnet_data:
        try:
            print(f"  TabNet checkpoint keys: {list(tabnet_data.keys())}")
            
            # Check if TabNet predictions are already saved
            if 'test_predictions' in tabnet_data:
                tabnet_preds = tabnet_data['test_predictions']
                print(f"    test_predictions keys: {list(tabnet_preds.keys()) if isinstance(tabnet_preds, dict) else 'not a dict'}")
                
                if isinstance(tabnet_preds, dict) and 'pred_proba' in tabnet_preds and 'pred_binary' in tabnet_preds:
                    predictions = pd.to_numeric(tabnet_preds['pred_binary'], errors='coerce').astype(int)
                    probabilities = pd.to_numeric(tabnet_preds['pred_proba'], errors='coerce').astype(float)
                    
                    model_predictions['ml_tabnet'] = {
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'model_type': 'ml',
                        'feature_type': 'tabnet'
                    }
                    
                    # Get metrics if available
                    if 'tabnet_results' in tabnet_data:
                        results = tabnet_data['tabnet_results']
                        if 'test_metrics' in results:
                            metrics = results['test_metrics']
                            print(f"  ✓ TabNet: Acc={metrics.get('accuracy', 0):.4f}, F1={metrics.get('f1', 0):.4f}, AUC={metrics.get('auc', 0):.4f}")
                        else:
                            print("  ✓ TabNet: Predictions loaded (metrics not available)")
                    else:
                        print("  ✓ TabNet: Predictions loaded (results not available)")
                else:
                    print("  ⚠️  TabNet: Prediction data structure not recognized")
                    if isinstance(tabnet_preds, dict):
                        print(f"    Available keys: {list(tabnet_preds.keys())}")
            else:
                print("  ⚠️  TabNet: No test_predictions key found in checkpoint")
                
        except Exception as e:
            print(f"  ⚠️  TabNet prediction error: {e}")
            import traceback
            print(f"    Debug traceback: {traceback.format_exc()}")

    # Add transformer predictions
    print("Adding transformer predictions...")
    for model_name, results in transformer_results.items():
        try:
            if 'test_predictions' in results and 'test_probabilities' in results:
                predictions = results['test_predictions']
                probabilities = results['test_probabilities']
                
                # Ensure predictions are integers and probabilities are floats
                if predictions.dtype != int:
                    # Convert to numeric first, handling any string values
                    predictions = pd.to_numeric(predictions, errors='coerce')
                    # If still not binary, apply threshold
                    if predictions.max() > 1 or predictions.min() < 0:
                        predictions = (predictions > 0.5).astype(int)
                    else:
                        predictions = predictions.astype(int)
                
                # Ensure probabilities are floats
                if probabilities.dtype != float:
                    probabilities = pd.to_numeric(probabilities, errors='coerce').astype(float)
                
                model_predictions[f'transformer_{model_name}'] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'model_type': 'transformer',
                    'feature_type': model_name
                }
                
                metrics = results['test_metrics']
                print(f"  ✓ {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
            else:
                print(f"  ⚠️  {model_name}: No predictions available, only metrics stored")
                # Could still include in analysis if we have metrics but no predictions
                
        except Exception as e:
            print(f"  ❌ Error with transformer {model_name}: {e}")
            import traceback
            print(f"    Debug traceback: {traceback.format_exc()}")

    print(f"\n✓ Total models collected: {len(model_predictions)}")

    # ============================================================================
    # 6.3 Error Pattern Analysis
    # ============================================================================

    print("\n6.3 Error Pattern Analysis")
    print("-" * 40)

    error_analysis_results = {}
    
    for model_name, pred_data in model_predictions.items():
        predictions = pred_data['predictions']
        probabilities = pred_data['probabilities']
        
        # Calculate errors
        errors = predictions != y_test
        false_positives = (predictions == 1) & (y_test == 0)
        false_negatives = (predictions == 0) & (y_test == 1)
        
        # Get sequences for error analysis
        error_sequences = test_df[errors]['Sequence'].values
        error_positions = test_df[errors]['Position'].values
        
        fp_sequences = test_df[false_positives]['Sequence'].values
        fp_positions = test_df[false_positives]['Position'].values
        
        fn_sequences = test_df[false_negatives]['Sequence'].values
        fn_positions = test_df[false_negatives]['Position'].values
        
        # Extract amino acids at error positions
        fp_amino_acids = []
        fn_amino_acids = []
        
        for seq, pos in zip(fp_sequences, fp_positions):
            if 0 < pos <= len(seq):
                fp_amino_acids.append(seq[int(pos)-1])
        
        for seq, pos in zip(fn_sequences, fn_positions):
            if 0 < pos <= len(seq):
                fn_amino_acids.append(seq[int(pos)-1])
        
        # Analyze sequence windows around errors
        window_size = 5
        fp_windows = []
        fn_windows = []
        
        for seq, pos in zip(fp_sequences, fp_positions):
            pos_idx = int(pos) - 1
            start = max(0, pos_idx - window_size)
            end = min(len(seq), pos_idx + window_size + 1)
            window = seq[start:end]
            fp_windows.append(window)
        
        for seq, pos in zip(fn_sequences, fn_positions):
            pos_idx = int(pos) - 1
            start = max(0, pos_idx - window_size)
            end = min(len(seq), pos_idx + window_size + 1)
            window = seq[start:end]
            fn_windows.append(window)
        
        error_analysis_results[model_name] = {
            'total_errors': errors.sum(),
            'error_rate': errors.mean(),
            'false_positives': false_positives.sum(),
            'false_negatives': false_negatives.sum(),
            'fp_amino_acids': fp_amino_acids,
            'fn_amino_acids': fn_amino_acids,
            'fp_windows': fp_windows,
            'fn_windows': fn_windows,
            'fp_confidences': probabilities[false_positives],
            'fn_confidences': probabilities[false_negatives]
        }
        
        print(f"{model_name}: {errors.sum()} errors ({errors.mean():.3f} rate), FP: {false_positives.sum()}, FN: {false_negatives.sum()}")

    # ============================================================================
    # 6.4 Model Agreement Analysis
    # ============================================================================

    print("\n6.4 Model Agreement Analysis")
    print("-" * 40)

    # Create prediction and probability matrices
    model_names = list(model_predictions.keys())
    n_models = len(model_names)
    n_samples = len(y_test)
    
    prediction_matrix = np.zeros((n_samples, n_models), dtype=int)
    probability_matrix = np.zeros((n_samples, n_models))
    
    for i, model_name in enumerate(model_names):
        prediction_matrix[:, i] = model_predictions[model_name]['predictions']
        probability_matrix[:, i] = model_predictions[model_name]['probabilities']
    
    # Calculate consensus
    consensus_predictions = (prediction_matrix.mean(axis=1) > 0.5).astype(int)
    consensus_confidence = probability_matrix.mean(axis=1)
    
    # Analyze agreement patterns
    agreement_counts = prediction_matrix.sum(axis=1)
    unanimous_correct = (agreement_counts == n_models) & (consensus_predictions == y_test)
    unanimous_incorrect = (agreement_counts == 0) & (consensus_predictions != y_test)
    split_decisions = (agreement_counts > 0) & (agreement_counts < n_models)
    
    print(f"Unanimous correct predictions: {unanimous_correct.sum()} ({unanimous_correct.mean()*100:.1f}%)")
    print(f"Unanimous incorrect predictions: {unanimous_incorrect.sum()} ({unanimous_incorrect.mean()*100:.1f}%)")
    print(f"Split decisions: {split_decisions.sum()} ({split_decisions.mean()*100:.1f}%)")
    
    # Model-specific unique contributions
    model_unique_correct = {}
    
    for i, model_name in enumerate(model_names):
        predictions = prediction_matrix[:, i]
        
        # Cases where this model is correct but others are wrong
        model_correct = predictions == y_test
        others_wrong = (prediction_matrix.sum(axis=1) - predictions) < (n_models - 1)
        unique_correct = model_correct & others_wrong
        
        model_unique_correct[model_name] = unique_correct.sum()

    consensus_analysis = {
        'unanimous_correct': unanimous_correct,
        'unanimous_incorrect': unanimous_incorrect,
        'split_decisions': split_decisions,
        'consensus_predictions': consensus_predictions,
        'consensus_confidence': consensus_confidence,
        'model_unique_correct': model_unique_correct,
        'prediction_matrix': prediction_matrix,
        'probability_matrix': probability_matrix
    }

    # ============================================================================
    # 6.5 Diversity Metrics
    # ============================================================================

    print("\n6.5 Calculating Diversity Metrics")
    print("-" * 40)

    # Error correlation analysis
    error_matrix = (prediction_matrix != y_test[:, np.newaxis]).astype(int)
    error_correlations = np.corrcoef(error_matrix.T)
    
    # Calculate diversity metrics
    diversity_metrics = {}
    
    # Disagreement measure
    disagreement = 0
    n_pairs = 0
    for i in range(n_models):
        for j in range(i+1, n_models):
            disagreement += (error_matrix[:, i] != error_matrix[:, j]).mean()
            n_pairs += 1
    
    diversity_metrics['average_disagreement'] = disagreement / n_pairs
    
    # Q-statistic
    q_stats = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            n11 = ((error_matrix[:, i] == 1) & (error_matrix[:, j] == 1)).sum()
            n00 = ((error_matrix[:, i] == 0) & (error_matrix[:, j] == 0)).sum()
            n10 = ((error_matrix[:, i] == 1) & (error_matrix[:, j] == 0)).sum()
            n01 = ((error_matrix[:, i] == 0) & (error_matrix[:, j] == 1)).sum()
            
            if (n11*n00 + n10*n01) > 0:
                q = (n11*n00 - n10*n01) / (n11*n00 + n10*n01)
                q_stats.append(q)
    
    diversity_metrics['average_q_statistic'] = np.mean(q_stats)
    diversity_metrics['error_correlations'] = error_correlations
    
    print(f"Average disagreement: {diversity_metrics['average_disagreement']:.3f}")
    print(f"Average Q-statistic: {diversity_metrics['average_q_statistic']:.3f}")

    # ============================================================================
    # 6.6 Generate Visualizations
    # ============================================================================

    print("\n6.6 Generating Visualizations")
    print("-" * 40)

    plot_dir = os.path.join(BASE_DIR, 'plots', 'error_analysis')

    # 1. Error rate comparison
    plt.figure(figsize=(12, 6))
    error_rates = [error_analysis_results[model]['error_rate'] for model in model_names]
    plt.bar(range(len(model_names)), error_rates)
    plt.xticks(range(len(model_names)), [name.replace('_', '\n') for name in model_names], rotation=45, ha='right')
    plt.ylabel('Error Rate')
    plt.title('Error Rate Comparison Across Models')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_rate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 2. Consensus analysis pie chart
    plt.figure(figsize=(10, 8))
    consensus_data = [
        unanimous_correct.sum(),
        unanimous_incorrect.sum(),
        split_decisions.sum()
    ]
    labels = ['Unanimous Correct', 'Unanimous Incorrect', 'Split Decisions']
    colors = ['green', 'red', 'orange']
    
    plt.pie(consensus_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Model Agreement Patterns')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'consensus_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 3. Error correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(error_correlations,
                xticklabels=[name.replace('_', '\n') for name in model_names],
                yticklabels=[name.replace('_', '\n') for name in model_names],
                annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1)
    plt.title('Error Correlation Between Models')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 4. Confidence distribution for errors
    plt.figure(figsize=(15, 10))
    
    # Create subplots for FP and FN confidences
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # False Positive Confidences
    fp_confidences = []
    fp_labels = []
    for model_name in model_names:
        fp_conf = error_analysis_results[model_name]['fp_confidences']
        if len(fp_conf) > 0:
            fp_confidences.extend(fp_conf)
            fp_labels.extend([model_name.replace('_', '\n')] * len(fp_conf))
    
    if fp_confidences:
        axes[0, 0].hist(fp_confidences, bins=20, alpha=0.7, color='red')
        axes[0, 0].set_title('False Positive Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
    
    # False Negative Confidences
    fn_confidences = []
    fn_labels = []
    for model_name in model_names:
        fn_conf = error_analysis_results[model_name]['fn_confidences']
        if len(fn_conf) > 0:
            fn_confidences.extend(fn_conf)
            fn_labels.extend([model_name.replace('_', '\n')] * len(fn_conf))
    
    if fn_confidences:
        axes[0, 1].hist(fn_confidences, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('False Negative Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
    
    # Consensus confidence vs accuracy
    correct_predictions = consensus_predictions == y_test
    axes[1, 0].scatter(consensus_confidence[correct_predictions], 
                      np.ones(correct_predictions.sum()), 
                      alpha=0.5, color='green', label='Correct')
    axes[1, 0].scatter(consensus_confidence[~correct_predictions], 
                      np.zeros((~correct_predictions).sum()), 
                      alpha=0.5, color='red', label='Incorrect')
    axes[1, 0].set_xlabel('Consensus Confidence')
    axes[1, 0].set_ylabel('Prediction Correctness')
    axes[1, 0].set_title('Consensus Confidence vs Correctness')
    axes[1, 0].legend()
    
    # Model unique contributions
    unique_contributions = list(model_unique_correct.values())
    model_labels = [name.replace('_', '\n') for name in model_unique_correct.keys()]
    
    axes[1, 1].bar(range(len(unique_contributions)), unique_contributions)
    axes[1, 1].set_xticks(range(len(model_labels)))
    axes[1, 1].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Unique Correct Predictions')
    axes[1, 1].set_title('Model Unique Contributions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'comprehensive_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print("✓ All visualizations generated")

    # ============================================================================
    # 6.7 Save Results
    # ============================================================================

    print("\n6.7 Saving Analysis Results")
    print("-" * 40)

    # Save error analysis summary
    error_summary_data = []
    
    for model_name, results in error_analysis_results.items():
        fp_conf_mean = np.mean(results['fp_confidences']) if len(results['fp_confidences']) > 0 else 0
        fn_conf_mean = np.mean(results['fn_confidences']) if len(results['fn_confidences']) > 0 else 0
        
        error_summary_data.append({
            'Model': model_name,
            'Total_Errors': results['total_errors'],
            'Error_Rate': results['error_rate'],
            'False_Positives': results['false_positives'],
            'False_Negatives': results['false_negatives'],
            'FP_Mean_Confidence': fp_conf_mean,
            'FN_Mean_Confidence': fn_conf_mean
        })
    
    error_summary_df = pd.DataFrame(error_summary_data)
    error_summary_df.to_csv(os.path.join(BASE_DIR, 'tables', 'error_analysis', 'error_summary.csv'), index=False)
    print("✓ Saved error_summary.csv")

    # Save consensus analysis summary
    consensus_summary = {
        'Unanimous_Correct': unanimous_correct.sum(),
        'Unanimous_Incorrect': unanimous_incorrect.sum(),
        'Split_Decisions': split_decisions.sum(),
        'Consensus_Accuracy': (consensus_predictions == y_test).mean(),
        'Average_Disagreement': diversity_metrics['average_disagreement'],
        'Average_Q_Statistic': diversity_metrics['average_q_statistic']
    }
    
    consensus_df = pd.DataFrame([consensus_summary])
    consensus_df.to_csv(os.path.join(BASE_DIR, 'tables', 'error_analysis', 'consensus_summary.csv'), index=False)
    print("✓ Saved consensus_summary.csv")

    # Save model unique contributions
    unique_contributions_df = pd.DataFrame([
        {'Model': model, 'Unique_Correct_Predictions': count}
        for model, count in model_unique_correct.items()
    ])
    unique_contributions_df.to_csv(os.path.join(BASE_DIR, 'tables', 'error_analysis', 'model_unique_contributions.csv'), index=False)
    print("✓ Saved model_unique_contributions.csv")

    # Save detailed error patterns (pickle for complex data)
    detailed_patterns = {
        'fp_amino_acid_patterns': {},
        'fn_amino_acid_patterns': {},
        'fp_window_patterns': {},
        'fn_window_patterns': {}
    }
    
    for model_name, results in error_analysis_results.items():
        detailed_patterns['fp_amino_acid_patterns'][model_name] = Counter(results['fp_amino_acids'])
        detailed_patterns['fn_amino_acid_patterns'][model_name] = Counter(results['fn_amino_acids'])
        detailed_patterns['fp_window_patterns'][model_name] = Counter(results['fp_windows'])
        detailed_patterns['fn_window_patterns'][model_name] = Counter(results['fn_windows'])
    
    with open(os.path.join(BASE_DIR, 'tables', 'error_analysis', 'detailed_error_patterns.pkl'), 'wb') as f:
        pickle.dump(detailed_patterns, f)
    print("✓ Saved detailed_error_patterns.pkl")

    # ============================================================================
    # 6.8 Save Checkpoint
    # ============================================================================

    checkpoint_data = {
        'error_analysis_results': error_analysis_results,
        'model_predictions': model_predictions,
        'consensus_analysis': consensus_analysis,
        'diversity_metrics': diversity_metrics,
        'model_names': model_names,
        'test_performance': {
            'consensus_accuracy': (consensus_predictions == y_test).mean(),
            'unanimous_correct_count': unanimous_correct.sum(),
            'unanimous_incorrect_count': unanimous_incorrect.sum(),
            'split_decisions_count': split_decisions.sum()
        }
    }

    metadata = {
        'models_analyzed': len(model_predictions),
        'total_test_samples': len(y_test),
        'consensus_accuracy': float((consensus_predictions == y_test).mean()),
        'average_disagreement': float(diversity_metrics['average_disagreement']),
        'average_q_statistic': float(diversity_metrics['average_q_statistic']),
        'ml_models_count': sum(1 for name in model_names if name.startswith('ml_')),
        'transformer_models_count': sum(1 for name in model_names if name.startswith('transformer_')),
        'analysis_completed': True
    }

    progress_tracker.mark_completed(
        "error_analysis",
        metadata=metadata,
        checkpoint_data=checkpoint_data
    )

    print("✓ Error analysis checkpoint saved!")

# ============================================================================
# 6.9 Generate Comprehensive Report
# ============================================================================

print("\n6.9 Generating Comprehensive Report")
print("-" * 40)

# Create comprehensive text report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("COMPREHENSIVE ERROR ANALYSIS REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Experiment: {EXPERIMENT_NAME}")
report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

# Executive Summary
report_lines.append("EXECUTIVE SUMMARY")
report_lines.append("-" * 40)
report_lines.append(f"Total Models Analyzed: {len(model_predictions)}")
report_lines.append(f"Test Dataset Size: {len(y_test)} samples")
report_lines.append(f"Class Distribution - Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")
report_lines.append("")

if not skip_to_summary:
    report_lines.append("CONSENSUS PERFORMANCE")
    report_lines.append("-" * 40)
    report_lines.append(f"Consensus Accuracy: {(consensus_predictions == y_test).mean():.4f}")
    report_lines.append(f"Unanimous Correct: {unanimous_correct.sum()} ({unanimous_correct.mean()*100:.1f}%)")
    report_lines.append(f"Unanimous Incorrect: {unanimous_incorrect.sum()} ({unanimous_incorrect.mean()*100:.1f}%)")
    report_lines.append(f"Split Decisions: {split_decisions.sum()} ({split_decisions.mean()*100:.1f}%)")
    report_lines.append("")

    report_lines.append("MODEL DIVERSITY")
    report_lines.append("-" * 40)
    report_lines.append(f"Average Disagreement: {diversity_metrics['average_disagreement']:.3f}")
    report_lines.append(f"Average Q-statistic: {diversity_metrics['average_q_statistic']:.3f}")
    report_lines.append("")

    report_lines.append("INDIVIDUAL MODEL PERFORMANCE")
    report_lines.append("-" * 40)
    for model_name, results in error_analysis_results.items():
        report_lines.append(f"{model_name}:")
        report_lines.append(f"  Error Rate: {results['error_rate']:.4f}")
        report_lines.append(f"  False Positives: {results['false_positives']}")
        report_lines.append(f"  False Negatives: {results['false_negatives']}")
        
        if len(results['fp_confidences']) > 0:
            report_lines.append(f"  FP Avg Confidence: {np.mean(results['fp_confidences']):.4f}")
        if len(results['fn_confidences']) > 0:
            report_lines.append(f"  FN Avg Confidence: {np.mean(results['fn_confidences']):.4f}")
        report_lines.append("")

    report_lines.append("TOP MODEL UNIQUE CONTRIBUTIONS")
    report_lines.append("-" * 40)
    sorted_unique = sorted(model_unique_correct.items(), key=lambda x: x[1], reverse=True)
    for i, (model, count) in enumerate(sorted_unique[:5]):
        report_lines.append(f"{i+1}. {model}: {count} unique correct predictions")
    report_lines.append("")

report_lines.append("FILES GENERATED")
report_lines.append("-" * 40)
report_lines.append("Tables:")
report_lines.append("  - tables/error_analysis/error_summary.csv")
report_lines.append("  - tables/error_analysis/consensus_summary.csv")
report_lines.append("  - tables/error_analysis/model_unique_contributions.csv")
report_lines.append("  - tables/error_analysis/detailed_error_patterns.pkl")
report_lines.append("")
report_lines.append("Plots:")
report_lines.append("  - plots/error_analysis/error_rate_comparison.png")
report_lines.append("  - plots/error_analysis/consensus_analysis.png")
report_lines.append("  - plots/error_analysis/error_correlation_matrix.png")
report_lines.append("  - plots/error_analysis/comprehensive_error_analysis.png")
report_lines.append("")
report_lines.append("Checkpoints:")
report_lines.append("  - checkpoints/error_analysis.pkl")

# Save report
report_content = "\n".join(report_lines)
report_path = os.path.join(BASE_DIR, 'reports', 'error_analysis', 'comprehensive_error_analysis_report.txt')
with open(report_path, 'w') as f:
    f.write(report_content)

print(f"✓ Comprehensive report saved to: {report_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SECTION 6 SUMMARY")
print("="*80)

if not skip_to_summary:
    print(f"\nModels Analyzed: {len(model_predictions)}")
    print(f"- ML Models: {sum(1 for name in model_names if name.startswith('ml_'))}")
    print(f"- Transformer Models: {sum(1 for name in model_names if name.startswith('transformer_'))}")

    print(f"\nTest Dataset: {len(y_test)} samples")
    print(f"- Positive: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"- Negative: {len(y_test) - sum(y_test)} ({(len(y_test) - sum(y_test))/len(y_test)*100:.1f}%)")

    print("\nConsensus Performance:")
    print(f"- Consensus Accuracy: {(consensus_predictions == y_test).mean():.4f}")
    print(f"- Unanimous Correct: {unanimous_correct.sum()} ({unanimous_correct.mean()*100:.1f}%)")
    print(f"- Unanimous Incorrect: {unanimous_incorrect.sum()} ({unanimous_incorrect.mean()*100:.1f}%)")
    print(f"- Split Decisions: {split_decisions.sum()} ({split_decisions.mean()*100:.1f}%)")

    print("\nModel Diversity:")
    print(f"- Average Disagreement: {diversity_metrics['average_disagreement']:.3f}")
    print(f"- Average Q-statistic: {diversity_metrics['average_q_statistic']:.3f}")

    print("\nTop 3 Models with Most Unique Correct Predictions:")
    sorted_unique = sorted(model_unique_correct.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (model, count) in enumerate(sorted_unique):
        print(f"{i+1}. {model}: {count} unique correct predictions")

else:
    print("✓ Error analysis loaded from checkpoint")
    
    # Load checkpoint data to show summary
    error_checkpoint = progress_tracker.resume_from_checkpoint("error_analysis")
    if error_checkpoint:
        print(f"✓ Models analyzed: {error_checkpoint['metadata']['models_analyzed']}")
        print(f"✓ Test samples: {error_checkpoint['metadata']['total_test_samples']}")
        print(f"✓ Consensus accuracy: {error_checkpoint['metadata']['consensus_accuracy']:.4f}")
        print(f"✓ Average disagreement: {error_checkpoint['metadata']['average_disagreement']:.3f}")
        print(f"✓ ML models: {error_checkpoint['metadata']['ml_models_count']}")
        print(f"✓ Transformer models: {error_checkpoint['metadata']['transformer_models_count']}")
    print("✓ All results available in checkpoint data")

print("\n✅ SECTION 6: COMPREHENSIVE ERROR ANALYSIS COMPLETED!")
print("="*80)