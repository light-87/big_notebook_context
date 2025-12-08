# ============================================================================
# SECTION 7: SUPERVISOR FEEDBACK IMPLEMENTATION
# ============================================================================
# This section implements the supervisor's requested analyses using existing
# predictions without requiring model retraining.
#
# Tasks Covered:
# - A1: Per-residue (S/T/Y) metrics
# - A2: Calibration analysis (Brier score, reliability diagrams)
# - A3: Precision-Recall curves
# - A4: Threshold selection rationale
# - A5: 95% Confidence intervals for ALL metrics
# - C1: Export protein split lists
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: SUPERVISOR FEEDBACK IMPLEMENTATION")
print("="*80)

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 7.0 Configuration and Setup
# ============================================================================

EXPERIMENT_NAME = "phosphorylation_prediction_exp_3"
BASE_DIR = "results/exp_3"
RANDOM_SEED = 42
N_BOOTSTRAP = 1000  # For confidence intervals

# Create output directories
output_dirs = [
    os.path.join(BASE_DIR, 'tables', 'supervisor_feedback'),
    os.path.join(BASE_DIR, 'plots', 'supervisor_feedback'),
    os.path.join(BASE_DIR, 'reports', 'supervisor_feedback')
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Base Directory: {BASE_DIR}")
print(f"Bootstrap iterations: {N_BOOTSTRAP}")

# ============================================================================
# 7.1 Load Required Data
# ============================================================================

print("\n7.1 Loading Required Data")
print("-" * 50)

# Load from checkpoints
try:
    # Data loading checkpoint
    data_checkpoint = progress_tracker.resume_from_checkpoint("data_loading")
    df_final = data_checkpoint['df_final']
    print("Loaded df_final from data_loading checkpoint")

    # Data splitting checkpoint
    split_checkpoint = progress_tracker.resume_from_checkpoint("data_splitting")
    train_indices = split_checkpoint['train_indices']
    val_indices = split_checkpoint['val_indices']
    test_indices = split_checkpoint['test_indices']
    train_proteins_list = split_checkpoint['train_proteins_list']
    val_proteins_list = split_checkpoint['val_proteins_list']
    test_proteins_list = split_checkpoint['test_proteins_list']
    print("Loaded split indices from data_splitting checkpoint")

    # ML models checkpoint
    ml_checkpoint = progress_tracker.resume_from_checkpoint("ml_models_enhanced")
    ml_predictions = ml_checkpoint['test_predictions']
    print("Loaded ML predictions from ml_models_enhanced checkpoint")

except Exception as e:
    print(f"Error loading checkpoints: {e}")
    print("Please ensure Sections 1-4 have been run first.")
    raise

# Load transformer predictions
print("\nLoading transformer predictions...")
transformer_predictions = {}

master_results_path = os.path.join(BASE_DIR, 'transformers', 'master_results.csv')
if os.path.exists(master_results_path):
    master_df = pd.read_csv(master_results_path)

    for _, row in master_df.iterrows():
        model_name = row['model_name']
        base_name = '_'.join(model_name.split('_')[:2])  # e.g., transformer_v1

        pred_file = os.path.join(BASE_DIR, 'transformers', model_name, 'predictions', 'test_predictions.csv')
        if os.path.exists(pred_file):
            pred_df = pd.read_csv(pred_file)

            # Extract predictions and probabilities
            if 'prediction_binary' in pred_df.columns:
                transformer_predictions[base_name] = {
                    'predictions': pred_df['prediction_binary'].values.astype(int),
                    'probabilities': pred_df['prediction_prob'].values.astype(float)
                }
                print(f"  Loaded {base_name}: {len(pred_df)} predictions")
else:
    print("Warning: No transformer predictions found")

# Get test data
test_df = df_final.iloc[test_indices].copy()
y_test = test_df['target'].values
test_residue_types = test_df['AA'].values

print(f"\nTest set size: {len(test_df)}")
print(f"Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")
print(f"Residue distribution: S={sum(test_residue_types=='S')}, T={sum(test_residue_types=='T')}, Y={sum(test_residue_types=='Y')}")

# ============================================================================
# 7.2 Task A1: Per-Residue (S/T/Y) Metrics
# ============================================================================

print("\n" + "="*80)
print("TASK A1: Per-Residue (S/T/Y) Metrics")
print("="*80)

def compute_per_residue_metrics(y_true, y_pred, y_prob, residue_types, model_name):
    """Compute metrics separately for S, T, Y residues"""
    results = []

    for residue in ['S', 'T', 'Y']:
        mask = residue_types == residue
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        y_true_r = y_true[mask]
        y_pred_r = y_pred[mask]
        y_prob_r = y_prob[mask]

        # Handle edge cases
        n_pos = sum(y_true_r)
        n_neg = len(y_true_r) - n_pos

        result = {
            'model': model_name,
            'residue': residue,
            'n_samples': n_samples,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'prevalence': n_pos / n_samples if n_samples > 0 else 0
        }

        # Compute metrics (with error handling)
        try:
            result['precision'] = precision_score(y_true_r, y_pred_r, zero_division=0)
            result['recall'] = recall_score(y_true_r, y_pred_r, zero_division=0)
            result['f1'] = f1_score(y_true_r, y_pred_r, zero_division=0)
            result['accuracy'] = accuracy_score(y_true_r, y_pred_r)

            if len(np.unique(y_true_r)) > 1:
                result['auc'] = roc_auc_score(y_true_r, y_prob_r)
            else:
                result['auc'] = np.nan

            result['mcc'] = matthews_corrcoef(y_true_r, y_pred_r)
        except Exception as e:
            print(f"Warning: Error computing metrics for {model_name}/{residue}: {e}")
            result['precision'] = result['recall'] = result['f1'] = np.nan
            result['accuracy'] = result['auc'] = result['mcc'] = np.nan

        results.append(result)

    return results

# Collect all model predictions
all_predictions = {}

# Add transformer predictions
for model_name, pred_data in transformer_predictions.items():
    all_predictions[model_name] = pred_data

# Add ML predictions
for feature_type, pred_data in ml_predictions.items():
    all_predictions[f'ml_{feature_type}'] = {
        'predictions': pred_data['predictions'],
        'probabilities': pred_data.get('probabilities', pred_data['predictions'].astype(float))
    }

# Compute per-residue metrics for all models
per_residue_results = []

for model_name, pred_data in all_predictions.items():
    y_pred = pred_data['predictions']
    y_prob = pred_data['probabilities']

    results = compute_per_residue_metrics(y_test, y_pred, y_prob, test_residue_types, model_name)
    per_residue_results.extend(results)

# Create results DataFrame
per_residue_df = pd.DataFrame(per_residue_results)

# Display results
print("\nPer-Residue Metrics Summary:")
print("-" * 80)

# Pivot for better display
for model_name in per_residue_df['model'].unique():
    model_data = per_residue_df[per_residue_df['model'] == model_name]
    print(f"\n{model_name}:")
    display_cols = ['residue', 'n_samples', 'precision', 'recall', 'f1', 'auc']
    print(model_data[display_cols].to_string(index=False))

# Save to CSV
per_residue_df.to_csv(
    os.path.join(BASE_DIR, 'tables', 'supervisor_feedback', 'per_residue_metrics.csv'),
    index=False
)
print(f"\nSaved per-residue metrics to tables/supervisor_feedback/per_residue_metrics.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# F1 by residue type
ax1 = axes[0, 0]
pivot_f1 = per_residue_df.pivot(index='model', columns='residue', values='f1')
pivot_f1.plot(kind='bar', ax=ax1, color=['#2ecc71', '#3498db', '#e74c3c'])
ax1.set_title('F1 Score by Residue Type')
ax1.set_xlabel('Model')
ax1.set_ylabel('F1 Score')
ax1.legend(title='Residue')
ax1.tick_params(axis='x', rotation=45)

# AUC by residue type
ax2 = axes[0, 1]
pivot_auc = per_residue_df.pivot(index='model', columns='residue', values='auc')
pivot_auc.plot(kind='bar', ax=ax2, color=['#2ecc71', '#3498db', '#e74c3c'])
ax2.set_title('AUC by Residue Type')
ax2.set_xlabel('Model')
ax2.set_ylabel('AUC')
ax2.legend(title='Residue')
ax2.tick_params(axis='x', rotation=45)

# Sample distribution
ax3 = axes[1, 0]
sample_counts = per_residue_df.groupby('residue')[['n_positive', 'n_negative']].first()
sample_counts.plot(kind='bar', ax=ax3, color=['#27ae60', '#c0392b'])
ax3.set_title('Sample Distribution by Residue Type')
ax3.set_xlabel('Residue')
ax3.set_ylabel('Count')
ax3.legend(['Positive', 'Negative'])
ax3.tick_params(axis='x', rotation=0)

# Performance gap (best model per residue)
ax4 = axes[1, 1]
best_f1 = per_residue_df.groupby('residue')['f1'].max()
best_f1.plot(kind='bar', ax=ax4, color=['#2ecc71', '#3498db', '#e74c3c'])
ax4.set_title('Best F1 Score per Residue Type')
ax4.set_xlabel('Residue')
ax4.set_ylabel('F1 Score')
ax4.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, 'plots', 'supervisor_feedback', 'per_residue_metrics.png'),
    dpi=300, bbox_inches='tight'
)
plt.close()

print("Saved per-residue visualization to plots/supervisor_feedback/per_residue_metrics.png")

# ============================================================================
# 7.3 Task A2: Calibration Analysis
# ============================================================================

print("\n" + "="*80)
print("TASK A2: Calibration Analysis")
print("="*80)

def calibration_analysis(y_true, y_prob, model_name, n_bins=10):
    """Compute calibration metrics and curve data"""
    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts[bin_counts > 0] / len(y_true)))

    return {
        'model': model_name,
        'brier_score': brier,
        'ece': ece,
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }

# Compute calibration for all models
calibration_results = []
calibration_curves = {}

for model_name, pred_data in all_predictions.items():
    y_prob = pred_data['probabilities']

    result = calibration_analysis(y_test, y_prob, model_name)
    calibration_results.append({
        'model': result['model'],
        'brier_score': result['brier_score'],
        'ece': result['ece']
    })
    calibration_curves[model_name] = {
        'prob_true': result['prob_true'],
        'prob_pred': result['prob_pred']
    }

calibration_df = pd.DataFrame(calibration_results)
calibration_df = calibration_df.sort_values('brier_score')

print("\nCalibration Metrics Summary:")
print("-" * 60)
print(calibration_df.to_string(index=False))

# Save calibration scores
calibration_df.to_csv(
    os.path.join(BASE_DIR, 'tables', 'supervisor_feedback', 'calibration_scores.csv'),
    index=False
)

# Create reliability diagrams
n_models = len(calibration_curves)
n_cols = min(3, n_models)
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
if n_rows == 1 and n_cols == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, (model_name, curve_data) in enumerate(calibration_curves.items()):
    ax = axes[idx]

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Model calibration curve
    brier = calibration_df[calibration_df['model'] == model_name]['brier_score'].values[0]
    ax.plot(curve_data['prob_pred'], curve_data['prob_true'], 's-',
            label=f'Brier={brier:.4f}', linewidth=2, markersize=8)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Reliability Diagram: {model_name}')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

# Hide unused axes
for idx in range(len(calibration_curves), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, 'plots', 'supervisor_feedback', 'reliability_diagrams.png'),
    dpi=300, bbox_inches='tight'
)
plt.close()

# Combined reliability diagram
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

colors = plt.cm.Set1(np.linspace(0, 1, len(calibration_curves)))
for (model_name, curve_data), color in zip(calibration_curves.items(), colors):
    brier = calibration_df[calibration_df['model'] == model_name]['brier_score'].values[0]
    plt.plot(curve_data['prob_pred'], curve_data['prob_true'], 's-',
             label=f'{model_name} (Brier={brier:.4f})', color=color, linewidth=2, markersize=6)

plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title('Reliability Diagrams - All Models', fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, 'plots', 'supervisor_feedback', 'reliability_diagrams_combined.png'),
    dpi=300, bbox_inches='tight'
)
plt.close()

print("\nSaved calibration analysis to:")
print("  - tables/supervisor_feedback/calibration_scores.csv")
print("  - plots/supervisor_feedback/reliability_diagrams.png")
print("  - plots/supervisor_feedback/reliability_diagrams_combined.png")

# ============================================================================
# 7.4 Task A3: Precision-Recall Curves
# ============================================================================

print("\n" + "="*80)
print("TASK A3: Precision-Recall Curves")
print("="*80)

# Compute PR curves and average precision
pr_results = []

plt.figure(figsize=(12, 8))

colors = plt.cm.Set1(np.linspace(0, 1, len(all_predictions)))
for (model_name, pred_data), color in zip(all_predictions.items(), colors):
    y_prob = pred_data['probabilities']

    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plt.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})',
             color=color, linewidth=2)

    pr_results.append({
        'model': model_name,
        'average_precision': ap
    })

# Add baseline (no skill classifier)
baseline = sum(y_test) / len(y_test)
plt.axhline(y=baseline, color='gray', linestyle='--', label=f'No skill (P={baseline:.3f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14)
plt.legend(loc='lower left', fontsize=9)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, 'plots', 'supervisor_feedback', 'pr_curves.png'),
    dpi=300, bbox_inches='tight'
)
plt.close()

pr_df = pd.DataFrame(pr_results).sort_values('average_precision', ascending=False)
print("\nAverage Precision Scores:")
print("-" * 40)
print(pr_df.to_string(index=False))

pr_df.to_csv(
    os.path.join(BASE_DIR, 'tables', 'supervisor_feedback', 'average_precision_scores.csv'),
    index=False
)

print("\nSaved PR curves to plots/supervisor_feedback/pr_curves.png")

# ============================================================================
# 7.5 Task A4: Threshold Selection Rationale
# ============================================================================

print("\n" + "="*80)
print("TASK A4: Threshold Selection Analysis")
print("="*80)

def threshold_analysis(y_true, y_prob, model_name, thresholds=np.arange(0.1, 1.0, 0.05)):
    """Analyze performance at different thresholds"""
    results = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        # Skip if all predictions are same class
        if len(np.unique(y_pred)) < 2:
            continue

        results.append({
            'model': model_name,
            'threshold': thresh,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'n_positive_pred': y_pred.sum(),
            'n_negative_pred': (1 - y_pred).sum()
        })

    return pd.DataFrame(results)

# Analyze best models
best_models = ['transformer_v1', 'transformer_v2', 'ml_physicochemical']
available_best = [m for m in best_models if m in all_predictions]

threshold_results = []
for model_name in available_best:
    y_prob = all_predictions[model_name]['probabilities']
    result = threshold_analysis(y_test, y_prob, model_name)
    threshold_results.append(result)

if threshold_results:
    threshold_df = pd.concat(threshold_results, ignore_index=True)

    # Find optimal thresholds
    optimal_thresholds = []
    for model_name in available_best:
        model_data = threshold_df[threshold_df['model'] == model_name]
        if len(model_data) > 0:
            optimal_idx = model_data['f1'].idxmax()
            optimal_row = model_data.loc[optimal_idx]
            optimal_thresholds.append({
                'model': model_name,
                'optimal_threshold': optimal_row['threshold'],
                'optimal_f1': optimal_row['f1'],
                'f1_at_0.5': model_data[model_data['threshold'] == 0.5]['f1'].values[0] if 0.5 in model_data['threshold'].values else np.nan,
                'improvement': optimal_row['f1'] - (model_data[model_data['threshold'] == 0.5]['f1'].values[0] if 0.5 in model_data['threshold'].values else optimal_row['f1'])
            })

    optimal_df = pd.DataFrame(optimal_thresholds)

    print("\nOptimal Threshold Analysis:")
    print("-" * 60)
    print(optimal_df.to_string(index=False))

    print("\nThreshold Selection Rationale:")
    print("-" * 60)
    print("1. Default threshold of 0.5 is standard for balanced datasets")
    print("2. Our dataset is balanced (50/50), supporting 0.5 as appropriate")

    for _, row in optimal_df.iterrows():
        if abs(row['improvement']) < 0.01:
            print(f"3. {row['model']}: 0.5 is near-optimal (improvement < 1%)")
        else:
            print(f"3. {row['model']}: Optimal at {row['optimal_threshold']:.2f} (+{row['improvement']*100:.1f}% F1)")

    # Save results
    threshold_df.to_csv(
        os.path.join(BASE_DIR, 'tables', 'supervisor_feedback', 'threshold_analysis.csv'),
        index=False
    )
    optimal_df.to_csv(
        os.path.join(BASE_DIR, 'tables', 'supervisor_feedback', 'optimal_thresholds.csv'),
        index=False
    )

    # Visualization
    fig, axes = plt.subplots(1, len(available_best), figsize=(5*len(available_best), 4))
    if len(available_best) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, available_best):
        model_data = threshold_df[threshold_df['model'] == model_name]

        ax.plot(model_data['threshold'], model_data['precision'], 'b-', label='Precision', linewidth=2)
        ax.plot(model_data['threshold'], model_data['recall'], 'g-', label='Recall', linewidth=2)
        ax.plot(model_data['threshold'], model_data['f1'], 'r-', label='F1', linewidth=2)

        # Mark 0.5 threshold
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')

        # Mark optimal
        optimal = optimal_df[optimal_df['model'] == model_name]['optimal_threshold'].values[0]
        ax.axvline(x=optimal, color='orange', linestyle=':', alpha=0.7, label=f'Optimal ({optimal:.2f})')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name}')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, 'plots', 'supervisor_feedback', 'threshold_analysis.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    print("\nSaved threshold analysis to:")
    print("  - tables/supervisor_feedback/threshold_analysis.csv")
    print("  - tables/supervisor_feedback/optimal_thresholds.csv")
    print("  - plots/supervisor_feedback/threshold_analysis.png")

# ============================================================================
# 7.6 Task A5: 95% Confidence Intervals for ALL Metrics
# ============================================================================

print("\n" + "="*80)
print("TASK A5: 95% Confidence Intervals")
print("="*80)

def bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute 95% CIs for all primary metrics using bootstrap"""
    np.random.seed(seed)
    n_samples = len(y_true)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'mcc': [],
        'brier': []
    }

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        y_pr = y_prob[indices]

        # Skip if only one class in sample
        if len(np.unique(y_t)) < 2:
            continue

        try:
            metrics['accuracy'].append(accuracy_score(y_t, y_p))
            metrics['precision'].append(precision_score(y_t, y_p, zero_division=0))
            metrics['recall'].append(recall_score(y_t, y_p, zero_division=0))
            metrics['f1'].append(f1_score(y_t, y_p, zero_division=0))
            metrics['auc'].append(roc_auc_score(y_t, y_pr))
            metrics['mcc'].append(matthews_corrcoef(y_t, y_p))
            metrics['brier'].append(brier_score_loss(y_t, y_pr))
        except Exception:
            continue

    # Compute CIs
    alpha = 1 - ci
    results = {}
    for metric, values in metrics.items():
        if len(values) > 0:
            values = np.array(values)
            lower = np.percentile(values, alpha/2 * 100)
            upper = np.percentile(values, (1 - alpha/2) * 100)
            mean = np.mean(values)
            std = np.std(values)
            results[metric] = {
                'mean': mean,
                'std': std,
                'lower_95ci': lower,
                'upper_95ci': upper,
                'ci_width': upper - lower
            }

    return results

# Compute CIs for all models
print(f"\nComputing {N_BOOTSTRAP} bootstrap iterations for each model...")
print("This may take a few minutes...")

ci_results = []

for model_name, pred_data in all_predictions.items():
    print(f"  Processing {model_name}...", end=" ", flush=True)

    y_pred = pred_data['predictions']
    y_prob = pred_data['probabilities']

    ci_data = bootstrap_confidence_intervals(y_test, y_pred, y_prob, n_bootstrap=N_BOOTSTRAP)

    for metric, values in ci_data.items():
        ci_results.append({
            'model': model_name,
            'metric': metric,
            'point_estimate': values['mean'],
            'std': values['std'],
            'lower_95ci': values['lower_95ci'],
            'upper_95ci': values['upper_95ci'],
            'ci_width': values['ci_width']
        })

    print("Done")

ci_df = pd.DataFrame(ci_results)

# Display results in a nice format
print("\n95% Confidence Intervals Summary:")
print("-" * 100)

# Pivot for better display
for model_name in ci_df['model'].unique():
    model_data = ci_df[ci_df['model'] == model_name]
    print(f"\n{model_name}:")

    for _, row in model_data.iterrows():
        metric = row['metric']
        point = row['point_estimate']
        lower = row['lower_95ci']
        upper = row['upper_95ci']
        print(f"  {metric:12s}: {point:.4f} [{lower:.4f}, {upper:.4f}]")

# Save to CSV
ci_df.to_csv(
    os.path.join(BASE_DIR, 'tables', 'supervisor_feedback', 'confidence_intervals.csv'),
    index=False
)

# Create formatted table for paper
paper_format = []
for model_name in ci_df['model'].unique():
    model_data = ci_df[ci_df['model'] == model_name]
    row = {'Model': model_name}

    for _, m in model_data.iterrows():
        metric = m['metric']
        row[f'{metric}_point'] = f"{m['point_estimate']:.3f}"
        row[f'{metric}_ci'] = f"[{m['lower_95ci']:.3f}, {m['upper_95ci']:.3f}]"

    paper_format.append(row)

paper_df = pd.DataFrame(paper_format)
paper_df.to_csv(
    os.path.join(BASE_DIR, 'tables', 'supervisor_feedback', 'confidence_intervals_paper_format.csv'),
    index=False
)

# Visualization: CI plot for F1 scores
f1_data = ci_df[ci_df['metric'] == 'f1'].sort_values('point_estimate', ascending=True)

plt.figure(figsize=(10, 6))
y_pos = np.arange(len(f1_data))

plt.barh(y_pos, f1_data['point_estimate'], xerr=[
    f1_data['point_estimate'] - f1_data['lower_95ci'],
    f1_data['upper_95ci'] - f1_data['point_estimate']
], capsize=5, color='steelblue', alpha=0.7)

plt.yticks(y_pos, f1_data['model'])
plt.xlabel('F1 Score')
plt.title('F1 Scores with 95% Confidence Intervals')
plt.xlim([0.6, 0.9])
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, 'plots', 'supervisor_feedback', 'f1_confidence_intervals.png'),
    dpi=300, bbox_inches='tight'
)
plt.close()

print("\nSaved confidence intervals to:")
print("  - tables/supervisor_feedback/confidence_intervals.csv")
print("  - tables/supervisor_feedback/confidence_intervals_paper_format.csv")
print("  - plots/supervisor_feedback/f1_confidence_intervals.png")

# ============================================================================
# 7.7 Task C1: Export Protein Split Lists
# ============================================================================

print("\n" + "="*80)
print("TASK C1: Export Protein Split Lists")
print("="*80)

# Ensure data directory exists
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Create comprehensive JSON with all splits
splits = {
    'train': list(train_proteins_list) if hasattr(train_proteins_list, '__iter__') else [train_proteins_list],
    'validation': list(val_proteins_list) if hasattr(val_proteins_list, '__iter__') else [val_proteins_list],
    'test': list(test_proteins_list) if hasattr(test_proteins_list, '__iter__') else [test_proteins_list]
}

# Add metadata
splits['metadata'] = {
    'total_proteins': len(splits['train']) + len(splits['validation']) + len(splits['test']),
    'train_count': len(splits['train']),
    'val_count': len(splits['validation']),
    'test_count': len(splits['test']),
    'split_ratio': '70/15/15',
    'random_seed': 42,
    'splitting_strategy': 'protein-based (no leakage)',
    'experiment': EXPERIMENT_NAME
}

# Sample counts per split
for split_name, protein_list in [('train', splits['train']), ('validation', splits['validation']), ('test', splits['test'])]:
    if split_name == 'train':
        samples = len(train_indices)
    elif split_name == 'validation':
        samples = len(val_indices)
    else:
        samples = len(test_indices)
    splits['metadata'][f'{split_name}_samples'] = samples

# Save as JSON
with open(os.path.join(data_dir, 'protein_splits.json'), 'w') as f:
    json.dump(splits, f, indent=2)

# Save as individual text files
for split_name in ['train', 'validation', 'test']:
    with open(os.path.join(data_dir, f'{split_name}_proteins.txt'), 'w') as f:
        f.write('\n'.join(splits[split_name]))

print("\nProtein Split Summary:")
print("-" * 40)
print(f"Train proteins: {len(splits['train'])}")
print(f"Validation proteins: {len(splits['validation'])}")
print(f"Test proteins: {len(splits['test'])}")
print(f"Total proteins: {splits['metadata']['total_proteins']}")

print("\nFiles created:")
print(f"  - {data_dir}/protein_splits.json")
print(f"  - {data_dir}/train_proteins.txt")
print(f"  - {data_dir}/validation_proteins.txt")
print(f"  - {data_dir}/test_proteins.txt")

# Also create FASTA file for external tools
print("\nCreating FASTA file for external tool benchmarking...")
test_proteins = df_final.iloc[test_indices].groupby('Header').first()

with open(os.path.join(data_dir, 'test_sequences.fasta'), 'w') as f:
    for header, row in test_proteins.iterrows():
        f.write(f'>{header}\n{row["Sequence"]}\n')

# Create site information file
test_sites = df_final.iloc[test_indices][['Header', 'Position', 'AA', 'target']].copy()
test_sites.to_csv(os.path.join(data_dir, 'test_sites.csv'), index=False)

print(f"  - {data_dir}/test_sequences.fasta ({len(test_proteins)} proteins)")
print(f"  - {data_dir}/test_sites.csv ({len(test_sites)} sites)")

# ============================================================================
# 7.8 Summary Report
# ============================================================================

print("\n" + "="*80)
print("SECTION 7 SUMMARY - SUPERVISOR FEEDBACK IMPLEMENTATION")
print("="*80)

summary_report = """
# Supervisor Feedback Implementation Summary

## Completed Tasks

### A1: Per-Residue (S/T/Y) Metrics
- Computed separate precision, recall, F1, AUC for Serine, Threonine, Tyrosine
- Results saved to: tables/supervisor_feedback/per_residue_metrics.csv
- Visualization: plots/supervisor_feedback/per_residue_metrics.png

### A2: Calibration Analysis
- Computed Brier scores for all models
- Generated reliability diagrams (individual and combined)
- Computed Expected Calibration Error (ECE)
- Results saved to: tables/supervisor_feedback/calibration_scores.csv
- Visualizations: plots/supervisor_feedback/reliability_diagrams*.png

### A3: Precision-Recall Curves
- Generated PR curves for all models
- Computed Average Precision (AP) scores
- Results saved to: tables/supervisor_feedback/average_precision_scores.csv
- Visualization: plots/supervisor_feedback/pr_curves.png

### A4: Threshold Selection Rationale
- Analyzed performance across thresholds 0.1-0.9
- Identified optimal thresholds for each model
- Documented rationale for 0.5 threshold (balanced dataset)
- Results saved to: tables/supervisor_feedback/threshold_analysis.csv
- Visualization: plots/supervisor_feedback/threshold_analysis.png

### A5: 95% Confidence Intervals
- Computed bootstrap CIs for ALL metrics (accuracy, precision, recall, F1, AUC, MCC, Brier)
- Used 1000 bootstrap iterations
- Results saved to: tables/supervisor_feedback/confidence_intervals.csv
- Paper-format table: tables/supervisor_feedback/confidence_intervals_paper_format.csv
- Visualization: plots/supervisor_feedback/f1_confidence_intervals.png

### C1: Export Protein Split Lists
- Exported UniProt IDs for train/val/test splits
- Created JSON format: data/protein_splits.json
- Created individual text files: data/*_proteins.txt
- Created FASTA file for external tools: data/test_sequences.fasta
- Created site information: data/test_sites.csv

## Remaining Tasks (Require Additional Work)

### B1: Context Window Ablation
- Requires retraining TransformerV1 with window sizes: +/-1, +/-3, +/-5, +/-10
- See implementation plan for code template

### D1: External Tool Benchmarking
- Requires manual steps: running MusiteDeep, GPS, NetPhos
- Test sequences prepared in data/test_sequences.fasta
- See implementation plan for comparison code

### E1-E2: Interpretability
- SHAP analysis for CatBoost (code template in plan)
- Attention visualization for Transformer (code template in plan)

### F1-F2: Robustness Testing
- Hard negative strategy (requires kinase motif database)
- Cross-dataset validation (requires external dataset download)

"""

# Save summary report
with open(os.path.join(BASE_DIR, 'reports', 'supervisor_feedback', 'implementation_summary.md'), 'w') as f:
    f.write(summary_report)

print(summary_report)

# Save checkpoint
try:
    progress_tracker.mark_completed(
        "supervisor_feedback",
        metadata={
            'tasks_completed': ['A1', 'A2', 'A3', 'A4', 'A5', 'C1'],
            'tasks_remaining': ['B1', 'D1', 'E1', 'E2', 'F1', 'F2'],
            'n_bootstrap': N_BOOTSTRAP
        },
        checkpoint_data={
            'per_residue_metrics': per_residue_df.to_dict(),
            'calibration_scores': calibration_df.to_dict(),
            'pr_scores': pr_df.to_dict(),
            'confidence_intervals': ci_df.to_dict(),
            'protein_splits': splits
        }
    )
    print("\n Checkpoint saved successfully!")
except Exception as e:
    print(f"\nWarning: Could not save checkpoint: {e}")

print("\n" + "="*80)
print(" Section 7: Supervisor Feedback Implementation COMPLETED")
print("="*80)
