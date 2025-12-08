# Supervisor Feedback Implementation Plan
## Phosphorylation Site Prediction Project

**Created:** December 2025
**Status:** Planning Document
**Current Best Performance:** 81.60% F1 (Soft Voting Ensemble)

---

## Executive Summary

This document provides a prioritized implementation plan for the remaining supervisor requests. Based on analysis of your codebase, I've identified:

- **12 computational tasks** that can reuse existing predictions
- **4 tasks requiring retraining** or new model runs
- **3 tasks requiring external tools/data** (manual steps needed)
- **3 nice-to-have tasks** (lower priority)

**Estimated total effort:** ~15-20 hours of implementation + external tool setup time

---

## Current State Analysis

### What Exists (Can Reuse)
| Asset | Location | Description |
|-------|----------|-------------|
| Test predictions | `results/exp_3/transformers/*/predictions/test_predictions.csv` | Probability outputs for both transformers |
| ML predictions | Checkpoint: `ml_models_enhanced` | Predictions for all 5 feature types |
| Train/Val/Test indices | Checkpoint: `data_splitting` | Sample indices for each split |
| Protein lists | `train_proteins_list`, `val_proteins_list`, `test_proteins_list` | UniProt IDs saved in checkpoint |
| df_final | Checkpoint: `data_loading` | Full dataset with AA column (S/T/Y) |
| Feature matrices | Checkpoint: `feature_extraction` | All 5 feature types extracted |

### What's Missing
- Per-residue (S/T/Y) metrics
- Calibration analysis
- PR curves
- Confidence intervals for all metrics
- Context window ablation results
- External tool benchmarks
- SHAP/feature importance visualizations
- Attention visualizations

---

## Priority 1: Core Evaluation (Can Do Immediately)

These tasks can be completed using existing predictions without retraining.

### Task A1: Per-Residue (S/T/Y) Metrics
**Complexity:** Easy (2 hours)
**Dependencies:** None
**Reuses:** Existing test predictions + df_final['AA'] column

```python
# Implementation approach
def compute_per_residue_metrics(y_true, y_pred, y_prob, residue_types):
    """
    Compute metrics separately for S, T, Y residues
    """
    results = {}
    for residue in ['S', 'T', 'Y']:
        mask = residue_types == residue
        results[residue] = {
            'n_samples': mask.sum(),
            'precision': precision_score(y_true[mask], y_pred[mask]),
            'recall': recall_score(y_true[mask], y_pred[mask]),
            'f1': f1_score(y_true[mask], y_pred[mask]),
            'auc': roc_auc_score(y_true[mask], y_prob[mask])
        }
    return results
```

**Output:** Table with S/T/Y breakdown for each model

---

### Task A2: Calibration Analysis
**Complexity:** Easy (2 hours)
**Dependencies:** None
**Reuses:** Existing probability outputs

```python
from sklearn.calibration import calibration_curve, brier_score_loss
import matplotlib.pyplot as plt

def calibration_analysis(y_true, y_prob, model_name, n_bins=10):
    """Generate calibration plots and Brier score"""
    # Brier score (lower is better)
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Plot reliability diagram
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, 's-', label=f'{model_name} (Brier={brier:.4f})')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Plot - {model_name}')
    plt.legend()

    return brier, prob_true, prob_pred
```

**Output:** Reliability diagrams + Brier scores table

---

### Task A3: Precision-Recall Curves
**Complexity:** Easy (1.5 hours)
**Dependencies:** None
**Reuses:** Existing probability outputs

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curves(models_dict, y_true):
    """Plot PR curves for all models"""
    plt.figure(figsize=(10, 8))

    for model_name, y_prob in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
```

**Output:** PR curve comparison figure + AP scores table

---

### Task A4: Threshold Selection Rationale
**Complexity:** Easy (1.5 hours)
**Dependencies:** Task A3 (PR curves)
**Reuses:** Existing probability outputs

```python
def threshold_analysis(y_true, y_prob, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Analyze performance at different thresholds"""
    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        results.append({
            'threshold': thresh,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'n_positive_predictions': y_pred.sum()
        })

    # Find optimal threshold (max F1)
    optimal_idx = np.argmax([r['f1'] for r in results])

    return pd.DataFrame(results), thresholds[optimal_idx]
```

**Output:**
- Table of metrics at different thresholds
- Documentation of why 0.5 was used (balanced dataset, standard practice)
- Optimal threshold if different from 0.5

---

### Task A5: Confidence Intervals for All Metrics
**Complexity:** Medium (3 hours)
**Dependencies:** None
**Reuses:** Existing predictions

```python
from scipy import stats
import numpy as np

def bootstrap_confidence_intervals(y_true, y_pred, y_prob, n_bootstrap=1000, ci=0.95):
    """Compute 95% CIs for all primary metrics using bootstrap"""
    n_samples = len(y_true)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'mcc': []
    }

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        y_pr = y_prob[indices]

        # Skip if only one class in sample
        if len(np.unique(y_t)) < 2:
            continue

        metrics['accuracy'].append(accuracy_score(y_t, y_p))
        metrics['precision'].append(precision_score(y_t, y_p))
        metrics['recall'].append(recall_score(y_t, y_p))
        metrics['f1'].append(f1_score(y_t, y_p))
        metrics['auc'].append(roc_auc_score(y_t, y_pr))
        metrics['mcc'].append(matthews_corrcoef(y_t, y_p))

    # Compute CIs
    alpha = 1 - ci
    results = {}
    for metric, values in metrics.items():
        lower = np.percentile(values, alpha/2 * 100)
        upper = np.percentile(values, (1 - alpha/2) * 100)
        mean = np.mean(values)
        results[metric] = {
            'mean': mean,
            'lower_95ci': lower,
            'upper_95ci': upper,
            'ci_width': upper - lower
        }

    return results
```

**Output:** Table with point estimates and 95% CIs for all metrics

---

## Priority 2: Ablation Studies (Requires Some Retraining)

### Task B1: Context Window Ablation for TransformerV1
**Complexity:** Hard (6-8 hours including GPU time)
**Dependencies:** None (can run independently)
**Requires:** Retraining TransformerV1 with different window sizes

```python
# Implementation approach
WINDOW_SIZES = [1, 3, 5, 10]  # ±1, ±3 (current), ±5, ±10

def run_window_ablation():
    results = []
    for window in WINDOW_SIZES:
        # Modify WINDOW_SIZE in Section 5
        # Retrain TransformerV1
        # Record val/test performance
        results.append({
            'window_size': f'±{window}',
            'total_positions': 2 * window + 1,
            'feature_dim': (2 * window + 1) * 320,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'test_auc': test_auc
        })
    return pd.DataFrame(results)
```

**Output:**
- Table comparing window sizes
- Line plot of performance vs. window size
- Recommendation for optimal window

**Estimated time per window:** ~30-60 minutes training (depends on GPU)

---

## Priority 3: Release Train/Val/Test Protein Lists (Easy, Important)

### Task C1: Extract and Document Protein Split Lists
**Complexity:** Very Easy (30 minutes)
**Dependencies:** None
**Reuses:** Existing checkpoint data

```python
def export_protein_splits():
    """Export UniProt IDs for each split"""
    import json

    # Load from checkpoint
    checkpoint = progress_tracker.resume_from_checkpoint("data_splitting")

    # Extract protein lists
    splits = {
        'train': list(checkpoint['train_proteins_list']),
        'validation': list(checkpoint['val_proteins_list']),
        'test': list(checkpoint['test_proteins_list'])
    }

    # Add statistics
    splits['metadata'] = {
        'total_proteins': len(splits['train']) + len(splits['validation']) + len(splits['test']),
        'train_count': len(splits['train']),
        'val_count': len(splits['validation']),
        'test_count': len(splits['test']),
        'split_ratio': '70/15/15',
        'random_seed': 42
    }

    # Save as JSON
    with open('data/protein_splits.json', 'w') as f:
        json.dump(splits, f, indent=2)

    # Save as individual files
    for split_name, proteins in splits.items():
        if split_name != 'metadata':
            with open(f'data/{split_name}_proteins.txt', 'w') as f:
                f.write('\n'.join(proteins))

    return splits
```

**Output:**
- `protein_splits.json` (complete JSON with all splits)
- `train_proteins.txt`, `val_proteins.txt`, `test_proteins.txt`

---

## Priority 4: Benchmarking Against External Tools

### Task D1: External Tool Benchmarking
**Complexity:** Hard (8-12 hours + setup time)
**Dependencies:** Task C1 (protein lists for fair comparison)
**Manual Steps Required:** YES

#### Required External Tools:
| Tool | URL | Setup Difficulty |
|------|-----|-----------------|
| **MusiteDeep** | https://github.com/duolinwang/MusiteDeep | Medium (Docker available) |
| **GPS** | http://gps.biocuckoo.org/ | Easy (Web server) |
| **NetPhos** | https://services.healthtech.dtu.dk/service.php?NetPhos-3.1 | Easy (Web server) |
| **DeepPhos** | https://github.com/USTC-HIlab/DeepPhos | Medium (requires setup) |

#### Implementation Steps:

1. **Prepare test sequences** (I can help with this):
```python
def prepare_benchmark_sequences():
    """Extract test sequences in FASTA format for external tools"""
    test_df = df_final.iloc[test_indices]

    # Group by protein to avoid duplicates
    proteins = test_df.groupby('Header').first()

    # Write FASTA file
    with open('data/test_sequences.fasta', 'w') as f:
        for header, row in proteins.iterrows():
            f.write(f'>{header}\n{row["Sequence"]}\n')

    # Write site information for comparison
    test_sites = test_df[['Header', 'Position', 'AA', 'target']].copy()
    test_sites.to_csv('data/test_sites.csv', index=False)

    return proteins, test_sites
```

2. **Manual steps** (you need to do):
   - Run MusiteDeep on test_sequences.fasta
   - Submit to GPS web server
   - Submit to NetPhos web server
   - Collect predictions from each tool

3. **Parse and compare** (I can help with this):
```python
def compare_external_tools(external_predictions_dict):
    """Compare external tool predictions with our models"""
    results = []
    for tool_name, predictions in external_predictions_dict.items():
        # Align predictions with our test set
        # Compute metrics
        results.append({
            'tool': tool_name,
            'f1': f1_score(y_test, predictions),
            'auc': roc_auc_score(y_test, predictions),
            # ... other metrics
        })
    return pd.DataFrame(results)
```

**Output:**
- Comparison table with all tools
- Bar chart comparing performance
- Discussion of differences in methodology

---

## Priority 5: Interpretability & Biology

### Task E1: SHAP/Permutation Importance for CatBoost
**Complexity:** Medium (3-4 hours)
**Dependencies:** None
**Reuses:** Trained CatBoost model + physicochemical features

```python
import shap

def compute_shap_importance(model, X_test, feature_names):
    """Compute SHAP values for CatBoost model"""
    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', dpi=300)

    # Feature importance bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_importance.png', dpi=300)

    # Map to positions
    position_importance = aggregate_by_position(shap_values, feature_names)

    return shap_values, position_importance

def aggregate_by_position(shap_values, feature_names):
    """Aggregate SHAP values by position (for physicochemical: 41 positions x 16 properties)"""
    # Parse feature names to extract position
    # Sum absolute SHAP values per position
    position_importance = {}
    for i, name in enumerate(feature_names):
        # Extract position from feature name (e.g., "pos_-20_hydrophobicity")
        pos = extract_position(name)
        if pos not in position_importance:
            position_importance[pos] = 0
        position_importance[pos] += np.abs(shap_values[:, i]).mean()

    return position_importance
```

**Output:**
- SHAP summary plot
- Position importance heatmap
- Top features table with biological interpretation
- Kinase motif mapping (if positions align with known motifs)

---

### Task E2: Attention/Embedding Visualization
**Complexity:** Medium (4-5 hours)
**Dependencies:** None
**Requires:** Loading trained TransformerV1 model

```python
import torch
from transformers import AutoModel, AutoTokenizer

def visualize_attention(model, sequence, position, tokenizer):
    """Visualize ESM-2 attention patterns around phosphorylation site"""
    # Extract window
    window = sequence[max(0, position-20):position+21]

    # Tokenize
    inputs = tokenizer(window, return_tensors="pt")

    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # List of attention matrices per layer

    # Average attention across heads and layers
    avg_attention = torch.stack(attentions).mean(dim=[0, 1, 2])

    # Plot attention heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(avg_attention.numpy(), cmap='viridis')
    plt.title(f'Attention Pattern around Position {position}')
    plt.xlabel('Sequence Position')
    plt.ylabel('Sequence Position')

    return avg_attention

def visualize_embeddings(model, positive_sequences, negative_sequences, tokenizer):
    """t-SNE visualization of ESM-2 embeddings"""
    from sklearn.manifold import TSNE

    embeddings = []
    labels = []

    # Get embeddings for positive and negative sites
    for seq in positive_sequences:
        emb = get_embedding(model, seq, tokenizer)
        embeddings.append(emb)
        labels.append(1)

    for seq in negative_sequences:
        emb = get_embedding(model, seq, tokenizer)
        embeddings.append(emb)
        labels.append(0)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[labels==0, 0], embeddings_2d[labels==0, 1],
                alpha=0.5, label='Negative', c='blue')
    plt.scatter(embeddings_2d[labels==1, 0], embeddings_2d[labels==1, 1],
                alpha=0.5, label='Positive', c='red')
    plt.legend()
    plt.title('ESM-2 Embedding Space (t-SNE)')

    return embeddings_2d
```

**Output:**
- Attention heatmaps for example sequences
- t-SNE visualization of embedding space
- Embedding clusters analysis

---

## Priority 6: Negative Sampling & Robustness (Harder Tasks)

### Task F1: Hard-Negative Strategy
**Complexity:** Hard (8-10 hours)
**Dependencies:** None (but benefits from Task E1)
**Requires:** Kinase motif database + retraining

```python
def create_hard_negatives():
    """Create hard negatives based on kinase motif similarity"""

    # 1. Load kinase motifs (need to obtain this data)
    kinase_motifs = load_kinase_motifs()  # e.g., from PhosphoSitePlus

    # 2. Score all non-phosphorylated S/T/Y sites by motif similarity
    def motif_similarity_score(sequence, position, motifs):
        window = sequence[position-5:position+6]  # ±5 around site
        max_score = 0
        for motif in motifs:
            score = compute_pssm_score(window, motif)
            max_score = max(max_score, score)
        return max_score

    # 3. Select hard negatives (high motif similarity, not phosphorylated)
    hard_neg_candidates = []
    for _, row in df_merged.iterrows():
        seq = row['Sequence']
        positive_positions = set(...)  # from labels

        for pos, aa in enumerate(seq):
            if aa in 'STY' and pos not in positive_positions:
                score = motif_similarity_score(seq, pos, kinase_motifs)
                if score > threshold:  # High similarity to kinase motif
                    hard_neg_candidates.append({
                        'Header': row['Header'],
                        'Position': pos,
                        'AA': aa,
                        'motif_score': score
                    })

    # 4. Sample hard negatives
    hard_negatives_df = pd.DataFrame(hard_neg_candidates)
    hard_negatives_df = hard_negatives_df.nlargest(len(positive_samples), 'motif_score')

    return hard_negatives_df
```

**Manual Steps Required:**
- Download kinase motif database (PhosphoSitePlus, HOMD, etc.)
- Parse motif files into usable format

**Output:**
- Hard negative dataset
- Performance comparison table (random vs. hard negatives)
- Analysis of what makes negatives "hard"

---

### Task F2: Cross-Dataset Validation
**Complexity:** Hard (6-8 hours)
**Dependencies:** None
**Manual Steps Required:** YES (need to download external dataset)

#### External Datasets Options:
| Dataset | URL | Difficulty |
|---------|-----|------------|
| **PhosphoSitePlus** (newer version) | https://www.phosphosite.org/ | Medium |
| **dbPTM** | https://dbptm.mbc.nctu.edu.tw/ | Medium |
| **UniProt PTM annotations** | https://www.uniprot.org/ | Easy |
| **EPSD** | http://epsd.biocuckoo.org/ | Medium |

```python
def cross_dataset_validation(external_dataset_path):
    """Test trained model on independent dataset"""

    # 1. Load external dataset
    external_df = load_external_phospho_data(external_dataset_path)

    # 2. Filter to proteins NOT in our training set
    train_proteins = set(train_proteins_list)
    external_df = external_df[~external_df['UniProt_ID'].isin(train_proteins)]

    # 3. Generate features for external data
    # (use same feature extraction pipeline)

    # 4. Run predictions with trained models
    predictions = model.predict(external_features)

    # 5. Compute metrics
    results = compute_metrics(external_df['target'], predictions)

    return results
```

**Output:**
- Cross-dataset performance table
- Analysis of performance drop (if any)
- Dataset overlap statistics

---

### Task F3: Temporal Holdout
**Complexity:** Medium-Hard (4-6 hours)
**Dependencies:** None
**Data Requirement:** Timestamp information for phosphorylation discoveries

**Assessment:** This may not be feasible if your dataset doesn't include discovery dates. Check PhosphoSitePlus for timestamp information.

```python
def check_temporal_data_availability():
    """Check if we have timestamp information"""
    # PhosphoSitePlus includes 'PMID' and publication dates
    # Could use publication dates as proxy for discovery time

    # If available:
    # - Split by date (train on pre-2020, test on 2020+)
    # - Compare temporal vs. random split performance
    pass
```

---

## Priority 7: Nice-to-Have (Lower Priority)

### Task G1: Decision Curve Analysis
**Complexity:** Easy (2 hours)
**Dependencies:** None

```python
def decision_curve_analysis(y_true, y_prob, thresholds=np.arange(0, 1, 0.01)):
    """Decision curve for clinical/practical utility"""
    net_benefits = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        n = len(y_true)

        net_benefit = tp/n - fp/n * (thresh / (1 - thresh))
        net_benefits.append(net_benefit)

    plt.plot(thresholds, net_benefits, label='Model')
    plt.plot(thresholds, [0]*len(thresholds), 'k--', label='Treat None')
    # ... treat all line
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
```

---

### Task G2: Error Taxonomy
**Complexity:** Medium (3-4 hours)
**Dependencies:** Task E1 (for feature-based analysis)

```python
def error_taxonomy(test_df, predictions, probabilities):
    """Categorize errors by type"""

    # Get false positives and false negatives
    fp_mask = (predictions == 1) & (test_df['target'] == 0)
    fn_mask = (predictions == 0) & (test_df['target'] == 1)

    error_analysis = {
        'false_positives': {
            'motif_like': count_motif_like_fps(test_df[fp_mask]),
            'low_complexity': count_low_complexity(test_df[fp_mask]),
            'disordered': count_disordered_regions(test_df[fp_mask])
        },
        'false_negatives': {
            # Similar analysis
        }
    }

    return error_analysis
```

**Requires:**
- IUPred or similar disorder predictor
- Low-complexity region detection (SEG algorithm)
- Kinase motif database

---

## Dependency Graph

```
                    ┌─────────────────────────────────────────┐
                    │         PRIORITY 1 (IMMEDIATE)          │
                    │    Can use existing predictions         │
                    └─────────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
    │ A1: S/T/Y   │           │ A2: Calib.  │           │ A5: 95% CIs │
    │   Metrics   │           │   Analysis  │           │ All Metrics │
    └─────────────┘           └─────────────┘           └─────────────┘
           │                          │
           │                          ▼
           │                  ┌─────────────┐
           │                  │ A3: PR      │
           │                  │   Curves    │
           │                  └─────────────┘
           │                          │
           │                          ▼
           │                  ┌─────────────┐
           │                  │ A4: Thresh  │
           │                  │   Analysis  │
           │                  └─────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │         PRIORITY 2 (IMPORTANT)          │
    │    Protein lists for benchmarking       │
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ C1: Export  │
    │ Protein IDs │─────────────────────────────┐
    └─────────────┘                             │
           │                                    │
           ▼                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              PRIORITY 3 (REQUIRES RETRAINING)               │
    └─────────────────────────────────────────────────────────────┘
           │                                    │
           ▼                                    ▼
    ┌─────────────┐                     ┌─────────────┐
    │ B1: Window  │                     │ E1: SHAP    │
    │   Ablation  │                     │  Analysis   │
    │  (RETRAIN)  │                     │             │
    └─────────────┘                     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ E2: Attn.   │
                                        │   Viz       │
                                        └─────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │         PRIORITY 4 (REQUIRES EXTERNAL RESOURCES)            │
    │         Manual steps required - cannot automate             │
    └─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ D1: External│      │ F1: Hard    │      │ F2: Cross   │
    │   Tools     │      │   Negatives │      │   Dataset   │
    │ (MANUAL)    │      │ (Needs Data)│      │ (Needs Data)│
    └─────────────┘      └─────────────┘      └─────────────┘
```

---

## Suggested File/Notebook Structure

```
big_notebook_context/
├── Section7_supervisor_feedback.py     # NEW: All supervisor feedback analyses
├── Notebooks/
│   ├── Supervisor_Feedback/
│   │   ├── 01_Per_Residue_Metrics.ipynb      # Task A1
│   │   ├── 02_Calibration_Analysis.ipynb     # Task A2
│   │   ├── 03_PR_Curves_Thresholds.ipynb     # Tasks A3, A4
│   │   ├── 04_Confidence_Intervals.ipynb     # Task A5
│   │   ├── 05_Window_Ablation.ipynb          # Task B1
│   │   ├── 06_SHAP_Interpretability.ipynb    # Task E1
│   │   ├── 07_Attention_Visualization.ipynb  # Task E2
│   │   ├── 08_External_Benchmarking.ipynb    # Task D1
│   │   └── 09_Hard_Negatives.ipynb           # Task F1
│   │
├── data/
│   ├── protein_splits.json                   # Exported protein lists
│   ├── train_proteins.txt
│   ├── val_proteins.txt
│   ├── test_proteins.txt
│   ├── test_sequences.fasta                  # For external tools
│   └── external_datasets/                    # Downloaded datasets
│       ├── phosphositeplus_2024/
│       └── dbptm/
│
├── results/exp_3/
│   ├── tables/
│   │   ├── supervisor_feedback/
│   │   │   ├── per_residue_metrics.csv
│   │   │   ├── calibration_scores.csv
│   │   │   ├── confidence_intervals.csv
│   │   │   ├── threshold_analysis.csv
│   │   │   ├── window_ablation.csv
│   │   │   └── external_tool_comparison.csv
│   │
│   ├── plots/
│   │   ├── supervisor_feedback/
│   │   │   ├── reliability_diagrams.png
│   │   │   ├── pr_curves.png
│   │   │   ├── threshold_analysis.png
│   │   │   ├── window_ablation.png
│   │   │   ├── shap_summary.png
│   │   │   ├── shap_position_importance.png
│   │   │   ├── attention_heatmaps.png
│   │   │   └── embedding_tsne.png
│   │
│   └── reports/
│       └── supervisor_feedback/
│           ├── evaluation_summary.md
│           └── ablation_summary.md
```

---

## Implementation Order (Recommended)

### Week 1: Core Evaluation (Can start immediately)
| Day | Task | Time | Output |
|-----|------|------|--------|
| 1 | A1: Per-residue metrics | 2h | S/T/Y metrics table |
| 1 | A2: Calibration analysis | 2h | Brier scores + reliability plots |
| 2 | A3: PR curves | 1.5h | PR curve figure |
| 2 | A4: Threshold analysis | 1.5h | Threshold table + rationale |
| 3 | A5: Confidence intervals | 3h | 95% CIs for all metrics |
| 3 | C1: Export protein lists | 0.5h | JSON + TXT files |

### Week 2: Interpretability + Ablation
| Day | Task | Time | Output |
|-----|------|------|--------|
| 1-2 | E1: SHAP analysis | 4h | SHAP plots + importance |
| 3 | E2: Attention visualization | 4h | Attention heatmaps |
| 4-5 | B1: Window ablation | 6-8h | Ablation table + figure |

### Week 3+: External Resources (As Time Permits)
| Task | Effort | Manual Steps |
|------|--------|--------------|
| D1: External tools | 8-12h | Run external tools on test set |
| F1: Hard negatives | 8-10h | Download kinase motif database |
| F2: Cross-dataset | 6-8h | Download PhosphoSitePlus |

---

## Manual Steps Checklist

### Immediate (No Downloads Needed)
- [ ] Run Section7_supervisor_feedback.py for Tasks A1-A5, C1

### External Tool Setup (For Task D1)
- [ ] Install MusiteDeep: `git clone https://github.com/duolinwang/MusiteDeep`
- [ ] Access GPS web server: http://gps.biocuckoo.org/
- [ ] Access NetPhos web server: https://services.healthtech.dtu.dk/
- [ ] Submit test_sequences.fasta to each tool
- [ ] Collect and parse predictions

### Data Downloads (For Tasks F1, F2)
- [ ] PhosphoSitePlus: Create account at https://www.phosphosite.org/
- [ ] Download latest kinase-substrate dataset
- [ ] Download site dataset (for cross-validation)
- [ ] dbPTM (alternative): https://dbptm.mbc.nctu.edu.tw/

---

## Summary Table

| Category | Task | Complexity | Time | Dependencies | Manual Steps |
|----------|------|------------|------|--------------|--------------|
| **Evaluation** | A1: S/T/Y metrics | Easy | 2h | None | No |
| | A2: Calibration | Easy | 2h | None | No |
| | A3: PR curves | Easy | 1.5h | None | No |
| | A4: Threshold | Easy | 1.5h | A3 | No |
| | A5: 95% CIs | Medium | 3h | None | No |
| **Ablation** | B1: Window ablation | Hard | 6-8h | None | No (but GPU needed) |
| **Protein Lists** | C1: Export IDs | Very Easy | 0.5h | None | No |
| **Benchmarking** | D1: External tools | Hard | 8-12h | C1 | YES |
| **Robustness** | F1: Hard negatives | Hard | 8-10h | None | YES (data) |
| | F2: Cross-dataset | Hard | 6-8h | None | YES (data) |
| **Interpretability** | E1: SHAP | Medium | 4h | None | No |
| | E2: Attention viz | Medium | 4-5h | None | No |
| **Nice-to-have** | G1: Decision curve | Easy | 2h | None | No |
| | G2: Error taxonomy | Medium | 3-4h | E1 | Partial |

---

## Minimum Viable Changes (MVP)

If you have limited time, focus on these **essential items** that address core supervisor concerns:

1. **A1: Per-residue (S/T/Y) metrics** - Shows residue-specific performance
2. **A5: Confidence intervals** - Statistical rigor for all claims
3. **A2-A3: Calibration + PR curves** - Standard evaluation completeness
4. **C1: Protein lists** - Reproducibility and benchmarking foundation
5. **E1: SHAP analysis** - Biological interpretability

This MVP can be completed in **~12 hours** and addresses the most critical supervisor feedback.

---

## Questions for Supervisor

Before implementing everything, consider asking your supervisor:

1. **Benchmarking priority:** Which external tools are most important to compare against?
2. **Hard negatives:** Is there a specific kinase motif database they prefer?
3. **Cross-dataset:** Which external dataset (PhosphoSitePlus vs. dbPTM) is more appropriate?
4. **Temporal holdout:** Is this strictly required, or can we note it as future work?
5. **Window ablation:** Are 4 window sizes sufficient, or should we test more?

---

*Document generated for phosphorylation site prediction project supervisor feedback implementation.*
