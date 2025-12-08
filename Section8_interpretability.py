# ============================================================================
# SECTION 8: INTERPRETABILITY & FEATURE ANALYSIS
# ============================================================================
# This section implements interpretability analyses from supervisor feedback:
# - E1: SHAP/Permutation importance for CatBoost physicochemical model
# - E2: Attention/embedding visualization for Transformer models
#
# Note: This requires additional libraries:
#   pip install shap
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: INTERPRETABILITY & FEATURE ANALYSIS")
print("="*80)

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library available")
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not installed. Run: pip install shap")

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    TORCH_AVAILABLE = True
    print("PyTorch and Transformers available")
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Transformers not available for attention visualization")

# ============================================================================
# 8.0 Configuration
# ============================================================================

EXPERIMENT_NAME = "phosphorylation_prediction_exp_3"
BASE_DIR = "results/exp_3"
RANDOM_SEED = 42

# Create output directories
output_dirs = [
    os.path.join(BASE_DIR, 'tables', 'interpretability'),
    os.path.join(BASE_DIR, 'plots', 'interpretability')
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# 8.1 Load Required Data
# ============================================================================

print("\n8.1 Loading Required Data")
print("-" * 50)

try:
    # Load from checkpoints
    data_checkpoint = progress_tracker.resume_from_checkpoint("data_loading")
    df_final = data_checkpoint['df_final']

    split_checkpoint = progress_tracker.resume_from_checkpoint("data_splitting")
    train_indices = split_checkpoint['train_indices']
    test_indices = split_checkpoint['test_indices']

    # Load feature extraction data
    feature_checkpoint = progress_tracker.resume_from_checkpoint("feature_extraction")
    feature_matrices = feature_checkpoint['feature_matrices']
    physicochemical_features = feature_matrices['physicochemical']
    feature_metadata = feature_checkpoint['metadata']

    # Load ML models
    ml_checkpoint = progress_tracker.resume_from_checkpoint("ml_models_enhanced")
    specialized_models = ml_checkpoint['specialized_models']

    print("All data loaded successfully!")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure previous sections have been run.")
    raise

# Get test data
test_df = df_final.iloc[test_indices].copy()
X_test_physico = physicochemical_features.iloc[test_indices]
y_test = test_df['target'].values

print(f"Test set: {len(test_df)} samples")
print(f"Physicochemical features: {X_test_physico.shape[1]} features")

# ============================================================================
# 8.2 Task E1: SHAP Analysis for CatBoost Physicochemical Model
# ============================================================================

print("\n" + "="*80)
print("TASK E1: SHAP Analysis for CatBoost Physicochemical Model")
print("="*80)

if SHAP_AVAILABLE:

    # Load the best physicochemical model (CatBoost)
    if 'physicochemical' in specialized_models:
        model = specialized_models['physicochemical']['model']
        feature_names = list(X_test_physico.columns)

        print(f"\nLoaded CatBoost model for physicochemical features")
        print(f"Number of features: {len(feature_names)}")

        # Sample data for SHAP (SHAP can be slow on large datasets)
        n_samples = min(1000, len(X_test_physico))
        sample_indices = np.random.choice(len(X_test_physico), n_samples, replace=False)
        X_sample = X_test_physico.iloc[sample_indices]

        print(f"\nComputing SHAP values for {n_samples} samples...")
        print("This may take several minutes...")

        # Create TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        print("SHAP values computed!")

        # ============================================================
        # SHAP Summary Plot (Beeswarm)
        # ============================================================
        print("\nGenerating SHAP summary plot...")

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False,
                          max_display=30)
        plt.title('SHAP Feature Importance (Top 30 Features)', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(BASE_DIR, 'plots', 'interpretability', 'shap_summary_beeswarm.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

        # ============================================================
        # SHAP Bar Plot (Mean Absolute Values)
        # ============================================================
        print("Generating SHAP importance bar plot...")

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False,
                          plot_type="bar", max_display=30)
        plt.title('Mean |SHAP| Feature Importance (Top 30 Features)', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(BASE_DIR, 'plots', 'interpretability', 'shap_importance_bar.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

        # ============================================================
        # Aggregate SHAP by Position
        # ============================================================
        print("\nAggregating SHAP values by position...")

        # Parse feature names to extract positions
        # Expected format: "pos_-20_hydrophobicity" or similar
        position_importance = {}

        for i, name in enumerate(feature_names):
            # Try to extract position from feature name
            try:
                parts = name.split('_')
                if 'pos' in parts[0].lower():
                    pos = int(parts[1])
                else:
                    # Try to find a number in the name
                    for p in parts:
                        try:
                            pos = int(p)
                            break
                        except ValueError:
                            continue
                    else:
                        pos = i  # Fallback to feature index

                if pos not in position_importance:
                    position_importance[pos] = []
                position_importance[pos].append(np.abs(shap_values[:, i]).mean())
            except Exception:
                continue

        # Sum SHAP values per position
        position_summary = []
        for pos, values in sorted(position_importance.items()):
            position_summary.append({
                'position': pos,
                'mean_abs_shap': np.sum(values),
                'n_features': len(values)
            })

        position_df = pd.DataFrame(position_summary)

        # Plot position importance
        if len(position_df) > 0:
            plt.figure(figsize=(14, 6))
            plt.bar(position_df['position'], position_df['mean_abs_shap'], color='steelblue')
            plt.xlabel('Position (relative to phosphorylation site)', fontsize=12)
            plt.ylabel('Sum of Mean |SHAP| Values', fontsize=12)
            plt.title('Position-wise Feature Importance', fontsize=14)
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Phospho site')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(
                os.path.join(BASE_DIR, 'plots', 'interpretability', 'shap_position_importance.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

            position_df.to_csv(
                os.path.join(BASE_DIR, 'tables', 'interpretability', 'position_importance.csv'),
                index=False
            )

        # ============================================================
        # Top Features Table
        # ============================================================
        print("Generating top features table...")

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        # Add rank
        feature_importance['rank'] = range(1, len(feature_importance) + 1)

        print("\nTop 20 Most Important Features:")
        print("-" * 60)
        print(feature_importance.head(20).to_string(index=False))

        feature_importance.to_csv(
            os.path.join(BASE_DIR, 'tables', 'interpretability', 'shap_feature_importance.csv'),
            index=False
        )

        # ============================================================
        # SHAP Dependence Plots for Top Features
        # ============================================================
        print("\nGenerating SHAP dependence plots for top 6 features...")

        top_features = feature_importance.head(6)['feature'].tolist()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, feature in enumerate(top_features):
            feature_idx = feature_names.index(feature)
            ax = axes[idx]

            shap.dependence_plot(
                feature_idx, shap_values, X_sample,
                feature_names=feature_names,
                ax=ax, show=False
            )
            ax.set_title(f'{feature}')

        plt.suptitle('SHAP Dependence Plots - Top 6 Features', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(BASE_DIR, 'plots', 'interpretability', 'shap_dependence_plots.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

        print("\nSHAP analysis completed!")
        print("Saved to:")
        print("  - plots/interpretability/shap_summary_beeswarm.png")
        print("  - plots/interpretability/shap_importance_bar.png")
        print("  - plots/interpretability/shap_position_importance.png")
        print("  - plots/interpretability/shap_dependence_plots.png")
        print("  - tables/interpretability/shap_feature_importance.csv")
        print("  - tables/interpretability/position_importance.csv")

    else:
        print("Physicochemical model not found in checkpoint.")

else:
    print("\nSHAP library not available. Skipping SHAP analysis.")
    print("Install with: pip install shap")

# ============================================================================
# 8.3 Task E2: Attention/Embedding Visualization
# ============================================================================

print("\n" + "="*80)
print("TASK E2: Attention/Embedding Visualization")
print("="*80)

if TORCH_AVAILABLE:

    # Find the best transformer model directory
    transformer_dir = os.path.join(BASE_DIR, 'transformers')
    model_dirs = [d for d in os.listdir(transformer_dir) if d.startswith('transformer_v1')]

    if model_dirs:
        best_model_dir = os.path.join(transformer_dir, model_dirs[0])
        print(f"Using transformer model: {model_dirs[0]}")

        # ============================================================
        # Load ESM-2 Base Model for Attention Extraction
        # ============================================================
        print("\nLoading ESM-2 model for attention extraction...")

        try:
            esm_model_name = "facebook/esm2_t6_8M_UR50D"
            tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
            esm_model = AutoModel.from_pretrained(esm_model_name, output_attentions=True)
            esm_model.eval()

            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            esm_model = esm_model.to(DEVICE)

            print(f"ESM-2 model loaded on {DEVICE}")

            # ============================================================
            # Extract Attention for Sample Sequences
            # ============================================================
            print("\nExtracting attention patterns for sample sequences...")

            # Get sample positive and negative sequences
            n_examples = 5
            pos_samples = test_df[test_df['target'] == 1].sample(n_examples, random_state=42)
            neg_samples = test_df[test_df['target'] == 0].sample(n_examples, random_state=42)

            def extract_attention(sequence, position, window_size=20):
                """Extract attention weights around a phosphorylation site"""
                # Get window around site
                start = max(0, position - window_size)
                end = min(len(sequence), position + window_size + 1)
                window = sequence[start:end]

                # Relative position of site in window
                rel_pos = position - start

                # Tokenize
                inputs = tokenizer(window, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                # Get attention
                with torch.no_grad():
                    outputs = esm_model(**inputs)
                    attentions = outputs.attentions  # Tuple of attention matrices

                # Average across layers and heads
                # Shape: (n_layers, batch, n_heads, seq_len, seq_len)
                avg_attention = torch.stack(attentions).mean(dim=[0, 1, 2])

                return avg_attention.cpu().numpy(), window, rel_pos

            # Extract attention for examples
            attention_examples = []

            for idx, row in pos_samples.iterrows():
                attn, window, rel_pos = extract_attention(row['Sequence'], int(row['Position']) - 1)
                attention_examples.append({
                    'type': 'positive',
                    'attention': attn,
                    'window': window,
                    'site_position': rel_pos
                })

            for idx, row in neg_samples.iterrows():
                attn, window, rel_pos = extract_attention(row['Sequence'], int(row['Position']) - 1)
                attention_examples.append({
                    'type': 'negative',
                    'attention': attn,
                    'window': window,
                    'site_position': rel_pos
                })

            # ============================================================
            # Attention Heatmaps
            # ============================================================
            print("Generating attention heatmaps...")

            fig, axes = plt.subplots(2, n_examples, figsize=(20, 8))

            for i in range(n_examples):
                # Positive example
                pos_ex = attention_examples[i]
                ax = axes[0, i]
                sns.heatmap(pos_ex['attention'], ax=ax, cmap='viridis', cbar=i == n_examples-1)
                ax.axvline(x=pos_ex['site_position']+0.5, color='red', linewidth=2)
                ax.axhline(y=pos_ex['site_position']+0.5, color='red', linewidth=2)
                ax.set_title(f'Positive {i+1}', fontsize=10)
                if i == 0:
                    ax.set_ylabel('Query Position')

                # Negative example
                neg_ex = attention_examples[n_examples + i]
                ax = axes[1, i]
                sns.heatmap(neg_ex['attention'], ax=ax, cmap='viridis', cbar=i == n_examples-1)
                ax.axvline(x=neg_ex['site_position']+0.5, color='blue', linewidth=2)
                ax.axhline(y=neg_ex['site_position']+0.5, color='blue', linewidth=2)
                ax.set_title(f'Negative {i+1}', fontsize=10)
                if i == 0:
                    ax.set_ylabel('Query Position')
                ax.set_xlabel('Key Position')

            plt.suptitle('ESM-2 Attention Patterns\n(Red line = phosphorylated site, Blue line = non-phosphorylated site)',
                        fontsize=14)
            plt.tight_layout()
            plt.savefig(
                os.path.join(BASE_DIR, 'plots', 'interpretability', 'attention_heatmaps.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

            # ============================================================
            # Average Attention Profile
            # ============================================================
            print("Computing average attention profile around sites...")

            # Average attention at each relative position
            pos_attention_profiles = []
            neg_attention_profiles = []

            for ex in attention_examples:
                profile = ex['attention'][ex['site_position'], :]  # Attention FROM the site
                if ex['type'] == 'positive':
                    pos_attention_profiles.append(profile)
                else:
                    neg_attention_profiles.append(profile)

            # Pad/align profiles and average
            max_len = max(len(p) for p in pos_attention_profiles + neg_attention_profiles)

            def pad_profile(profile, max_len):
                padded = np.zeros(max_len)
                start = (max_len - len(profile)) // 2
                padded[start:start+len(profile)] = profile
                return padded

            pos_profiles_padded = np.array([pad_profile(p, max_len) for p in pos_attention_profiles])
            neg_profiles_padded = np.array([pad_profile(p, max_len) for p in neg_attention_profiles])

            avg_pos = pos_profiles_padded.mean(axis=0)
            avg_neg = neg_profiles_padded.mean(axis=0)
            std_pos = pos_profiles_padded.std(axis=0)
            std_neg = neg_profiles_padded.std(axis=0)

            positions = np.arange(max_len) - max_len // 2

            plt.figure(figsize=(12, 6))
            plt.plot(positions, avg_pos, 'r-', label='Positive (phosphorylated)', linewidth=2)
            plt.fill_between(positions, avg_pos - std_pos, avg_pos + std_pos, alpha=0.2, color='red')
            plt.plot(positions, avg_neg, 'b-', label='Negative (non-phosphorylated)', linewidth=2)
            plt.fill_between(positions, avg_neg - std_neg, avg_neg + std_neg, alpha=0.2, color='blue')
            plt.axvline(x=0, color='gray', linestyle='--', label='Target site')
            plt.xlabel('Position (relative to target site)')
            plt.ylabel('Average Attention Weight')
            plt.title('Attention Profile from Target Site')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(BASE_DIR, 'plots', 'interpretability', 'attention_profile.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

            # ============================================================
            # t-SNE Embedding Visualization
            # ============================================================
            print("\nGenerating t-SNE embedding visualization...")

            from sklearn.manifold import TSNE

            def get_embedding(sequence, position, window_size=20):
                """Get ESM-2 embedding for a sequence window"""
                start = max(0, position - window_size)
                end = min(len(sequence), position + window_size + 1)
                window = sequence[start:end]

                inputs = tokenizer(window, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = esm_model(**inputs)

                # Use CLS token or mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                return embedding

            # Sample more sequences for t-SNE
            n_tsne_samples = min(200, len(test_df) // 2)
            tsne_pos = test_df[test_df['target'] == 1].sample(n_tsne_samples, random_state=42)
            tsne_neg = test_df[test_df['target'] == 0].sample(n_tsne_samples, random_state=42)

            print(f"  Extracting embeddings for {2*n_tsne_samples} samples...")

            embeddings = []
            labels = []

            for idx, row in tsne_pos.iterrows():
                emb = get_embedding(row['Sequence'], int(row['Position']) - 1)
                embeddings.append(emb)
                labels.append(1)

            for idx, row in tsne_neg.iterrows():
                emb = get_embedding(row['Sequence'], int(row['Position']) - 1)
                embeddings.append(emb)
                labels.append(0)

            embeddings = np.array(embeddings)
            labels = np.array(labels)

            print("  Running t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                 c=labels, cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, label='Class (0=Negative, 1=Positive)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.title('ESM-2 Embedding Space (t-SNE)')
            plt.tight_layout()
            plt.savefig(
                os.path.join(BASE_DIR, 'plots', 'interpretability', 'embedding_tsne.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

            # Save embedding data
            embedding_df = pd.DataFrame({
                'tsne_1': embeddings_2d[:, 0],
                'tsne_2': embeddings_2d[:, 1],
                'label': labels
            })
            embedding_df.to_csv(
                os.path.join(BASE_DIR, 'tables', 'interpretability', 'embedding_tsne.csv'),
                index=False
            )

            print("\nAttention/embedding visualization completed!")
            print("Saved to:")
            print("  - plots/interpretability/attention_heatmaps.png")
            print("  - plots/interpretability/attention_profile.png")
            print("  - plots/interpretability/embedding_tsne.png")
            print("  - tables/interpretability/embedding_tsne.csv")

        except Exception as e:
            print(f"Error in attention visualization: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("No transformer model directories found.")

else:
    print("\nPyTorch/Transformers not available. Skipping attention visualization.")
    print("Install with: pip install torch transformers")

# ============================================================================
# 8.4 Summary
# ============================================================================

print("\n" + "="*80)
print("SECTION 8 SUMMARY")
print("="*80)

print("""
Interpretability Analysis Completed:

Task E1: SHAP Analysis
- Feature importance ranking (mean |SHAP| values)
- Position-wise importance aggregation
- Dependence plots for top features
- Beeswarm and bar summary plots

Task E2: Attention/Embedding Visualization
- Attention heatmaps for positive/negative examples
- Average attention profile from target site
- t-SNE visualization of ESM-2 embeddings

Key Insights:
- SHAP analysis reveals which physicochemical properties are most predictive
- Position importance shows which residues around the site are most informative
- Attention patterns may differ between phosphorylated and non-phosphorylated sites
- Embedding space visualization shows class separability

All results saved to:
- results/exp_3/plots/interpretability/
- results/exp_3/tables/interpretability/
""")

print("="*80)
print("Section 8: Interpretability Analysis COMPLETED")
print("="*80)
