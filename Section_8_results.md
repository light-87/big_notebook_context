================================================================================
SECTION 8: INTERPRETABILITY & FEATURE ANALYSIS
================================================================================
SHAP library available
PyTorch and Transformers available

8.1 Loading Required Data
--------------------------------------------------
All data loaded successfully!
Test set: 10122 samples
Physicochemical features: 656 features

================================================================================
TASK E1: SHAP Analysis for CatBoost Physicochemical Model
================================================================================

Loaded CatBoost model for physicochemical features
Number of features: 656

Computing SHAP values for 1000 samples...
This may take several minutes...
SHAP values computed!

Generating SHAP summary plot...
Generating SHAP importance bar plot...

Aggregating SHAP values by position...
Generating top features table...

Top 20 Most Important Features:
------------------------------------------------------------
        feature  mean_abs_shap  rank
PC_pos18_prop05       0.620189     1
PC_pos11_prop11       0.572023     2
PC_pos15_prop06       0.516201     3
PC_pos10_prop11       0.382401     4
PC_pos17_prop06       0.364938     5
PC_pos09_prop12       0.329287     6
PC_pos02_prop06       0.328306     7
PC_pos01_prop12       0.319301     8
PC_pos07_prop09       0.307725     9
PC_pos09_prop02       0.284279    10
PC_pos16_prop10       0.281953    11
PC_pos23_prop11       0.278576    12
PC_pos02_prop15       0.254526    13
PC_pos06_prop12       0.247335    14
PC_pos16_prop06       0.243476    15
PC_pos14_prop01       0.220321    16
PC_pos12_prop03       0.215156    17
PC_pos05_prop11       0.208897    18
PC_pos14_prop07       0.189860    19
PC_pos30_prop01       0.157473    20

Generating SHAP dependence plots for top 6 features...

SHAP analysis completed!
Saved to:
  - plots/interpretability/shap_summary_beeswarm.png
  - plots/interpretability/shap_importance_bar.png
  - plots/interpretability/shap_position_importance.png
  - plots/interpretability/shap_dependence_plots.png
  - tables/interpretability/shap_feature_importance.csv
  - tables/interpretability/position_importance.csv

================================================================================
TASK E2: Attention/Embedding Visualization
================================================================================
Using transformer model: transformer_v1_20250703_112558

Loading ESM-2 model for attention extraction...
Some weights of EsmModel were not initialized from the model checkpoint at facebook/esm2_t6_8M_UR50D and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
ESM-2 model loaded on cuda

Extracting attention patterns for sample sequences...
Generating attention heatmaps...
Computing average attention profile around sites...

Generating t-SNE embedding visualization...
  Extracting embeddings for 400 samples...
  Running t-SNE...

Attention/embedding visualization completed!
Saved to:
  - plots/interpretability/attention_heatmaps.png
  - plots/interpretability/attention_profile.png
  - plots/interpretability/embedding_tsne.png
  - tables/interpretability/embedding_tsne.csv

================================================================================
SECTION 8 SUMMARY
================================================================================

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

================================================================================
Section 8: Interpretability Analysis COMPLETED
================================================================================