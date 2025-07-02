# ============================================================================
# SECTION 4: ENHANCED MACHINE LEARNING MODELS - STRATEGIC ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: ENHANCED MACHINE LEARNING MODELS - STRATEGIC ARCHITECTURE")
print("="*80)

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, using XGBoost as fallback")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available, skipping interpretability analysis")

import gc
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# 4.1 Configuration & Setup
# ============================================================================

print("\n4.1 Configuration & Setup")
print("-" * 40)

# Selected optimal configurations
SELECTED_CONFIGS = {
    'physicochemical': {
        'method': 'mutual_info_500',
        'model': 'catboost',
        'features': 500,
        'expected_f1': 0.7820,
        'description': 'Feature Selection Champion'
    },
    'binary': {
        'method': 'pca_100',
        'model': 'xgboost',
        'features': 100,
        'expected_f1': 0.7527,
        'description': 'High Efficiency'
    },
    'aac': {
        'method': 'polynomial',
        'model': 'xgboost',
        'features': 210,
        'expected_f1': 0.7192,
        'description': 'Feature Interactions'
    },
    'dpc': {
        'method': 'pca_30',
        'model': 'catboost',
        'features': 30,
        'expected_f1': 0.7188,
        'description': 'Optimal PCA'
    },
    'tpc': {
        'method': 'pca_50',
        'model': 'catboost',
        'features': 50,
        'expected_f1': 0.6858,
        'description': 'PCA Champion'
    }
}

# Model configurations (without early stopping in base params)
MODEL_CONFIGS = {
    'xgboost': {
        'class': xgb.XGBClassifier,
        'params': {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_SEED,
            'eval_metric': 'logloss',
            'verbose': False
        }
    },
    'catboost': {
        'class': CatBoostClassifier if CATBOOST_AVAILABLE else xgb.XGBClassifier,
        'params': {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': RANDOM_SEED,
            'verbose': False
        } if CATBOOST_AVAILABLE else {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': RANDOM_SEED,
            'eval_metric': 'logloss',
            'verbose': False
        }
    },
    'lightgbm': {
        'class': lgb.LGBMClassifier,
        'params': {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': RANDOM_SEED,
            'verbose': -1
        }
    }
}

print("✓ Selected configurations loaded")
for feature_type, config in SELECTED_CONFIGS.items():
    print(f"  {feature_type.upper()}: {config['description']} -> {config['method']} + {config['model']}")

# ============================================================================
# 4.2 Load Required Data
# ============================================================================

print("\n4.2 Loading Required Data")
print("-" * 40)

# Check if variables exist from previous sections
required_vars = ['y_train', 'y_val', 'y_test', 'train_indices', 'val_indices', 'test_indices', 'cv_folds']
missing_vars = [var for var in required_vars if var not in locals()]

if missing_vars or 'aac_features' not in locals():
    print("Loading required data from previous checkpoints...")
    
    # Load from Section 3 (data splitting)
    checkpoint_data = progress_tracker.resume_from_checkpoint("data_splitting")
    if checkpoint_data:
        train_indices = checkpoint_data['train_indices']
        val_indices = checkpoint_data['val_indices'] 
        test_indices = checkpoint_data['test_indices']
        cv_folds = checkpoint_data['cv_folds']
        train_proteins = checkpoint_data.get('train_proteins', None)
        print("✓ Data splits loaded")
        
        # Load feature matrices from Section 2
        feature_checkpoint = progress_tracker.resume_from_checkpoint("feature_extraction")
        if feature_checkpoint:
            feature_matrices = feature_checkpoint['feature_matrices']
            metadata = feature_checkpoint['metadata']
            
            # Individual feature matrices
            aac_features = feature_matrices['aac']
            dpc_features = feature_matrices['dpc']
            tpc_features = feature_matrices['tpc']
            binary_features = feature_matrices['binary']
            physicochemical_features = feature_matrices['physicochemical']
            combined_features_matrix = feature_matrices['combined']
            
            # Metadata
            Header_array = metadata['Header']
            Position_array = metadata['Position']
            target_array = metadata['target']
            
            print("✓ Feature matrices loaded")
            print(f"  AAC: {aac_features.shape}")
            print(f"  DPC: {dpc_features.shape}")
            print(f"  TPC: {tpc_features.shape}")
            print(f"  Binary: {binary_features.shape}")
            print(f"  Physicochemical: {physicochemical_features.shape}")
            print(f"  Combined: {combined_features_matrix.shape}")
            
        else:
            raise ValueError("❌ Feature extraction checkpoint not found. Please run Section 2 first.")
    else:
        raise ValueError("❌ Data splitting checkpoint not found. Please run Section 3 first.")

# Create data splits for each feature type
feature_datasets = {
    'aac': aac_features.iloc[train_indices],
    'dpc': dpc_features.iloc[train_indices],
    'tpc': tpc_features.iloc[train_indices],
    'binary': binary_features.iloc[train_indices],
    'physicochemical': physicochemical_features.iloc[train_indices]
}

# Extract targets
y_train = target_array[train_indices]
y_val = target_array[val_indices] 
y_test = target_array[test_indices]

print(f"✓ Train samples: {len(y_train)}")
print(f"✓ Validation samples: {len(y_val)}")
print(f"✓ Test samples: {len(y_test)}")

# ============================================================================
# 4.3 Feature Transformation Functions
# ============================================================================

print("\n4.3 Setting Up Feature Transformations")
print("-" * 40)

def apply_mutual_info_selection(X_train, y_train, n_features=500):
    """Apply mutual information feature selection"""
    print(f"  Applying mutual information selection (top {n_features} features)...")
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_transformed = selector.fit_transform(X_train, y_train)
    return X_transformed, selector

def apply_pca_transformation(X_train, n_components=100):
    """Apply PCA transformation with proper standardization"""
    print(f"  Applying PCA transformation ({n_components} components)...")
    
    # CRITICAL: Standardize features before PCA (especially important for TPC)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(f"    Features standardized (mean=0, std=1)")
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_transformed = pca.fit_transform(X_scaled)
    
    var_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"    Variance explained: {var_explained:.2f}%")
    
    return X_transformed, (scaler, pca)

def apply_polynomial_features(X_train):
    """Apply polynomial feature interactions"""
    print("  Applying polynomial feature interactions (degree=2)...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_transformed = poly.fit_transform(X_train)
    print(f"    Features: {X_train.shape[1]} → {X_transformed.shape[1]}")
    return X_transformed, poly

def apply_variance_pca_hybrid(X_train, n_components=100, variance_threshold=0.01):
    """Apply variance threshold + PCA hybrid method"""
    print(f"  Applying variance threshold + PCA hybrid ({n_components} components)...")
    
    # Step 1: Variance threshold
    var_selector = VarianceThreshold(threshold=variance_threshold)
    X_var_selected = var_selector.fit_transform(X_train)
    print(f"    After variance threshold: {X_train.shape[1]} → {X_var_selected.shape[1]}")
    
    # Step 2: PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var_selected)
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_transformed = pca.fit_transform(X_scaled)
    
    var_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"    Variance explained: {var_explained:.2f}%")
    
    return X_transformed, (var_selector, scaler, pca)

# ============================================================================
# 4.4 Cross-Validation Training Function
# ============================================================================

def train_and_evaluate_cv(model, X_train, y_train, feature_name, cv_folds, train_proteins, 
                         model_name="model", use_early_stopping=True):
    """Train and evaluate model using cross-validation"""
    
    print(f"    Training {model_name} on {feature_name} features...")
    
    cv_results = []
    fold_predictions = {}
    
    for fold_idx, fold_data in enumerate(cv_folds):
        start_time = time.time()
        
        # Extract fold indices from fold_data dictionary
        train_fold_idx = fold_data['train_indices']
        val_fold_idx = fold_data['val_indices']
        
        # Convert to numpy arrays if they're lists
        if isinstance(train_fold_idx, list):
            train_fold_idx = np.array(train_fold_idx)
        if isinstance(val_fold_idx, list):
            val_fold_idx = np.array(val_fold_idx)
        
        # Get fold data
        X_fold_train = X_train[train_fold_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[train_fold_idx]
        X_fold_val = X_train[val_fold_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[val_fold_idx]
        y_fold_train = y_train[train_fold_idx]
        y_fold_val = y_train[val_fold_idx]
        
        # Train model
        fold_model = model.__class__(**model.get_params())
        
        if use_early_stopping:
            # Add early stopping and train with validation set
            if isinstance(fold_model, xgb.XGBClassifier):
                # Set early stopping for XGBoost
                fold_model.set_params(early_stopping_rounds=50)
                fold_model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
            elif CATBOOST_AVAILABLE and isinstance(fold_model, CatBoostClassifier):
                # Set early stopping for CatBoost
                fold_model.set_params(early_stopping_rounds=50)
                fold_model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=(X_fold_val, y_fold_val),
                    verbose=False
                )
            elif isinstance(fold_model, lgb.LGBMClassifier):
                # Set early stopping for LightGBM
                fold_model.set_params(early_stopping_rounds=50)
                fold_model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
            else:
                # No early stopping support, just fit normally
                fold_model.fit(X_fold_train, y_fold_train)
        else:
            # Train without early stopping
            fold_model.fit(X_fold_train, y_fold_train)
        
        # Predictions
        y_pred = fold_model.predict(X_fold_val)
        y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate metrics
        fold_results = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_fold_val, y_pred),
            'precision': precision_score(y_fold_val, y_pred, zero_division=0),
            'recall': recall_score(y_fold_val, y_pred, zero_division=0),
            'f1': f1_score(y_fold_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_fold_val, y_pred_proba),
            'mcc': matthews_corrcoef(y_fold_val, y_pred),
            'training_time': time.time() - start_time
        }
        
        cv_results.append(fold_results)
        fold_predictions[fold_idx] = {
            'y_true': y_fold_val,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"      Fold {fold_idx+1}: F1={fold_results['f1']:.4f}, AUC={fold_results['auc']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc', 'training_time']:
        values = [result[metric] for result in cv_results]
        avg_metrics[f'{metric}_mean'] = np.mean(values)
        avg_metrics[f'{metric}_std'] = np.std(values)
        avg_metrics[f'{metric}_ci'] = (
            np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values)),
            np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values))
        )
    
    return {
        'cv_results': cv_results,
        'avg_metrics': avg_metrics,
        'fold_predictions': fold_predictions
    }

# ============================================================================
# 4.5 Individual Feature Type Experiments
# ============================================================================

print("\n4.5 Individual Feature Type Experiments with Optimal Configurations")
print("-" * 40)

individual_results = {}
transformers = {}
baseline_results = {}

for feature_type, config in SELECTED_CONFIGS.items():
    print(f"\n🔬 Processing {feature_type.upper()} features...")
    print(f"   Configuration: {config['description']}")
    print(f"   Method: {config['method']}")
    print(f"   Model: {config['model']}")
    print(f"   Expected F1: {config['expected_f1']:.4f}")
    
    # Get feature data
    X_feature = feature_datasets[feature_type].copy()
    print(f"   Original shape: {X_feature.shape}")
    
    # ========================================================================
    # BASELINE: No transformation
    # ========================================================================
    print(f"\n   📊 BASELINE - No Transformation:")
    
    # Create baseline model
    model_config = MODEL_CONFIGS[config['model']]
    baseline_model = model_config['class'](**model_config['params'])
    
    # Train baseline
    baseline_result = train_and_evaluate_cv(
        baseline_model, X_feature, y_train, f"{feature_type}_baseline", 
        cv_folds, train_proteins, model_name=f"baseline_{config['model']}"
    )
    
    baseline_results[feature_type] = baseline_result
    print(f"   ✓ Baseline F1: {baseline_result['avg_metrics']['f1_mean']:.4f}±{baseline_result['avg_metrics']['f1_std']:.4f}")
    
    # ========================================================================
    # SELECTED TRANSFORMATION
    # ========================================================================
    print(f"\n   🎯 SELECTED - {config['description']}:")
    
    # Apply transformation based on selected method
    if config['method'] == 'mutual_info_500':
        X_transformed, transformer = apply_mutual_info_selection(X_feature, y_train, n_features=500)
    elif config['method'] == 'pca_100':
        X_transformed, transformer = apply_pca_transformation(X_feature, n_components=100)
    elif config['method'] == 'pca_30':
        X_transformed, transformer = apply_pca_transformation(X_feature, n_components=30)
    elif config['method'] == 'pca_50':
        X_transformed, transformer = apply_pca_transformation(X_feature, n_components=50)
    elif config['method'] == 'polynomial':
        X_transformed, transformer = apply_polynomial_features(X_feature)
    elif config['method'] == 'variance_pca_200':
        X_transformed, transformer = apply_variance_pca_hybrid(X_feature, n_components=200)
    else:
        raise ValueError(f"Unknown transformation method: {config['method']}")
    
    print(f"   Transformed shape: {X_transformed.shape}")
    
    # Create and train model
    selected_model = model_config['class'](**model_config['params'])
    
    # Train selected configuration
    selected_result = train_and_evaluate_cv(
        selected_model, X_transformed, y_train, f"{feature_type}_selected", 
        cv_folds, train_proteins, model_name=f"selected_{config['model']}"
    )
    
    individual_results[feature_type] = selected_result
    transformers[feature_type] = transformer
    
    print(f"   ✓ Selected F1: {selected_result['avg_metrics']['f1_mean']:.4f}±{selected_result['avg_metrics']['f1_std']:.4f}")
    
    # Compare with baseline
    improvement = selected_result['avg_metrics']['f1_mean'] - baseline_result['avg_metrics']['f1_mean']
    print(f"   📈 Improvement: {improvement:+.4f} ({improvement/baseline_result['avg_metrics']['f1_mean']*100:+.2f}%)")
    
    # Memory cleanup
    del X_feature, X_transformed
    gc.collect()

print("\n" + "="*60)
print("INDIVIDUAL FEATURE TYPE RESULTS SUMMARY")
print("="*60)

for feature_type, config in SELECTED_CONFIGS.items():
    baseline_f1 = baseline_results[feature_type]['avg_metrics']['f1_mean']
    selected_f1 = individual_results[feature_type]['avg_metrics']['f1_mean']
    improvement = selected_f1 - baseline_f1
    
    print(f"{feature_type.upper():>15}: Baseline={baseline_f1:.4f} → Selected={selected_f1:.4f} ({improvement:+.4f})")

# ============================================================================
# 4.6 Hierarchical Multi-Model Architecture
# ============================================================================

print("\n4.6 Hierarchical Multi-Model Architecture")
print("-" * 40)

print("Building specialized models for each feature type...")

# Train final models on full training set
specialized_models = {}
feature_predictions = {}

for feature_type, config in SELECTED_CONFIGS.items():
    print(f"\n🏗️ Training final {feature_type.upper()} specialist...")
    
    # Get and transform feature data
    X_feature = feature_datasets[feature_type].copy()
    transformer = transformers[feature_type]
    
    # Apply transformation
    if config['method'] == 'mutual_info_500':
        X_train_transformed = transformer.transform(X_feature)
    elif config['method'] in ['pca_100', 'pca_30', 'pca_50']:
        scaler, pca = transformer
        X_scaled = scaler.transform(X_feature)  # Apply standardization
        X_train_transformed = pca.transform(X_scaled)  # Then PCA
    elif config['method'] == 'polynomial':
        X_train_transformed = transformer.transform(X_feature)
    elif config['method'] == 'variance_pca_200':
        var_selector, scaler, pca = transformer
        X_var_selected = var_selector.transform(X_feature)
        X_scaled = scaler.transform(X_var_selected)  # Apply standardization
        X_train_transformed = pca.transform(X_scaled)  # Then PCA
    
    # Train final model
    model_config = MODEL_CONFIGS[config['model']]
    final_model = model_config['class'](**model_config['params'])
    
    # Use validation set for early stopping if available
    X_val_feature = eval(f"{feature_type}_features").iloc[val_indices]
    
    # Apply same transformation to validation
    if config['method'] == 'mutual_info_500':
        X_val_transformed = transformer.transform(X_val_feature)
    elif config['method'] in ['pca_100', 'pca_30', 'pca_50']:
        scaler, pca = transformer
        X_val_scaled = scaler.transform(X_val_feature)
        X_val_transformed = pca.transform(X_val_scaled)
    elif config['method'] == 'polynomial':
        X_val_transformed = transformer.transform(X_val_feature)
    elif config['method'] == 'variance_pca_200':
        var_selector, scaler, pca = transformer
        X_val_var = var_selector.transform(X_val_feature)
        X_val_scaled = scaler.transform(X_val_var)
        X_val_transformed = pca.transform(X_val_scaled)
    
    # Fit with early stopping
    if isinstance(final_model, xgb.XGBClassifier):
        final_model.set_params(early_stopping_rounds=50)
        final_model.fit(
            X_train_transformed, y_train,
            eval_set=[(X_val_transformed, y_val)],
            verbose=False
        )
    elif CATBOOST_AVAILABLE and isinstance(final_model, CatBoostClassifier):
        final_model.set_params(early_stopping_rounds=50)
        final_model.fit(
            X_train_transformed, y_train,
            eval_set=(X_val_transformed, y_val),
            verbose=False
        )
    elif isinstance(final_model, lgb.LGBMClassifier):
        final_model.set_params(early_stopping_rounds=50)
        final_model.fit(
            X_train_transformed, y_train,
            eval_set=[(X_val_transformed, y_val)],
            verbose=False
        )
    else:
        final_model.fit(X_train_transformed, y_train)
    
    # Generate predictions for meta-learning
    val_predictions = final_model.predict_proba(X_val_transformed)[:, 1]
    
    specialized_models[feature_type] = {
        'model': final_model,
        'transformer': transformer,
        'config': config,
        'val_predictions': val_predictions
    }
    
    feature_predictions[feature_type] = val_predictions
    
    print(f"   ✓ {feature_type.upper()} specialist trained")

# ============================================================================
# 4.7 Dynamic Ensemble with Performance-Based Weighting
# ============================================================================

print("\n4.7 Dynamic Ensemble with Performance-Based Weighting")
print("-" * 40)

# Calculate individual model performance on validation set
model_weights = {}
model_confidence_scores = {}

print("Calculating performance-based weights...")

for feature_type in SELECTED_CONFIGS.keys():
    val_preds = feature_predictions[feature_type]
    val_preds_binary = (val_preds > 0.5).astype(int)
    
    # Calculate performance metrics
    f1 = f1_score(y_val, val_preds_binary)
    accuracy = accuracy_score(y_val, val_preds_binary)
    auc = roc_auc_score(y_val, val_preds)
    
    # Calculate confidence (distance from decision boundary)
    confidence = np.abs(val_preds - 0.5) * 2  # Scale to [0, 1]
    avg_confidence = np.mean(confidence)
    
    # Weight = performance * confidence
    weight = (f1 * 0.5 + accuracy * 0.3 + auc * 0.2) * avg_confidence
    
    model_weights[feature_type] = weight
    model_confidence_scores[feature_type] = {
        'f1': f1,
        'accuracy': accuracy,
        'auc': auc,
        'avg_confidence': avg_confidence,
        'weight': weight
    }
    
    print(f"  {feature_type.upper():>15}: F1={f1:.4f}, Acc={accuracy:.4f}, AUC={auc:.4f}, Conf={avg_confidence:.4f} → Weight={weight:.4f}")

# Normalize weights
total_weight = sum(model_weights.values())
normalized_weights = {k: v/total_weight for k, v in model_weights.items()}

print(f"\nNormalized weights:")
for feature_type, weight in normalized_weights.items():
    print(f"  {feature_type.upper():>15}: {weight:.4f} ({weight*100:.1f}%)")

# Create ensemble predictions
print(f"\nCreating ensemble predictions...")
ensemble_val_predictions = np.zeros(len(y_val))

for feature_type, weight in normalized_weights.items():
    ensemble_val_predictions += weight * feature_predictions[feature_type]

# Evaluate ensemble on validation set
ensemble_val_binary = (ensemble_val_predictions > 0.5).astype(int)
ensemble_f1 = f1_score(y_val, ensemble_val_binary)
ensemble_accuracy = accuracy_score(y_val, ensemble_val_binary)
ensemble_auc = roc_auc_score(y_val, ensemble_val_predictions)

print(f"\n🎯 ENSEMBLE VALIDATION PERFORMANCE:")
print(f"   F1 Score: {ensemble_f1:.4f}")
print(f"   Accuracy: {ensemble_accuracy:.4f}")
print(f"   AUC: {ensemble_auc:.4f}")

# Compare with best individual model
best_individual_f1 = max(model_confidence_scores[ft]['f1'] for ft in SELECTED_CONFIGS.keys())
ensemble_improvement = ensemble_f1 - best_individual_f1

print(f"\n📈 Ensemble vs Best Individual:")
print(f"   Best Individual F1: {best_individual_f1:.4f}")
print(f"   Ensemble F1: {ensemble_f1:.4f}")
print(f"   Improvement: {ensemble_improvement:+.4f} ({ensemble_improvement/best_individual_f1*100:+.2f}%)")

# ============================================================================
# 4.8 Combined Features Model
# ============================================================================

print("\n4.8 Combined Features Model with Optimal Transformations")
print("-" * 40)

print("Creating combined feature matrix with optimal transformations...")

# Apply transformations to create combined matrix
combined_transformed_features = []
feature_names_combined = []

for feature_type, config in SELECTED_CONFIGS.items():
    print(f"  Adding {feature_type.upper()} features ({config['description']})...")
    
    # Get original features
    X_feature = feature_datasets[feature_type].copy()
    transformer = transformers[feature_type]
    
    # Apply transformation
    if config['method'] == 'mutual_info_500':
        X_transformed = transformer.transform(X_feature)
        feature_names = [f"{feature_type}_{i}" for i in range(X_transformed.shape[1])]
    elif config['method'] in ['pca_100', 'pca_30', 'pca_50']:
        scaler, pca = transformer
        X_scaled = scaler.transform(X_feature)
        X_transformed = pca.transform(X_scaled)
        n_components = X_transformed.shape[1]
        feature_names = [f"{feature_type}_pc{i}" for i in range(n_components)]
    elif config['method'] == 'polynomial':
        X_transformed = transformer.transform(X_feature)
        feature_names = [f"{feature_type}_poly_{i}" for i in range(X_transformed.shape[1])]
    elif config['method'] == 'variance_pca_200':
        var_selector, scaler, pca = transformer
        X_var_selected = var_selector.transform(X_feature)
        X_scaled = scaler.transform(X_var_selected)
        X_transformed = pca.transform(X_scaled)
        feature_names = [f"{feature_type}_vpc{i}" for i in range(X_transformed.shape[1])]
    
    combined_transformed_features.append(X_transformed)
    feature_names_combined.extend(feature_names)
    
    print(f"    Added {X_transformed.shape[1]} features")

# Combine all transformed features
X_combined_transformed = np.hstack(combined_transformed_features)
print(f"\n✓ Combined transformed matrix shape: {X_combined_transformed.shape}")

# Train combined model (use best overall model - CatBoost or XGBoost)
print(f"\nTraining combined features model...")
best_model_name = 'catboost' if CATBOOST_AVAILABLE else 'xgboost'
combined_model_config = MODEL_CONFIGS[best_model_name]
combined_model = combined_model_config['class'](**combined_model_config['params'])

# Cross-validation on combined features
combined_cv_results = train_and_evaluate_cv(
    combined_model, X_combined_transformed, y_train, "combined_optimal", 
    cv_folds, train_proteins, model_name=f"combined_{best_model_name}"
)

combined_f1 = combined_cv_results['avg_metrics']['f1_mean']
combined_std = combined_cv_results['avg_metrics']['f1_std']

print(f"\n🎯 COMBINED MODEL PERFORMANCE:")
print(f"   F1 Score: {combined_f1:.4f}±{combined_std:.4f}")
print(f"   Accuracy: {combined_cv_results['avg_metrics']['accuracy_mean']:.4f}±{combined_cv_results['avg_metrics']['accuracy_std']:.4f}")
print(f"   AUC: {combined_cv_results['avg_metrics']['auc_mean']:.4f}±{combined_cv_results['avg_metrics']['auc_std']:.4f}")

# Compare with ensemble
combined_vs_ensemble = combined_f1 - ensemble_f1
print(f"\n📊 Combined vs Ensemble:")
print(f"   Combined F1: {combined_f1:.4f}")
print(f"   Ensemble F1: {ensemble_f1:.4f}")
print(f"   Difference: {combined_vs_ensemble:+.4f}")

# ============================================================================
# 4.9 Feature Importance Analysis
# ============================================================================

print("\n4.9 Feature Importance Analysis")
print("-" * 40)

feature_importance_data = {}

# Train final combined model for importance analysis
print("Training final combined model for feature importance...")
final_combined_model = combined_model_config['class'](**combined_model_config['params'])

# Prepare validation data for early stopping
X_val_combined = []
for feature_type, config in SELECTED_CONFIGS.items():
    X_val_feature = eval(f"{feature_type}_features").iloc[val_indices]
    transformer = transformers[feature_type]
    
    if config['method'] == 'mutual_info_500':
        X_val_transformed = transformer.transform(X_val_feature)
    elif config['method'] in ['pca_100', 'pca_30', 'pca_50']:
        scaler, pca = transformer
        X_val_scaled = scaler.transform(X_val_feature)
        X_val_transformed = pca.transform(X_val_scaled)
    elif config['method'] == 'polynomial':
        X_val_transformed = transformer.transform(X_val_feature)
    elif config['method'] == 'variance_pca_200':
        var_selector, scaler, pca = transformer
        X_val_var = var_selector.transform(X_val_feature)
        X_val_scaled = scaler.transform(X_val_var)
        X_val_transformed = pca.transform(X_val_scaled)
    
    X_val_combined.append(X_val_transformed)

X_val_combined_array = np.hstack(X_val_combined)

# Fit final model
if isinstance(final_combined_model, xgb.XGBClassifier):
    final_combined_model.set_params(early_stopping_rounds=50)
    final_combined_model.fit(
        X_combined_transformed, y_train,
        eval_set=[(X_val_combined_array, y_val)],
        verbose=False
    )
elif CATBOOST_AVAILABLE and isinstance(final_combined_model, CatBoostClassifier):
    final_combined_model.set_params(early_stopping_rounds=50)
    final_combined_model.fit(
        X_combined_transformed, y_train,
        eval_set=(X_val_combined_array, y_val),
        verbose=False
    )
elif isinstance(final_combined_model, lgb.LGBMClassifier):
    final_combined_model.set_params(early_stopping_rounds=50)
    final_combined_model.fit(
        X_combined_transformed, y_train,
        eval_set=[(X_val_combined_array, y_val)],
        verbose=False
    )
else:
    final_combined_model.fit(X_combined_transformed, y_train)

# Extract feature importance
if hasattr(final_combined_model, 'feature_importances_'):
    importances = final_combined_model.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names_combined,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Group by feature type
    importance_by_type = {}
    for feature_type in SELECTED_CONFIGS.keys():
        type_mask = importance_df['feature'].str.startswith(feature_type)
        type_importance = importance_df[type_mask]['importance'].sum()
        importance_by_type[feature_type] = type_importance
    
    feature_importance_data['combined_model'] = {
        'feature_importance': importance_df,
        'importance_by_type': importance_by_type,
        'top_features': importance_df.head(20).to_dict('records')
    }
    
    print("✓ Feature importance extracted")
    print("Top 10 most important features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:>25}: {row['importance']:.6f}")
    
    print(f"\nImportance by feature type:")
    for feature_type, importance in sorted(importance_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature_type.upper():>15}: {importance:.4f} ({importance/sum(importance_by_type.values())*100:.1f}%)")

# SHAP analysis if available
if SHAP_AVAILABLE:
    print(f"\nCalculating SHAP values (this may take a while)...")
    try:
        # Use a sample for SHAP analysis to save time
        sample_size = min(1000, X_combined_transformed.shape[0])
        sample_indices = np.random.choice(X_combined_transformed.shape[0], sample_size, replace=False)
        X_sample = X_combined_transformed[sample_indices]
        
        explainer = shap.TreeExplainer(final_combined_model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        shap_df = pd.DataFrame({
            'feature': feature_names_combined,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        
        feature_importance_data['shap_analysis'] = {
            'shap_values': shap_values,
            'shap_importance': shap_df,
            'sample_indices': sample_indices
        }
        
        print("✓ SHAP analysis completed")
        print("Top 10 features by SHAP importance:")
        for i, row in shap_df.head(10).iterrows():
            print(f"  {row['feature']:>25}: {row['shap_importance']:.6f}")
            
    except Exception as e:
        print(f"⚠️ SHAP analysis failed: {e}")

# ============================================================================
# 4.10 Comprehensive Visualizations
# ============================================================================

print("\n4.10 Creating Comprehensive Visualizations")
print("-" * 40)

# Create plots directory
plot_dir = os.path.join(BASE_DIR, 'plots', 'ml_models')
os.makedirs(plot_dir, exist_ok=True)

# Set publication quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# 1. Performance Comparison - Individual vs Selected vs Ensemble
print("  Creating performance comparison plot...")

methods = ['Baseline', 'Selected', 'Ensemble', 'Combined']
f1_scores = []
auc_scores = []
feature_types = list(SELECTED_CONFIGS.keys())

# Individual baseline and selected scores
baseline_f1s = [baseline_results[ft]['avg_metrics']['f1_mean'] for ft in feature_types]
selected_f1s = [individual_results[ft]['avg_metrics']['f1_mean'] for ft in feature_types]
baseline_aucs = [baseline_results[ft]['avg_metrics']['auc_mean'] for ft in feature_types]
selected_aucs = [individual_results[ft]['avg_metrics']['auc_mean'] for ft in feature_types]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# F1 Score comparison
x = np.arange(len(feature_types))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_f1s, width, label='Baseline', alpha=0.8, color='lightblue')
bars2 = ax1.bar(x + width/2, selected_f1s, width, label='Selected Config', alpha=0.8, color='darkblue')

ax1.axhline(y=ensemble_f1, color='red', linestyle='--', alpha=0.8, label=f'Ensemble ({ensemble_f1:.4f})')
ax1.axhline(y=combined_f1, color='green', linestyle='--', alpha=0.8, label=f'Combined ({combined_f1:.4f})')

ax1.set_xlabel('Feature Type')
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 Score Comparison: Individual vs Ensemble vs Combined')
ax1.set_xticks(x)
ax1.set_xticklabels([ft.upper() for ft in feature_types], rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# AUC comparison
bars3 = ax2.bar(x - width/2, baseline_aucs, width, label='Baseline', alpha=0.8, color='lightcoral')
bars4 = ax2.bar(x + width/2, selected_aucs, width, label='Selected Config', alpha=0.8, color='darkred')

ax2.axhline(y=ensemble_auc, color='red', linestyle='--', alpha=0.8, label=f'Ensemble ({ensemble_auc:.4f})')
ax2.axhline(y=combined_cv_results['avg_metrics']['auc_mean'], color='green', linestyle='--', alpha=0.8, 
           label=f'Combined ({combined_cv_results["avg_metrics"]["auc_mean"]:.4f})')

ax2.set_xlabel('Feature Type')
ax2.set_ylabel('AUC Score')
ax2.set_title('AUC Score Comparison: Individual vs Ensemble vs Combined')
ax2.set_xticks(x)
ax2.set_xticklabels([ft.upper() for ft in feature_types], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 2. Model Weights Visualization
print("  Creating model weights visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart of ensemble weights
wedges, texts, autotexts = ax1.pie(
    normalized_weights.values(),
    labels=[ft.upper() for ft in normalized_weights.keys()],
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Set3(np.linspace(0, 1, len(normalized_weights)))
)
ax1.set_title('Ensemble Model Weights\n(Performance-Based)')

# Bar chart of individual performance components
components = ['F1', 'Accuracy', 'AUC', 'Confidence']
feature_types_list = list(model_confidence_scores.keys())
n_features = len(feature_types_list)

x = np.arange(n_features)
width = 0.2

for i, component in enumerate(components):
    values = []
    for ft in feature_types_list:
        if component.lower() == 'confidence':
            values.append(model_confidence_scores[ft]['avg_confidence'])
        else:
            values.append(model_confidence_scores[ft][component.lower()])
    
    ax2.bar(x + i*width, values, width, label=component, alpha=0.8)

ax2.set_xlabel('Feature Type')
ax2.set_ylabel('Score')
ax2.set_title('Performance Components for Weight Calculation')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels([ft.upper() for ft in feature_types_list], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'ensemble_weights.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 3. Feature Importance Plot
if 'combined_model' in feature_importance_data:
    print("  Creating feature importance visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Top 20 individual features
    top_features = feature_importance_data['combined_model']['feature_importance'].head(20)
    
    bars = ax1.barh(range(len(top_features)), top_features['importance'], alpha=0.8)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=8)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Top 20 Most Important Features\n(Combined Model)')
    ax1.grid(True, alpha=0.3)
    
    # Color bars by feature type
    colors = plt.cm.Set3(np.linspace(0, 1, len(SELECTED_CONFIGS)))
    color_map = {ft: colors[i] for i, ft in enumerate(SELECTED_CONFIGS.keys())}
    
    for i, (feature, bar) in enumerate(zip(top_features['feature'], bars)):
        for ft in SELECTED_CONFIGS.keys():
            if feature.startswith(ft):
                bar.set_color(color_map[ft])
                break
    
    # Feature type importance
    importance_by_type = feature_importance_data['combined_model']['importance_by_type']
    
    wedges, texts, autotexts = ax2.pie(
        importance_by_type.values(),
        labels=[ft.upper() for ft in importance_by_type.keys()],
        autopct='%1.1f%%',
        startangle=90,
        colors=[color_map[ft] for ft in importance_by_type.keys()]
    )
    ax2.set_title('Feature Importance by Type\n(Combined Model)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# 4. Training Performance Matrix
print("  Creating training performance matrix...")

# Create performance matrix
performance_data = []

for feature_type, config in SELECTED_CONFIGS.items():
    baseline_metrics = baseline_results[feature_type]['avg_metrics']
    selected_metrics = individual_results[feature_type]['avg_metrics']
    
    performance_data.append({
        'Feature Type': feature_type.upper(),
        'Method': config['description'],
        'Baseline F1': baseline_metrics['f1_mean'],
        'Selected F1': selected_metrics['f1_mean'],
        'Improvement': selected_metrics['f1_mean'] - baseline_metrics['f1_mean'],
        'Baseline AUC': baseline_metrics['auc_mean'],
        'Selected AUC': selected_metrics['auc_mean'],
        'Training Time (s)': selected_metrics['training_time_mean']
    })

# Add ensemble and combined
performance_data.append({
    'Feature Type': 'ENSEMBLE',
    'Method': 'Performance-Weighted',
    'Baseline F1': best_individual_f1,
    'Selected F1': ensemble_f1,
    'Improvement': ensemble_improvement,
    'Baseline AUC': max(model_confidence_scores[ft]['auc'] for ft in SELECTED_CONFIGS.keys()),
    'Selected AUC': ensemble_auc,
    'Training Time (s)': 0  # Ensemble doesn't require separate training
})

performance_data.append({
    'Feature Type': 'COMBINED',
    'Method': 'Optimal Transforms',
    'Baseline F1': np.mean([baseline_results[ft]['avg_metrics']['f1_mean'] for ft in SELECTED_CONFIGS.keys()]),
    'Selected F1': combined_f1,
    'Improvement': combined_f1 - np.mean([baseline_results[ft]['avg_metrics']['f1_mean'] for ft in SELECTED_CONFIGS.keys()]),
    'Baseline AUC': np.mean([baseline_results[ft]['avg_metrics']['auc_mean'] for ft in SELECTED_CONFIGS.keys()]),
    'Selected AUC': combined_cv_results['avg_metrics']['auc_mean'],
    'Training Time (s)': combined_cv_results['avg_metrics']['training_time_mean']
})

performance_df = pd.DataFrame(performance_data)

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Select numeric columns for heatmap
numeric_cols = ['Baseline F1', 'Selected F1', 'Improvement', 'Baseline AUC', 'Selected AUC']
heatmap_data = performance_df.set_index('Feature Type')[numeric_cols]

sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlBu_r', center=0, ax=ax, cbar_kws={'label': 'Score'})
ax.set_title('Performance Matrix: All Methods')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'performance_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ All visualizations saved to plots/ml_models/")

# ============================================================================
# 4.11 Save Comprehensive Checkpoint
# ============================================================================

print("\n4.11 Saving Comprehensive Checkpoint")
print("-" * 40)

# Prepare final test predictions for each model
test_predictions = {}

for feature_type, config in SELECTED_CONFIGS.items():
    print(f"  Generating test predictions for {feature_type.upper()}...")
    
    # Get test features
    X_test_feature = eval(f"{feature_type}_features").iloc[test_indices]
    transformer = transformers[feature_type]
    
    # Apply transformation
    if config['method'] == 'mutual_info_500':
        X_test_transformed = transformer.transform(X_test_feature)
    elif config['method'] in ['pca_100', 'pca_30', 'pca_50']:
        scaler, pca = transformer
        X_test_scaled = scaler.transform(X_test_feature)
        X_test_transformed = pca.transform(X_test_scaled)
    elif config['method'] == 'polynomial':
        X_test_transformed = transformer.transform(X_test_feature)
    elif config['method'] == 'variance_pca_200':
        var_selector, scaler, pca = transformer
        X_test_var = var_selector.transform(X_test_feature)
        X_test_scaled = scaler.transform(X_test_var)
        X_test_transformed = pca.transform(X_test_scaled)
    
    # Generate predictions
    model = specialized_models[feature_type]['model']
    test_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    test_pred_binary = (test_pred_proba > 0.5).astype(int)
    
    test_predictions[feature_type] = {
        'pred_proba': test_pred_proba,
        'pred_binary': test_pred_binary
    }

# Generate ensemble test predictions
ensemble_test_pred = np.zeros(len(y_test))
for feature_type, weight in normalized_weights.items():
    ensemble_test_pred += weight * test_predictions[feature_type]['pred_proba']

ensemble_test_binary = (ensemble_test_pred > 0.5).astype(int)

# Generate combined model test predictions
X_test_combined = []
for feature_type, config in SELECTED_CONFIGS.items():
    X_test_feature = eval(f"{feature_type}_features").iloc[test_indices]
    transformer = transformers[feature_type]
    
    if config['method'] == 'mutual_info_500':
        X_test_transformed = transformer.transform(X_test_feature)
    elif config['method'] in ['pca_100', 'pca_30', 'pca_50']:
        scaler, pca = transformer
        X_test_scaled = scaler.transform(X_test_feature)
        X_test_transformed = pca.transform(X_test_scaled)
    elif config['method'] == 'polynomial':
        X_test_transformed = transformer.transform(X_test_feature)
    elif config['method'] == 'variance_pca_200':
        var_selector, scaler, pca = transformer
        X_test_var = var_selector.transform(X_test_feature)
        X_test_scaled = scaler.transform(X_test_var)
        X_test_transformed = pca.transform(X_test_scaled)
    
    X_test_combined.append(X_test_transformed)

X_test_combined_array = np.hstack(X_test_combined)
combined_test_pred_proba = final_combined_model.predict_proba(X_test_combined_array)[:, 1]
combined_test_pred_binary = (combined_test_pred_proba > 0.5).astype(int)

# Calculate final test metrics
final_test_results = {}

# Individual models
for feature_type in SELECTED_CONFIGS.keys():
    pred_binary = test_predictions[feature_type]['pred_binary']
    pred_proba = test_predictions[feature_type]['pred_proba']
    
    final_test_results[feature_type] = {
        'accuracy': accuracy_score(y_test, pred_binary),
        'precision': precision_score(y_test, pred_binary, zero_division=0),
        'recall': recall_score(y_test, pred_binary, zero_division=0),
        'f1': f1_score(y_test, pred_binary, zero_division=0),
        'auc': roc_auc_score(y_test, pred_proba),
        'mcc': matthews_corrcoef(y_test, pred_binary)
    }

# Ensemble
final_test_results['ensemble'] = {
    'accuracy': accuracy_score(y_test, ensemble_test_binary),
    'precision': precision_score(y_test, ensemble_test_binary, zero_division=0),
    'recall': recall_score(y_test, ensemble_test_binary, zero_division=0),
    'f1': f1_score(y_test, ensemble_test_binary, zero_division=0),
    'auc': roc_auc_score(y_test, ensemble_test_pred),
    'mcc': matthews_corrcoef(y_test, ensemble_test_binary)
}

# Combined
final_test_results['combined'] = {
    'accuracy': accuracy_score(y_test, combined_test_pred_binary),
    'precision': precision_score(y_test, combined_test_pred_binary, zero_division=0),
    'recall': recall_score(y_test, combined_test_pred_binary, zero_division=0),
    'f1': f1_score(y_test, combined_test_pred_binary, zero_division=0),
    'auc': roc_auc_score(y_test, combined_test_pred_proba),
    'mcc': matthews_corrcoef(y_test, combined_test_pred_binary)
}

print("\n🎯 FINAL TEST SET RESULTS:")
print("="*50)

for model_name, metrics in final_test_results.items():
    print(f"{model_name.upper():>12}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

# Save comprehensive checkpoint
checkpoint_data = {
    'selected_configs': SELECTED_CONFIGS,
    'baseline_results': baseline_results,
    'individual_results': individual_results,
    'transformers': transformers,
    'specialized_models': specialized_models,
    'ensemble_weights': normalized_weights,
    'model_confidence_scores': model_confidence_scores,
    'combined_cv_results': combined_cv_results,
    'final_combined_model': final_combined_model,
    'feature_importance_data': feature_importance_data,
    'test_predictions': test_predictions,
    'final_test_results': final_test_results,
    'ensemble_test_predictions': {
        'pred_proba': ensemble_test_pred,
        'pred_binary': ensemble_test_binary
    },
    'combined_test_predictions': {
        'pred_proba': combined_test_pred_proba,
        'pred_binary': combined_test_pred_binary
    },
    'performance_summary': performance_df
}

# Add metadata
metadata = {
    'n_feature_types': len(SELECTED_CONFIGS),
    'total_selected_features': sum(config['features'] for config in SELECTED_CONFIGS.values()),
    'best_individual_f1': max(final_test_results[ft]['f1'] for ft in SELECTED_CONFIGS.keys()),
    'ensemble_f1': final_test_results['ensemble']['f1'],
    'combined_f1': final_test_results['combined']['f1'],
    'best_overall_method': max(final_test_results.items(), key=lambda x: x[1]['f1'])[0],
    'training_completed': time.strftime('%Y-%m-%d %H:%M:%S')
}

progress_tracker.mark_completed(
    "ml_models_enhanced",
    metadata=metadata,
    checkpoint_data=checkpoint_data
)

print(f"\n✅ Section 4 completed successfully!")
print(f"   Best individual model: {max(final_test_results.items(), key=lambda x: x[1]['f1'] if x[0] in SELECTED_CONFIGS else 0)[0].upper()} (F1={max(final_test_results[ft]['f1'] for ft in SELECTED_CONFIGS.keys()):.4f})")
print(f"   Ensemble F1: {final_test_results['ensemble']['f1']:.4f}")
print(f"   Combined F1: {final_test_results['combined']['f1']:.4f}")
print(f"   Best overall: {metadata['best_overall_method'].upper()} (F1={final_test_results[metadata['best_overall_method']]['f1']:.4f})")
print(f"   Checkpoint saved with {len(checkpoint_data)} components")

# Final memory cleanup
del X_combined_transformed, X_val_combined_array, X_test_combined_array
for ft in SELECTED_CONFIGS.keys():
    del feature_datasets[ft]
gc.collect()

print(f"\n🎉 Enhanced Section 4 complete! All models trained and evaluated.")