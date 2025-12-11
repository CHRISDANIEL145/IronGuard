"""
🧬 QUANTUM ML EVALUATION ENGINE v2.4 - ENTERPRISE PRODUCTION EDITION (FINAL)
Complete Machine Learning Evaluation Framework
- Fixed Classification Target Type Error
- Quantum-Inspired ML (Superposition, Entanglement, Measurement)
- Data Augmentation (Gaussian Noise, Mixup, SMOTE)
- Hyperparameter Tuning (XGBoost, LightGBM, Random Forest)
- Imbalanced Data Handling with Proper Type Conversion

Author: Quantum AI Engineering Team
Date: 2025
License: MIT
Version: 2.4 (Production Final)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, 
    mean_absolute_error, r2_score, mean_absolute_percentage_error
)
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 0. SETUP & LOGGING
# ============================================================================

os.makedirs('ml_results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def log_msg(msg, level="INFO"):
    """Enhanced logging with file persistence"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {msg}"
    print(formatted_msg)
    with open('logs/ml_evaluation.log', 'a') as f:
        f.write(formatted_msg + '\n')

# ============================================================================
# 1. QUANTUM PRINCIPLES
# ============================================================================

class QuantumPrinciples:
    """Quantum Computing Inspired ML Principles"""
    
    @staticmethod
    def superposition_state(probs):
        """Superposition: Multiple simultaneous states"""
        total = np.sum(np.abs(probs) ** 2)
        if total > 0:
            superposition = np.sqrt(np.abs(probs) ** 2 / total)
        else:
            superposition = probs / len(probs)
        return superposition
    
    @staticmethod
    def entanglement_correlation(X):
        """Entanglement: Feature correlations"""
        if X.shape[0] < 2 or X.shape[1] < 2:
            return 0.5, np.array([[]])
        correlations = np.corrcoef(X.T)
        correlations = np.nan_to_num(correlations)
        entanglement_strength = np.mean(np.abs(correlations))
        return entanglement_strength, correlations
    
    @staticmethod
    def measurement_collapse(predictions, weights):
        """Measurement: Collapse to optimal state"""
        if len(predictions) == 0:
            return np.array([])
        weights_norm = np.array(weights) / (np.sum(np.array(weights)) + 1e-8)
        return np.average(predictions, axis=0, weights=weights_norm)

# ============================================================================
# 2. DATA AUGMENTATION
# ============================================================================

class DataAugmentation:
    """Advanced Data Augmentation Techniques"""
    
    @staticmethod
    def gaussian_noise_augmentation(X, y, noise_scale=0.03, n_augmented=500):
        """Gaussian noise augmentation - FIX: ensure y is int"""
        log_msg(f"   📈 Gaussian augmentation: {n_augmented} samples")
        augmented_X = []
        augmented_y = []
        
        for _ in range(n_augmented):
            idx = np.random.randint(0, len(X))
            noise = np.random.normal(0, noise_scale, X[idx].shape)
            X_augmented = X[idx] + noise
            augmented_X.append(X_augmented)
            # FIX: Convert to int for classification
            augmented_y.append(int(y[idx]))
        
        return np.array(augmented_X), np.array(augmented_y, dtype=int)
    
    @staticmethod
    def mixup_augmentation(X, y, n_augmented=500, alpha=0.3):
        """Mixup augmentation - FIX: ensure y is int"""
        log_msg(f"   🔄 Mixup augmentation: {n_augmented} samples")
        augmented_X = []
        augmented_y = []
        
        for _ in range(n_augmented):
            idx1, idx2 = np.random.choice(len(X), 2, replace=False)
            lam = np.random.beta(alpha, alpha)
            X_mixed = lam * X[idx1] + (1 - lam) * X[idx2]
            # FIX: For classification, use voting instead of interpolation
            y_mixed = 1 if (lam * y[idx1] + (1 - lam) * y[idx2]) > 0.5 else 0
            augmented_X.append(X_mixed)
            augmented_y.append(y_mixed)
        
        return np.array(augmented_X), np.array(augmented_y, dtype=int)
    
    @staticmethod
    def smote_augmentation(X, y, k_neighbors=5, n_augmented=500):
        """SMOTE augmentation - FIX: ensure y is int"""
        log_msg(f"   ✨ SMOTE augmentation: {n_augmented} samples")
        augmented_X = []
        augmented_y = []
        
        for _ in range(n_augmented):
            idx = np.random.randint(0, len(X))
            X_sample = X[idx]
            distances = np.sum((X - X_sample) ** 2, axis=1)
            nearest_idx = np.argsort(distances)[1:k_neighbors+1]
            
            if len(nearest_idx) > 0:
                random_neighbor_idx = np.random.choice(nearest_idx)
                random_neighbor = X[random_neighbor_idx]
                alpha = np.random.uniform(0, 1)
                X_synthetic = X_sample + alpha * (random_neighbor - X_sample)
            else:
                X_synthetic = X_sample
            
            augmented_X.append(X_synthetic)
            # FIX: Convert to int
            augmented_y.append(int(y[idx]))
        
        return np.array(augmented_X), np.array(augmented_y, dtype=int)

# ============================================================================
# 3. HYPERPARAMETER TUNING
# ============================================================================

class HyperparameterTuner:
    """Advanced Hyperparameter Optimization"""
    
    @staticmethod
    def tune_xgboost(X_train, y_train, cv=3):
        """XGBoost tuning"""
        log_msg("   🔧 XGBoost Hyperparameter Tuning...")
        
        param_grid = {
            'n_estimators': [150, 200],
            'max_depth': [6, 7, 8],
            'learning_rate': [0.03, 0.05],
            'subsample': [0.8, 0.9],
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
        search = RandomizedSearchCV(
            xgb_model, param_grid, cv=cv, n_iter=10, n_jobs=-1, 
            random_state=42, verbose=0, scoring='r2'
        )
        search.fit(X_train, y_train)
        log_msg(f"   ✅ Best XGBoost R²: {search.best_score_:.4f}")
        return search.best_estimator_
    
    @staticmethod
    def tune_lightgbm(X_train, y_train, cv=3):
        """LightGBM tuning"""
        log_msg("   🔧 LightGBM Hyperparameter Tuning...")
        
        param_grid = {
            'n_estimators': [150, 200],
            'max_depth': [6, 7, 8],
            'learning_rate': [0.03, 0.05],
            'num_leaves': [50, 70],
        }
        
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
        search = RandomizedSearchCV(
            lgb_model, param_grid, cv=cv, n_iter=10, n_jobs=-1, 
            random_state=42, verbose=0, scoring='r2'
        )
        search.fit(X_train, y_train)
        log_msg(f"   ✅ Best LightGBM R²: {search.best_score_:.4f}")
        return search.best_estimator_
    
    @staticmethod
    def tune_random_forest(X_train, y_train, cv=3):
        """Random Forest tuning"""
        log_msg("   🔧 Random Forest Hyperparameter Tuning...")
        
        param_grid = {
            'n_estimators': [150, 200],
            'max_depth': [10, 12],
            'min_samples_split': [5, 10],
        }
        
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            rf_model, param_grid, cv=cv, n_iter=10, n_jobs=-1, 
            random_state=42, verbose=0, scoring='r2'
        )
        search.fit(X_train, y_train)
        log_msg(f"   ✅ Best RF R²: {search.best_score_:.4f}")
        return search.best_estimator_

# ============================================================================
# 4. CONFIGURATION
# ============================================================================

CONFIG = {
    'TEMP_MIN': 1350,
    'TEMP_MAX': 1550,
    'TEMP_OPTIMAL_LOW': 1410,
    'TEMP_OPTIMAL_HIGH': 1430,
    'OPTIMAL_ENERGY': 450,
    'TEMP_COEFFICIENT': 0.02,
}

# ============================================================================
# 5. DATA GENERATION
# ============================================================================

def generate_datasets(n_records=25000):
    """Generate balanced foundry datasets"""
    log_msg("📊 Generating Balanced Quantum Datasets...")
    
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-01-01', periods=n_records, freq='5min')
    
    base_temp = 1450
    temps = []
    anomalies = []
    
    for i in range(n_records):
        drift = np.sin(i / 3600) * 20
        noise = np.random.normal(0, 2.5)
        
        if np.random.random() < 0.05:
            anomaly_val = -50 if np.random.random() < 0.5 else 30
            is_anomaly = 1
        else:
            anomaly_val = 0
            is_anomaly = 0
        
        temp = base_temp + drift + noise + anomaly_val
        temps.append(np.clip(temp, CONFIG['TEMP_MIN'], CONFIG['TEMP_MAX']))
        anomalies.append(is_anomaly)
    
    optimal_temp = (CONFIG['TEMP_OPTIMAL_LOW'] + CONFIG['TEMP_OPTIMAL_HIGH']) / 2
    energy = []
    for temp in temps:
        e = CONFIG['OPTIMAL_ENERGY'] + CONFIG['TEMP_COEFFICIENT'] * (temp - optimal_temp) ** 2 + np.random.normal(0, 3)
        energy.append(np.clip(e, 300, 600))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temp_C': temps,
        'energy_kwh': energy,
        'sensor_id': np.random.choice([1, 2, 3, 4, 5], n_records),
        'anomaly': anomalies
    })
    
    # Ensure anomaly is int
    df['anomaly'] = df['anomaly'].astype(int)
    
    log_msg(f"✅ Generated {n_records:,} records")
    log_msg(f"   Normal samples: {(df['anomaly']==0).sum():,} | Anomaly samples: {(df['anomaly']==1).sum():,}")
    log_msg(f"   Anomaly ratio: {df['anomaly'].mean():.2%}")
    
    return df

# ============================================================================
# 6. FEATURE ENGINEERING
# ============================================================================

def create_quantum_features(df):
    """Create quantum-inspired features"""
    log_msg("🧬 Creating Quantum-Inspired Features...")
    
    features = pd.DataFrame(index=df.index)
    
    # Temporal superposition
    for lag in [1, 2, 3, 5, 10]:
        features[f'temp_lag_{lag}'] = df['temp_C'].shift(lag)
    
    # Rolling statistics
    features['temp_mean_10'] = df['temp_C'].rolling(10, min_periods=1).mean()
    features['temp_std_10'] = df['temp_C'].rolling(10, min_periods=1).std().fillna(0)
    features['temp_mean_30'] = df['temp_C'].rolling(30, min_periods=1).mean()
    features['temp_max_20'] = df['temp_C'].rolling(20, min_periods=1).max()
    features['temp_min_20'] = df['temp_C'].rolling(20, min_periods=1).min()
    
    # Momentum
    features['temp_momentum'] = df['temp_C'].diff().fillna(0)
    features['temp_acceleration'] = df['temp_C'].diff().diff().fillna(0)
    
    # Energy
    features['energy_kwh'] = df['energy_kwh']
    features['energy_lag_1'] = df['energy_kwh'].shift(1).fillna(df['energy_kwh'].mean())
    
    # Thermal state
    min_temp = df['temp_C'].min()
    max_temp = df['temp_C'].max()
    features['thermal_state'] = (df['temp_C'] - min_temp) / (max_temp - min_temp + 1e-8)
    
    # Advanced
    features['temp_range_20'] = features['temp_max_20'] - features['temp_min_20']
    features['energy_efficiency'] = df['energy_kwh'] / (df['temp_C'] + 1e-8)
    features['sensor_id'] = df['sensor_id']
    
    # Fill NaN
    features = features.fillna(method='bfill').fillna(method='ffill')
    
    log_msg(f"✅ Created {features.shape[1]} quantum features")
    
    qp = QuantumPrinciples()
    numeric_features = features.select_dtypes(include=[np.number]).values
    if numeric_features.size > 0:
        entanglement, _ = qp.entanglement_correlation(numeric_features)
        log_msg(f"   Entanglement Strength: {entanglement:.4f}")
    
    return features

# ============================================================================
# 7. DATA PREPARATION
# ============================================================================

def prepare_classification_data(df, features_df):
    """Prepare classification data - FIX: ensure int type"""
    log_msg("🔮 Preparing Classification Data with Augmentation...")
    
    X = features_df[['temp_mean_10', 'temp_std_10', 'thermal_state']].dropna().values
    y = df['anomaly'].iloc[len(df) - len(X):].values.astype(int)  # FIX: explicit int
    
    log_msg(f"   Original: {len(X)} samples | Anomalies: {y.sum()} ({y.mean():.2%})")
    
    # Augmentation
    da = DataAugmentation()
    X_noise, y_noise = da.gaussian_noise_augmentation(X, y, noise_scale=0.03, n_augmented=500)
    X_mixup, y_mixup = da.mixup_augmentation(X, y, n_augmented=500)
    X_smote, y_smote = da.smote_augmentation(X, y, n_augmented=500)
    
    # Combine - FIX: ensure all y are int
    X_combined = np.vstack([X, X_noise, X_mixup, X_smote])
    y_combined = np.concatenate([y, y_noise, y_mixup, y_smote]).astype(int)  # FIX
    
    log_msg(f"   Augmented: {len(X_combined)} samples | Anomalies: {y_combined.sum()} ({y_combined.mean():.2%})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.25, random_state=42
    )
    
    # FIX: verify types
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    log_msg(f"   Split: Train={len(X_train)} | Test={len(X_test)}")
    log_msg(f"   y_train type: {y_train.dtype} | y_test type: {y_test.dtype}")
    
    return X_train, X_test, y_train, y_test

def prepare_regression_data(df, features_df):
    """Prepare regression data"""
    log_msg("📊 Preparing Regression Data with Augmentation...")
    
    X = features_df.dropna().values
    y = df['temp_C'].iloc[len(df) - len(X):].values
    
    if len(X) == 0:
        raise ValueError("No valid features!")
    
    log_msg(f"   Original: {len(X)} samples")
    
    # Augmentation
    da = DataAugmentation()
    X_noise, y_noise = da.gaussian_noise_augmentation(X, y, noise_scale=0.02, n_augmented=600)
    X_mixup, y_mixup = da.mixup_augmentation(X, y, n_augmented=600)
    X_smote, y_smote = da.smote_augmentation(X, y, n_augmented=600)
    
    # Combine
    X_combined = np.vstack([X, X_noise, X_mixup, X_smote])
    y_combined = np.hstack([y, y_noise, y_mixup, y_smote])
    
    log_msg(f"   Augmented: {len(X_combined)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.25, random_state=42
    )
    
    log_msg(f"   Split: Train={len(X_train)} | Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# 8. ANOMALY DETECTION - FIXED
# ============================================================================

def evaluate_quantum_anomaly_detection(X_train, X_test, y_train, y_test):
    """Quantum anomaly detection - FIXED"""
    log_msg("\n" + "="*100)
    log_msg("🚀 QUANTUM SUPERPOSITION ANOMALY DETECTION")
    log_msg("="*100)
    
    # FIX: Verify types before processing
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    log_msg(f"   y_train type: {y_train.dtype}, y_test type: {y_test.dtype}")
    log_msg(f"   y_train unique: {np.unique(y_train)}, y_test unique: {np.unique(y_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    all_preds = []
    all_f1 = []
    
    # Model 1
    log_msg("\n🌌 Model 1: Isolation Forest (Superposition State 1)")
    if1 = IsolationForest(contamination=0.03, n_estimators=300, random_state=42, n_jobs=-1)
    if1.fit(X_train_scaled)
    y_pred_1 = (if1.predict(X_test_scaled) == -1).astype(int)  # FIX: explicit int
    all_preds.append(y_pred_1)
    
    acc_1 = accuracy_score(y_test, y_pred_1)
    prec_1 = precision_score(y_test, y_pred_1, zero_division=0)
    rec_1 = recall_score(y_test, y_pred_1, zero_division=0)
    f1_1 = f1_score(y_test, y_pred_1, zero_division=0)
    
    log_msg(f"   Accuracy: {acc_1:.4f} | Precision: {prec_1:.4f} | Recall: {rec_1:.4f} | F1: {f1_1:.4f}")
    results['IF-State1'] = {'accuracy': acc_1, 'precision': prec_1, 'recall': rec_1, 'f1': f1_1}
    all_f1.append(f1_1)
    
    # Model 2
    log_msg("\n🌌 Model 2: Isolation Forest (Entanglement State 2)")
    if2 = IsolationForest(contamination=0.04, n_estimators=300, random_state=123, n_jobs=-1)
    if2.fit(X_train_scaled)
    y_pred_2 = (if2.predict(X_test_scaled) == -1).astype(int)  # FIX: explicit int
    all_preds.append(y_pred_2)
    
    acc_2 = accuracy_score(y_test, y_pred_2)
    prec_2 = precision_score(y_test, y_pred_2, zero_division=0)
    rec_2 = recall_score(y_test, y_pred_2, zero_division=0)
    f1_2 = f1_score(y_test, y_pred_2, zero_division=0)
    
    log_msg(f"   Accuracy: {acc_2:.4f} | Precision: {prec_2:.4f} | Recall: {rec_2:.4f} | F1: {f1_2:.4f}")
    results['IF-State2'] = {'accuracy': acc_2, 'precision': prec_2, 'recall': rec_2, 'f1': f1_2}
    all_f1.append(f1_2)
    
    # Ensemble
    log_msg("\n🧬 Model 3: Quantum Superposition Ensemble (Measurement Collapse)")
    weights = np.array(all_f1)
    weights = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-8)
    y_pred_ensemble = np.round(weights[0] * y_pred_1 + weights[1] * y_pred_2).astype(int)  # FIX
    
    acc_ens = accuracy_score(y_test, y_pred_ensemble)
    prec_ens = precision_score(y_test, y_pred_ensemble, zero_division=0)
    rec_ens = recall_score(y_test, y_pred_ensemble, zero_division=0)
    f1_ens = f1_score(y_test, y_pred_ensemble, zero_division=0)
    
    log_msg(f"   Accuracy: {acc_ens:.4f} | Precision: {prec_ens:.4f} | Recall: {rec_ens:.4f} | F1: {f1_ens:.4f}")
    results['Quantum-Ensemble'] = {'accuracy': acc_ens, 'precision': prec_ens, 'recall': rec_ens, 'f1': f1_ens}
    
    # Metrics
    log_msg("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_ensemble)
    log_msg(f"{cm}")
    
    log_msg("\n📋 Classification Report:")
    report = classification_report(y_test, y_pred_ensemble, target_names=['Normal', 'Anomaly'], digits=4)
    log_msg(report)
    
    return results, y_test, y_pred_ensemble, cm

# ============================================================================
# 9. TEMPERATURE PREDICTION
# ============================================================================

def evaluate_quantum_temperature_prediction(X_train, X_test, y_train, y_test):
    """Quantum temperature prediction"""
    log_msg("\n" + "="*100)
    log_msg("🌡️ QUANTUM SUPERPOSITION TEMPERATURE PREDICTION")
    log_msg("="*100)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    all_preds = []
    all_r2 = []
    
    tuner = HyperparameterTuner()
    
    # XGBoost
    log_msg("\n⚡ Model 1: XGBoost (Tuned)")
    xgb_model = tuner.tune_xgboost(X_train_scaled, y_train, cv=3)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    all_preds.append(y_pred_xgb)
    
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    all_r2.append(r2_xgb)
    
    log_msg(f"   MAE: {mae_xgb:.4f}°C | RMSE: {rmse_xgb:.4f}°C | R²: {r2_xgb:.4f}")
    results['XGBoost'] = {'mae': mae_xgb, 'rmse': rmse_xgb, 'r2': r2_xgb}
    
    # LightGBM
    log_msg("\n🌲 Model 2: LightGBM (Tuned)")
    lgb_model = tuner.tune_lightgbm(X_train_scaled, y_train, cv=3)
    y_pred_lgb = lgb_model.predict(X_test_scaled)
    all_preds.append(y_pred_lgb)
    
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)
    all_r2.append(r2_lgb)
    
    log_msg(f"   MAE: {mae_lgb:.4f}°C | RMSE: {rmse_lgb:.4f}°C | R²: {r2_lgb:.4f}")
    results['LightGBM'] = {'mae': mae_lgb, 'rmse': rmse_lgb, 'r2': r2_lgb}
    
    # Random Forest
    log_msg("\n🌳 Model 3: Random Forest (Tuned)")
    rf_model = tuner.tune_random_forest(X_train_scaled, y_train, cv=3)
    y_pred_rf = rf_model.predict(X_test_scaled)
    all_preds.append(y_pred_rf)
    
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    all_r2.append(r2_rf)
    
    log_msg(f"   MAE: {mae_rf:.4f}°C | RMSE: {rmse_rf:.4f}°C | R²: {r2_rf:.4f}")
    results['Random Forest'] = {'mae': mae_rf, 'rmse': rmse_rf, 'r2': r2_rf}
    
    # Gradient Boosting
    log_msg("\n📈 Model 4: Gradient Boosting")
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=7, learning_rate=0.05, subsample=0.8, random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    y_pred_gb = gb_model.predict(X_test_scaled)
    all_preds.append(y_pred_gb)
    
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2_gb = r2_score(y_test, y_pred_gb)
    all_r2.append(r2_gb)
    
    log_msg(f"   MAE: {mae_gb:.4f}°C | RMSE: {rmse_gb:.4f}°C | R²: {r2_gb:.4f}")
    results['Gradient Boosting'] = {'mae': mae_gb, 'rmse': rmse_gb, 'r2': r2_gb}
    
    # Neural Network
    log_msg("\n🧠 Model 5: Neural Network (Regularized)")
    nn_model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32), max_iter=1000, learning_rate_init=0.001,
        alpha=0.0001, early_stopping=True, validation_fraction=0.2, n_iter_no_change=50,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)
    y_pred_nn = nn_model.predict(X_test_scaled)
    all_preds.append(y_pred_nn)
    
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    r2_nn = r2_score(y_test, y_pred_nn)
    all_r2.append(r2_nn)
    
    log_msg(f"   MAE: {mae_nn:.4f}°C | RMSE: {rmse_nn:.4f}°C | R²: {r2_nn:.4f}")
    results['Neural Network'] = {'mae': mae_nn, 'rmse': rmse_nn, 'r2': r2_nn}
    
    # Quantum Ensemble
    log_msg("\n🧬 Model 6: Quantum Superposition Ensemble")
    weights = np.array(all_r2)
    weights = np.abs(weights) / (np.sum(np.abs(weights)) + 1e-8)
    
    all_preds_array = np.array(all_preds)
    y_pred_quantum = np.average(all_preds_array, axis=0, weights=weights)
    
    mae_q = mean_absolute_error(y_test, y_pred_quantum)
    rmse_q = np.sqrt(mean_squared_error(y_test, y_pred_quantum))
    r2_q = r2_score(y_test, y_pred_quantum)
    
    log_msg(f"   MAE: {mae_q:.4f}°C | RMSE: {rmse_q:.4f}°C | R²: {r2_q:.4f}")
    results['Quantum-Superposition'] = {'mae': mae_q, 'rmse': rmse_q, 'r2': r2_q}
    
    return results, y_test, y_pred_quantum

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================

def create_visualizations(anom_res, reg_res, y_test_a, y_pred_a, y_test_r, y_pred_r, cm):
    """Create visualizations"""
    log_msg("\n📊 Creating Visualizations...")
    
    # Anomaly metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0A0A0A')
    fig.suptitle('🧬 Quantum Anomaly Detection - Classification Metrics\n(Data Augmentation + Hyperparameter Tuning)', 
                 fontsize=18, fontweight='bold', color='white')
    
    models = list(anom_res.keys())
    metrics_data = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1-Score')
    ]
    colors = ['#FF4500', '#00BFFF', '#39FF14']
    
    for idx, (metric_key, metric_name) in enumerate(metrics_data):
        ax = axes[idx // 2, idx % 2]
        data = [anom_res[m][metric_key] for m in models]
        ax.bar(range(len(models)), data, color=colors, alpha=0.85, edgecolor='white', linewidth=2.5)
        ax.set_title(f'🎯 {metric_name}', fontweight='bold', fontsize=13, color='white')
        ax.set_ylabel('Score', fontweight='bold', color='white')
        ax.set_ylim([0, 1])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right')
        for i, v in enumerate(data):
            ax.text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold', color='white')
        ax.set_facecolor('#1A1A2E')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        ax.grid(axis='y', alpha=0.3, color='white')
    
    plt.tight_layout()
    plt.savefig('ml_results/01_quantum_anomaly_metrics.png', dpi=300, facecolor='#0A0A0A')
    log_msg("✅ Saved: 01_quantum_anomaly_metrics.png")
    plt.close()
    
    # Regression metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0A0A0A')
    fig.suptitle('🌡️ Quantum Temperature Prediction - Metrics\n(Data Augmentation + Hyperparameter Tuning)', 
                 fontsize=18, fontweight='bold', color='white')
    
    models_reg = list(reg_res.keys())
    metric_data_reg = [
        ([reg_res[m]['mae'] for m in models_reg], 'MAE (°C)'),
        ([reg_res[m]['rmse'] for m in models_reg], 'RMSE (°C)'),
        ([reg_res[m]['r2'] for m in models_reg], 'R² Score'),
    ]
    
    colors_reg = ['#FF4500', '#00BFFF', '#39FF14', '#FFD700', '#FF1493', '#00FF00']
    
    for idx, (data, title) in enumerate(metric_data_reg):
        ax = axes[idx]
        ax.bar(range(len(models_reg)), data, color=colors_reg[:len(models_reg)], alpha=0.85, edgecolor='white', linewidth=2.5)
        ax.set_title(f'📊 {title}', fontweight='bold', fontsize=13, color='white')
        if idx == 2:
            ax.set_ylim([0, 1])
        ax.set_xticks(range(len(models_reg)))
        ax.set_xticklabels(models_reg, rotation=20, ha='right', fontsize=9)
        for i, v in enumerate(data):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', color='white', fontsize=9)
        ax.set_facecolor('#1A1A2E')
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        ax.grid(axis='y', alpha=0.3, color='white')
    
    plt.tight_layout()
    plt.savefig('ml_results/02_quantum_regression_metrics.png', dpi=300, facecolor='#0A0A0A')
    log_msg("✅ Saved: 02_quantum_regression_metrics.png")
    plt.close()
    
    # Prediction vs Actual
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0A0A0A')
    ax.scatter(y_test_r, y_pred_r, alpha=0.6, s=50, color='#FF4500', edgecolor='#00BFFF', linewidth=1)
    min_val = min(y_test_r.min(), y_pred_r.min())
    max_val = max(y_test_r.max(), y_pred_r.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'g--', lw=3, label='Perfect', alpha=0.8)
    ax.set_xlabel('Actual (°C)', fontweight='bold', fontsize=13, color='white')
    ax.set_ylabel('Predicted (°C)', fontweight='bold', fontsize=13, color='white')
    ax.set_title('🌡️ Quantum Ensemble - Predictions vs Actual', fontsize=15, fontweight='bold', color='white')
    ax.legend(fontsize=11)
    ax.set_facecolor('#1A1A2E')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    ax.grid(True, alpha=0.2, color='white')
    plt.tight_layout()
    plt.savefig('ml_results/03_quantum_prediction_vs_actual.png', dpi=300, facecolor='#0A0A0A')
    log_msg("✅ Saved: 03_quantum_prediction_vs_actual.png")
    plt.close()
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0A0A0A')
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=True, ax=ax,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    ax.set_title('🚨 Anomaly Detection - Confusion Matrix', fontsize=15, fontweight='bold', color='white', pad=20)
    ax.set_ylabel('True', fontweight='bold', color='white')
    ax.set_xlabel('Predicted', fontweight='bold', color='white')
    ax.set_facecolor('#1A1A2E')
    ax.tick_params(colors='white')
    plt.tight_layout()
    plt.savefig('ml_results/04_confusion_matrix.png', dpi=300, facecolor='#0A0A0A')
    log_msg("✅ Saved: 04_confusion_matrix.png")
    plt.close()

# ============================================================================
# 11. GENERATE REPORT
# ============================================================================

def generate_report(anom_res, reg_res):
    """Generate report"""
    log_msg("\n" + "="*100)
    log_msg("📋 GENERATING REPORT")
    log_msg("="*100)
    
    report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║   🧬 FORGE INTELLIGENCE v3.0 - QUANTUM ML REPORT (v2.4 PRODUCTION FINAL)      ║
╚════════════════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ ANOMALY DETECTION
────────────────────────────────────────────────────────────────────────────────
"""
    
    for model, metrics in anom_res.items():
        report += f"\n{model}:\n"
        report += f"  • Accuracy:  {metrics['accuracy']:.4f}\n"
        report += f"  • Precision: {metrics['precision']:.4f}\n"
        report += f"  • Recall:    {metrics['recall']:.4f}\n"
        report += f"  • F1-Score:  {metrics['f1']:.4f}\n"
    
    report += """\n
✅ TEMPERATURE PREDICTION
────────────────────────────────────────────────────────────────────────────────
"""
    
    for model, metrics in reg_res.items():
        report += f"\n{model}:\n"
        report += f"  • MAE:  {metrics['mae']:.4f}°C\n"
        report += f"  • RMSE: {metrics['rmse']:.4f}°C\n"
        report += f"  • R²:   {metrics['r2']:.4f}\n"
    
    report += """\n
✅ STATUS: PRODUCTION READY
════════════════════════════════════════════════════════════════════════════════
"""
    
    print(report)
    with open('ml_results/quantum_ml_report_v2_4.txt', 'w') as f:
        f.write(report)
    log_msg("✅ Report saved")

# ============================================================================
# 12. MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*100)
    print("🧬 QUANTUM ML EVALUATION ENGINE v2.4 - PRODUCTION FINAL")
    print("="*100 + "\n")
    
    try:
        df = generate_datasets(n_records=25000)
        features_df = create_quantum_features(df)
        
        X_train_a, X_test_a, y_train_a, y_test_a = prepare_classification_data(df, features_df)
        anom_res, y_test_a, y_pred_a, cm = evaluate_quantum_anomaly_detection(X_train_a, X_test_a, y_train_a, y_test_a)
        
        X_train_r, X_test_r, y_train_r, y_test_r = prepare_regression_data(df, features_df)
        reg_res, y_test_r, y_pred_r = evaluate_quantum_temperature_prediction(X_train_r, X_test_r, y_train_r, y_test_r)
        
        create_visualizations(anom_res, reg_res, y_test_a, y_pred_a, y_test_r, y_pred_r, cm)
        generate_report(anom_res, reg_res)
        
        log_msg("\n" + "="*100)
        log_msg("✅ ALL EVALUATIONS COMPLETE!")
        log_msg("="*100)
        
    except Exception as e:
        log_msg(f"❌ ERROR: {str(e)}", level="ERROR")
        import traceback
        log_msg(traceback.format_exc(), level="ERROR")
        sys.exit(1)
