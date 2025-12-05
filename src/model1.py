"""
Model1: Fixed Grid with Imputation

Model1 is a baseline algorithm to deal with irregular time series vital signs 
for predicting in-hospital mortality. It regularizes the first 48 hours of patient data 
onto a fixed time grid (configurable frequency, e.g., 6-hour buckets). 
For each time bucket and vital sign (heart rate, respiratory rate, O2 saturation), 
it extracts three statistics: mean, maximum, and minimum. 
These aggregated features are then standardized and fed into an XGBoost classifier. 
The model also include gridding strategy visualizer to show the bucketting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    classification_report, 
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class Model1:
    def __init__(self, freq='3H', random_state=42, n_stay_ids=None):
        self.freq = freq
        self.n_stay_ids = n_stay_ids
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            random_state=random_state, 
        )
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def regularize_time_series(self, df_vitals, df_patients):
        stay_ids = df_vitals['stay_id'].unique()[:self.n_stay_ids]
        self.stay_ids = stay_ids
        feature_list = []
        
        freq_hours = pd.Timedelta(self.freq).total_seconds() / 3600
        n_buckets = int(48 / freq_hours)
        
        for stay_id in stay_ids:
            df_stay = df_vitals[df_vitals['stay_id'] == stay_id].copy()
            start_time = df_stay['charttime'].min()
            
            df_stay['hours_from_start'] = (df_stay['charttime'] - start_time).dt.total_seconds() / 3600
            df_stay = df_stay[df_stay['hours_from_start'] < 48]
            
            df_stay['time_bucket'] = (df_stay['hours_from_start'] / freq_hours).astype(int)
            df_stay['time_bucket'] = df_stay['time_bucket'].clip(0, n_buckets - 1)
            
            features = {}
            features['stay_id'] = stay_id
            
            vital_vars = df_stay['variable'].unique()
            
            for bucket in range(n_buckets):
                df_bucket = df_stay[df_stay['time_bucket'] == bucket]
                
                start_hr = int(bucket * freq_hours)
                end_hr = int((bucket + 1) * freq_hours)
                time_label = f'{start_hr}-{end_hr}hr'
                
                for var in vital_vars:
                    df_var = df_bucket[df_bucket['variable'] == var]
                    
                    if len(df_var) > 0:
                        features[f'{var}_{time_label}_mean'] = df_var['value'].mean()
                        features[f'{var}_{time_label}_max'] = df_var['value'].max()
                        features[f'{var}_{time_label}_min'] = df_var['value'].min()
                    else:
                        features[f'{var}_{time_label}_mean'] = np.nan
                        features[f'{var}_{time_label}_max'] = np.nan
                        features[f'{var}_{time_label}_min'] = np.nan
            
            feature_list.append(features)
        
        features_df = pd.DataFrame(feature_list)
        features_df = features_df.merge(
            df_patients[['stay_id', 'died_in_hosp']], 
            on='stay_id', 
            how='left'
        )
        
        return features_df
    
    def prepare_data(self, df_features, test_size=0.2):
        X = df_features.drop(['stay_id', 'died_in_hosp'], axis=1)
        y = df_features['died_in_hosp']
        
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_train, X_test, y_train, y_test, verbose=True):
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        y_train_proba = self.predict_proba(X_train)[:, 1]
        y_test_proba = self.predict_proba(X_test)[:, 1]
        
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'test_auc': roc_auc_score(y_test, y_test_proba),
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        if verbose:
            print(f"\n=== XGBoost Results ===")
            print(f"Train Accuracy: {results['train_accuracy']:.4f}")
            print(f"Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Train AUC-ROC: {results['train_auc']:.4f}")
            print(f"Test AUC-ROC: {results['test_auc']:.4f}")
            print("\nTest Set Classification Report:")
            print(classification_report(y_test, y_test_pred))
        
        return results
    
    def visualize(self, stay_id, df_vitals):
        df_stay = df_vitals[df_vitals['stay_id'] == stay_id].copy()
        
        if len(df_stay) == 0:
            print(f"No data found for stay_id {stay_id}")
            return
        
        available_vars = df_stay['variable'].unique()
        print(f"Available variables for stay_id {stay_id}: {available_vars}")
        
        vital_vars = available_vars[:3] if len(available_vars) >= 3 else available_vars
        
        if len(vital_vars) == 0:
            print(f"No variables found for stay_id {stay_id}")
            return
        
        start_time = df_stay['charttime'].min()
        df_stay['hours_from_start'] = (df_stay['charttime'] - start_time).dt.total_seconds() / 3600
        df_stay = df_stay[df_stay['hours_from_start'] < 48]
        
        freq_hours = pd.Timedelta(self.freq).total_seconds() / 3600
        n_buckets = int(48 / freq_hours)
        df_stay['time_bucket'] = (df_stay['hours_from_start'] / freq_hours).astype(int)
        df_stay['time_bucket'] = df_stay['time_bucket'].clip(0, n_buckets - 1)
        
        fig, axes = plt.subplots(len(vital_vars), 1, figsize=(14, 4 * len(vital_vars)))
        if len(vital_vars) == 1:
            axes = [axes]
        fig.suptitle(f'Data Gridding Strategy for Stay ID: {stay_id}', fontsize=16, fontweight='bold')
        
        for idx, var_name in enumerate(vital_vars):
            ax = axes[idx]
            df_var = df_stay[df_stay['variable'] == var_name]
            
            if len(df_var) == 0:
                ax.text(0.5, 0.5, f'No data available for {var_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlim(0, 48)
                continue
            
            ax.scatter(df_var['hours_from_start'], df_var['value'], 
                      alpha=0.6, s=50, color='steelblue', label='Raw measurements', zorder=3)
            
            for bucket in range(n_buckets):
                bucket_start = bucket * freq_hours
                bucket_end = (bucket + 1) * freq_hours
                
                ax.axvspan(bucket_start, bucket_end, 
                          alpha=0.1 if bucket % 2 == 0 else 0.05, 
                          color='gray', zorder=1)
                
                df_bucket = df_var[df_var['time_bucket'] == bucket]
                
                if len(df_bucket) > 0:
                    mean_val = df_bucket['value'].mean()
                    max_val = df_bucket['value'].max()
                    min_val = df_bucket['value'].min()
                    
                    bucket_center = (bucket_start + bucket_end) / 2
                    
                    ax.plot([bucket_start, bucket_end], [mean_val, mean_val], 
                           'r-', linewidth=2, alpha=0.7, zorder=2)
                    ax.plot([bucket_start, bucket_end], [max_val, max_val], 
                           'g--', linewidth=1.5, alpha=0.6, zorder=2)
                    ax.plot([bucket_start, bucket_end], [min_val, min_val], 
                           'orange', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
            
            for bucket in range(n_buckets + 1):
                bucket_edge = bucket * freq_hours
                if bucket_edge <= 48:
                    ax.axvline(bucket_edge, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=2)
            
            ax.set_xlabel('Hours from start', fontsize=11)
            ax.set_ylabel(var_name, fontsize=11)
            ax.set_xlim(0, 48)
            ax.grid(True, alpha=0.3, zorder=0)
            ax.legend(['Raw measurements', 'Mean', 'Max', 'Min'], 
                     loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        if self.feature_names is None:
            print("Model must be trained first")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
