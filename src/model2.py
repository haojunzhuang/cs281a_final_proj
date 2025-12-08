"""
Model2: Irregularity and Missingness Explainable Modeling

Model2 extends the fixed-grid approach by explicitly modeling missingness 
and temporal irregularity as predictive signals. Beyond mean/max/min statistics 
per time bucket, it engineers features capture 6 more "missingness and irregularity features": 
presence indicators, measurement counts, sampling density, time gaps between observations
gap variability, and consecutive missing patterns. Using XGBoost with SHAP explainability, 
the model reveals which missingness patterns drive mortality predictions. 
This approach treats irregular sampling not as noise but as informative—sparse 
measurements may indicate patient stability while dense monitoring suggests deterioration. 
SHAP waterfall plots provide instance-level explanations showing how both measured values 
and measurement patterns contribute to each prediction.
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
import shap
import warnings
warnings.filterwarnings('ignore')


class Model2:
    def __init__(self, freq='3H', random_state=42, n_stay_ids=None):
        self.freq = freq
        self.n_stay_ids = n_stay_ids
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            base_score=0.5
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.stay_ids = None
        self.explainer = None
        
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
                        
                        features[f'{var}_{time_label}_present'] = 1
                        features[f'{var}_{time_label}_count'] = len(df_var)
                        
                        if len(df_var) > 1:
                            time_diffs = df_var['hours_from_start'].diff().dropna()
                            features[f'{var}_{time_label}_mean_gap'] = time_diffs.mean() if len(time_diffs) > 0 else 0
                            features[f'{var}_{time_label}_max_gap'] = time_diffs.max() if len(time_diffs) > 0 else 0
                            features[f'{var}_{time_label}_gap_cv'] = (time_diffs.std() / time_diffs.mean()) if len(time_diffs) > 0 and time_diffs.mean() > 0 else 0
                        else:
                            features[f'{var}_{time_label}_mean_gap'] = 0
                            features[f'{var}_{time_label}_max_gap'] = 0
                            features[f'{var}_{time_label}_gap_cv'] = 0
                        
                        features[f'{var}_{time_label}_density'] = len(df_var) / freq_hours
                    else:
                        features[f'{var}_{time_label}_mean'] = np.nan
                        features[f'{var}_{time_label}_max'] = np.nan
                        features[f'{var}_{time_label}_min'] = np.nan
                        features[f'{var}_{time_label}_present'] = 0
                        features[f'{var}_{time_label}_count'] = 0
                        features[f'{var}_{time_label}_mean_gap'] = np.nan
                        features[f'{var}_{time_label}_max_gap'] = np.nan
                        features[f'{var}_{time_label}_gap_cv'] = np.nan
                        features[f'{var}_{time_label}_density'] = 0
            
            for var in vital_vars:
                df_var_all = df_stay[df_stay['variable'] == var].sort_values('hours_from_start')
                
                if len(df_var_all) > 1:
                    all_gaps = df_var_all['hours_from_start'].diff().dropna()
                    features[f'{var}_overall_mean_gap'] = all_gaps.mean()
                    features[f'{var}_overall_max_gap'] = all_gaps.max()
                    features[f'{var}_overall_gap_cv'] = (all_gaps.std() / all_gaps.mean()) if all_gaps.mean() > 0 else 0
                else:
                    features[f'{var}_overall_mean_gap'] = np.nan
                    features[f'{var}_overall_max_gap'] = np.nan
                    features[f'{var}_overall_gap_cv'] = np.nan
                
                features[f'{var}_total_measurements'] = len(df_var_all)
                features[f'{var}_measurement_rate'] = len(df_var_all) / 48.0
                
                bucket_presence = []
                for bucket in range(n_buckets):
                    df_bucket = df_stay[df_stay['time_bucket'] == bucket]
                    df_var_bucket = df_bucket[df_bucket['variable'] == var]
                    bucket_presence.append(1 if len(df_var_bucket) > 0 else 0)
                
                features[f'{var}_buckets_present'] = sum(bucket_presence)
                features[f'{var}_buckets_missing'] = n_buckets - sum(bucket_presence)
                
                if len(bucket_presence) > 1:
                    consecutive_missing = []
                    current_missing = 0
                    for present in bucket_presence:
                        if present == 0:
                            current_missing += 1
                        else:
                            if current_missing > 0:
                                consecutive_missing.append(current_missing)
                            current_missing = 0
                    if current_missing > 0:
                        consecutive_missing.append(current_missing)
                    
                    features[f'{var}_max_consecutive_missing'] = max(consecutive_missing) if consecutive_missing else 0
                else:
                    features[f'{var}_max_consecutive_missing'] = 0
            
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
        self.X_train_background = X_train[:min(100, len(X_train))]
        try:
            self.explainer = shap.Explainer(self.model.predict_proba, self.X_train_background)
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer during fit: {e}")
            print("Explainer will be initialized on first use.")
            self.explainer = None
    
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
            print(f"\n=== Model2 XGBoost Results ===")
            print(f"Train Accuracy: {results['train_accuracy']:.4f}")
            print(f"Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Train AUC-ROC: {results['train_auc']:.4f}")
            print(f"Test AUC-ROC: {results['test_auc']:.4f}")
            print("\nTest Set Classification Report:")
            print(classification_report(y_test, y_test_pred))
        
        return results
    
    def plot_feature_importance(self, top_n=20, feature_type='all'):
        if self.feature_names is None:
            print("Model must be trained first")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if feature_type == 'missingness':
            importance_df = importance_df[importance_df['feature'].str.contains('present|count|missing|density')]
        elif feature_type == 'gaps':
            importance_df = importance_df[importance_df['feature'].str.contains('gap|rate')]
        elif feature_type == 'values':
            importance_df = importance_df[importance_df['feature'].str.contains('mean|max|min') & 
                                         ~importance_df['feature'].str.contains('gap')]

        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances ({feature_type})')
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def explain_prediction(self, X_test, idx=0, plot=True):
        if self.explainer is None:
            if not hasattr(self, 'X_train_background'):
                print("Model must be trained first")
                return None
            try:
                self.explainer = shap.Explainer(self.model.predict_proba, self.X_train_background)
            except Exception as e:
                print(f"Error initializing SHAP explainer: {e}")
                return None
        
        explanation = self.explainer(X_test)
        
        if plot:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation[idx][:, 1])
        
        return explanation.values[:, :, 1]
    
    def plot_shap_summary(self, X_test, max_display=20):
        if self.explainer is None:
            if not hasattr(self, 'X_train_background'):
                print("Model must be trained first")
                return None
            try:
                self.explainer = shap.Explainer(self.model.predict_proba, self.X_train_background)
            except Exception as e:
                print(f"Error initializing SHAP explainer: {e}")
                return None
        
        explanation = self.explainer(X_test)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(explanation[:, :, 1], max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
    
    def analyze_missingness_impact(self, X_test):
        if self.explainer is None:
            if not hasattr(self, 'X_train_background'):
                print("Model must be trained first")
                return None
            try:
                self.explainer = shap.Explainer(self.model.predict_proba, self.X_train_background)
            except Exception as e:
                print(f"Error initializing SHAP explainer: {e}")
                return None
        
        shap_values = self.explainer(X_test).values[:, :, 1]
        
        missingness_features = [i for i, name in enumerate(self.feature_names) 
                               if 'present' in name or 'count' in name or 'missing' in name or 'density' in name]
        
        if len(missingness_features) == 0:
            print("No missingness features found")
            return None
        
        missingness_impact = np.abs(shap_values[:, missingness_features]).mean(axis=0)
        missingness_names = [self.feature_names[i] for i in missingness_features]
        
        impact_df = pd.DataFrame({
            'feature': missingness_names,
            'mean_abs_shap': missingness_impact
        }).sort_values('mean_abs_shap', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.barh(impact_df['feature'][:20][::-1], impact_df['mean_abs_shap'][:20][::-1])
        plt.xlabel('Mean Absolute SHAP Value')
        plt.ylabel('Missingness Feature')
        plt.title('Impact of Missingness Features on Predictions')
        plt.tight_layout()
        plt.show()
        
        return impact_df
    
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
        fig.suptitle(f'Data Gridding Strategy with Missingness (Stay ID: {stay_id})', fontsize=16, fontweight='bold')
        
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
                
                df_bucket = df_var[df_var['time_bucket'] == bucket]
                
                if len(df_bucket) > 0:
                    ax.axvspan(bucket_start, bucket_end, 
                              alpha=0.15, color='green', zorder=1)
                    
                    mean_val = df_bucket['value'].mean()
                    max_val = df_bucket['value'].max()
                    min_val = df_bucket['value'].min()
                    
                    ax.plot([bucket_start, bucket_end], [mean_val, mean_val], 
                           'r-', linewidth=2, alpha=0.7, zorder=2)
                    ax.plot([bucket_start, bucket_end], [max_val, max_val], 
                           'g--', linewidth=1.5, alpha=0.6, zorder=2)
                    ax.plot([bucket_start, bucket_end], [min_val, min_val], 
                           'orange', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
                    
                    bucket_center = (bucket_start + bucket_end) / 2
                    ax.text(bucket_center, ax.get_ylim()[1] * 0.95, f'n={len(df_bucket)}', 
                           ha='center', fontsize=8, color='darkgreen', weight='bold')
                else:
                    ax.axvspan(bucket_start, bucket_end, 
                              alpha=0.2, color='red', zorder=1)
                    bucket_center = (bucket_start + bucket_end) / 2
                    ax.text(bucket_center, ax.get_ylim()[1] * 0.95 if len(df_var) > 0 else 0.5, 
                           'MISSING', ha='center', fontsize=8, color='darkred', weight='bold')
            
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

