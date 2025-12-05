import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_curve
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
        feature_list = []
        
        for stay_id in stay_ids:
            df_stay = df_vitals[df_vitals['stay_id'] == stay_id].copy()
            start_time = df_stay['charttime'].min()
            end_time = df_stay['charttime'].max()
            time_grid = pd.date_range(start=start_time, end=end_time, freq=self.freq)
            
            df_pivot = df_stay.pivot_table(
                index='charttime', 
                columns='variable', 
                values='value', 
                aggfunc='mean'
            )
            
            df_regular = df_pivot.reindex(time_grid)
            df_filled = df_regular.ffill().bfill()
            df_filled = df_filled.fillna(df_filled.mean())
            
            features = {}
            features['stay_id'] = stay_id
            
            for col in df_filled.columns:
                features[f'{col}_mean'] = df_filled[col].mean()
                features[f'{col}_std'] = df_filled[col].std()
                features[f'{col}_min'] = df_filled[col].min()
                features[f'{col}_max'] = df_filled[col].max()
                features[f'{col}_median'] = df_filled[col].median()
                features[f'{col}_first'] = df_filled[col].iloc[0] if len(df_filled) > 0 else np.nan
                features[f'{col}_last'] = df_filled[col].iloc[-1] if len(df_filled) > 0 else np.nan
            
            features['los_hours'] = len(time_grid)
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
    
    def plot_feature_importance(self, top_n=20):
        if self.feature_names is None:
            print("Model must be trained first")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_features = importance_df.head(top_n)
        
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
