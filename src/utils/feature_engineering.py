"""
Feature Engineering Utilities

This module provides advanced feature engineering capabilities for credit card churn prediction,
including feature selection, dimensionality reduction, and domain-specific feature creation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    f_classif, mutual_info_classif, chi2
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Advanced Feature Engineering Pipeline
    
    Provides comprehensive feature engineering capabilities including:
    - Feature selection methods
    - Dimensionality reduction
    - Domain-specific feature creation
    - Feature importance analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Feature Engineer
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.selected_features = None
        self.feature_importance = None
        self.reduction_model = None
        self.is_fitted = False
        
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for credit card churn prediction
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional domain features
        """
        df_engineered = df.copy()
        
        # Credit utilization features
        if 'credit_limit' in df.columns and 'total_revolv_bal' in df.columns:
            df_engineered['credit_utilization'] = df['total_revolv_bal'] / df['credit_limit']
            df_engineered['credit_utilization_bucket'] = pd.cut(
                df_engineered['credit_utilization'], 
                bins=[0, 0.3, 0.6, 0.9, 1.0], 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # Transaction behavior features
        if 'total_trans_amt' in df.columns and 'total_trans_ct' in df.columns:
            df_engineered['avg_transaction_amount'] = df['total_trans_amt'] / df['total_trans_ct']
            df_engineered['transaction_frequency'] = df['total_trans_ct'] / 12  # Per month
        
        # Customer tenure features
        if 'customer_age' in df.columns and 'months_on_book' in df.columns:
            df_engineered['age_to_tenure_ratio'] = df['customer_age'] / df['months_on_book']
            df_engineered['tenure_bucket'] = pd.cut(
                df['months_on_book'], 
                bins=[0, 12, 24, 36, 48, 100], 
                labels=['New', 'Short', 'Medium', 'Long', 'Very Long']
            )
        
        # Risk indicators
        if 'months_inactive_12_mon' in df.columns and 'contacts_count_12_mon' in df.columns:
            df_engineered['inactivity_risk'] = df['months_inactive_12_mon'] * df['contacts_count_12_mon']
            df_engineered['high_risk_indicator'] = (
                (df['months_inactive_12_mon'] >= 3) | 
                (df['contacts_count_12_mon'] >= 3)
            ).astype(int)
        
        # Product usage features
        if 'num_products' in df.columns:
            df_engineered['product_usage_bucket'] = pd.cut(
                df['num_products'], 
                bins=[0, 1, 2, 3, 10], 
                labels=['Single', 'Double', 'Triple', 'Multiple']
            )
        
        # Balance features
        if 'total_revolv_bal' in df.columns and 'avg_open_to_buy' in df.columns:
            df_engineered['balance_to_limit_ratio'] = df['total_revolv_bal'] / (df['total_revolv_bal'] + df['avg_open_to_buy'])
            df_engineered['available_credit_ratio'] = df['avg_open_to_buy'] / (df['total_revolv_bal'] + df['avg_open_to_buy'])
        
        return df_engineered
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                   method: str = 'random_forest') -> pd.DataFrame:
        """
        Calculate feature importance using different methods
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Method to use ('random_forest', 'mutual_info', 'f_score')
            
        Returns:
            DataFrame with feature importance scores
        """
        if method == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(X, y)
            importance_scores = model.feature_importances_
            
        elif method == 'mutual_info':
            importance_scores = mutual_info_classif(X, y, random_state=self.random_state)
            
        elif method == 'f_score':
            importance_scores, _ = f_classif(X, y)
            
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        return importance_df
    
    def select_features_by_importance(self, X: pd.DataFrame, y: pd.Series, 
                                    method: str = 'random_forest', 
                                    top_k: int = 20) -> pd.DataFrame:
        """
        Select top features based on importance scores
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Importance calculation method
            top_k: Number of top features to select
            
        Returns:
            DataFrame with selected features
        """
        importance_df = self.calculate_feature_importance(X, y, method)
        top_features = importance_df.head(top_k)['feature'].tolist()
        
        self.selected_features = top_features
        return X[top_features]
    
    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, 
                                 k: int = 20, score_func: str = 'f_classif') -> pd.DataFrame:
        """
        Select features using univariate statistical tests
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            score_func: Scoring function ('f_classif', 'mutual_info_classif', 'chi2')
            
        Returns:
            DataFrame with selected features
        """
        if score_func == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif score_func == 'mutual_info_classif':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif score_func == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
        else:
            raise ValueError(f"Unknown score function: {score_func}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features = selected_features
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, 
                          n_features: int = 20, estimator=None) -> pd.DataFrame:
        """
        Select features using Recursive Feature Elimination
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            estimator: Base estimator for RFE
            
        Returns:
            DataFrame with selected features
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features = selected_features
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def select_features_rfecv(self, X: pd.DataFrame, y: pd.Series, 
                            estimator=None, cv_folds: int = 5) -> pd.DataFrame:
        """
        Select features using RFE with cross-validation
        
        Args:
            X: Feature matrix
            y: Target variable
            estimator: Base estimator for RFECV
            cv_folds: Number of CV folds
            
        Returns:
            DataFrame with selected features
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        selector = RFECV(estimator=estimator, cv=cv_folds, scoring='f1')
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features = selected_features
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def select_features_model_based(self, X: pd.DataFrame, y: pd.Series, 
                                  estimator=None, threshold: str = 'median') -> pd.DataFrame:
        """
        Select features using model-based selection
        
        Args:
            X: Feature matrix
            y: Target variable
            estimator: Base estimator
            threshold: Threshold for feature selection
            
        Returns:
            DataFrame with selected features
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        selector = SelectFromModel(estimator=estimator, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selected_features = selected_features
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def reduce_dimensionality_pca(self, X: pd.DataFrame, n_components: int = 20, 
                                explained_variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Reduce dimensionality using PCA
        
        Args:
            X: Feature matrix
            n_components: Number of components (if None, use variance threshold)
            explained_variance_threshold: Minimum explained variance ratio
            
        Returns:
            Tuple of (transformed DataFrame, PCA info)
        """
        # Standardize features first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if n_components is None:
            # Find number of components for variance threshold
            pca = PCA()
            pca.fit(X_scaled)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= explained_variance_threshold) + 1
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create feature names
        feature_names = [f'PC_{i+1}' for i in range(n_components)]
        
        # Store PCA model
        self.reduction_model = pca
        
        # PCA information
        pca_info = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'total_explained_variance': np.sum(pca.explained_variance_ratio_)
        }
        
        return pd.DataFrame(X_pca, columns=feature_names, index=X.index), pca_info
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20, 
                              figsize: Tuple[int, int] = (12, 8)):
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_pca_variance(self, pca_info: Dict[str, Any], figsize: Tuple[int, int] = (12, 5)):
        """
        Plot PCA explained variance
        
        Args:
            pca_info: PCA information dictionary
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Explained variance ratio
        ax1.plot(range(1, len(pca_info['explained_variance_ratio']) + 1), 
                pca_info['explained_variance_ratio'])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True)
        
        # Cumulative explained variance
        ax2.plot(range(1, len(pca_info['cumulative_variance_ratio']) + 1), 
                pca_info['cumulative_variance_ratio'])
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_correlations(self, X: pd.DataFrame, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze feature correlations and identify highly correlated features
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold for identifying high correlations
            
        Returns:
            Dictionary with correlation analysis results
        """
        correlation_matrix = X.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Find features with high average correlation
        avg_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'average_correlations': avg_correlations,
            'highly_correlated_features': avg_correlations[avg_correlations >= threshold].index.tolist()
        }
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                               figsize: Tuple[int, int] = (15, 12)):
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def comprehensive_feature_engineering(self, df: pd.DataFrame, target_column: str, 
                                        feature_selection_method: str = 'rfe',
                                        n_features: int = 20) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive feature engineering pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_selection_method: Method for feature selection
            n_features: Number of features to select
            
        Returns:
            Tuple of (engineered DataFrame, engineering summary)
        """
        print("Starting comprehensive feature engineering...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Create domain features
        X_engineered = self.create_domain_features(X)
        print(f"Created domain features. New shape: {X_engineered.shape}")
        
        # Handle categorical features created in domain engineering
        categorical_cols = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X_engineered = pd.get_dummies(X_engineered, columns=categorical_cols, drop_first=True)
            print(f"Encoded categorical features. New shape: {X_engineered.shape}")
        
        # Feature selection
        if feature_selection_method == 'rfe':
            X_selected = self.select_features_rfe(X_engineered, y, n_features)
        elif feature_selection_method == 'importance':
            X_selected = self.select_features_by_importance(X_engineered, y, top_k=n_features)
        elif feature_selection_method == 'univariate':
            X_selected = self.select_features_univariate(X_engineered, y, k=n_features)
        elif feature_selection_method == 'model_based':
            X_selected = self.select_features_model_based(X_engineered, y)
        else:
            raise ValueError(f"Unknown feature selection method: {feature_selection_method}")
        
        print(f"Selected {len(self.selected_features)} features using {feature_selection_method}")
        
        # Add target back
        df_final = pd.concat([X_selected, y], axis=1)
        
        # Create summary
        engineering_summary = {
            'original_features': len(X.columns),
            'engineered_features': len(X_engineered.columns),
            'selected_features': len(self.selected_features),
            'feature_selection_method': feature_selection_method,
            'selected_feature_names': self.selected_features,
            'final_shape': df_final.shape
        }
        
        self.is_fitted = True
        
        print(f"Feature engineering completed. Final shape: {df_final.shape}")
        
        return df_final, engineering_summary
