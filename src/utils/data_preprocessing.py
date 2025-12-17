"""
Data Preprocessing Utilities

This module provides comprehensive data preprocessing functions for credit card churn prediction,
including handling missing values, encoding categorical variables, and data validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive Data Preprocessing Pipeline
    
    Handles all aspects of data preprocessing for credit card churn prediction,
    including missing value imputation, encoding, scaling, and validation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Data Preprocessor
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_names = None
        self.is_fitted = False
        
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data and return data quality report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data validation results
        """
        validation_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Check for potential issues
        issues = []
        
        # High missing value columns
        high_missing = [col for col, pct in validation_report['missing_percentage'].items() if pct > 50]
        if high_missing:
            issues.append(f"High missing values (>50%): {high_missing}")
        
        # Duplicate rows
        if validation_report['duplicate_rows'] > 0:
            issues.append(f"Duplicate rows found: {validation_report['duplicate_rows']}")
        
        # Memory usage
        if validation_report['memory_usage'] > 100:  # > 100 MB
            issues.append(f"Large memory usage: {validation_report['memory_usage']:.1f} MB")
        
        validation_report['issues'] = issues
        
        return validation_report
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'knn', 
                            categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for numeric columns ('mean', 'median', 'knn')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric missing values
        if numeric_cols and df_processed[numeric_cols].isnull().any().any():
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
                self.imputers['numeric'] = imputer
            else:
                imputer = SimpleImputer(strategy=strategy)
                df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
                self.imputers['numeric'] = imputer
        
        # Handle categorical missing values
        if categorical_cols and df_processed[categorical_cols].isnull().any().any():
            imputer = SimpleImputer(strategy=categorical_strategy, fill_value='Unknown')
            df_processed[categorical_cols] = imputer.fit_transform(df_processed[categorical_cols])
            self.imputers['categorical'] = imputer
        
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                   target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables using appropriate encoding strategies
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (for target encoding)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        # Identify categorical columns (including string[pyarrow] types)
        categorical_cols = []
        for col in df_encoded.columns:
            if col == target_column:
                continue
            dtype_str = str(df_encoded[col].dtype)
            if 'string' in dtype_str or dtype_str in ['object', 'category']:
                categorical_cols.append(col)
        
        for col in categorical_cols:
            # Convert pyarrow strings to regular strings first
            if 'string' in str(df_encoded[col].dtype):
                df_encoded[col] = df_encoded[col].astype(str)
                
            # Check if it's ordinal (education level)
            if col == 'education_level':
                # Ordinal encoding for education
                education_map = {
                    'Uneducated': 0,
                    'High School': 1,
                    'College': 2,
                    'Graduate': 3,
                    'Post-Graduate': 4,
                    'Doctorate': 5,
                    'Unknown': 2  # Default to College level
                }
                df_encoded[col] = df_encoded[col].map(education_map)
                self.encoders[col] = education_map
                
            else:
                # One-hot encoding for nominal variables
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
                self.encoders[col] = dummies.columns.tolist()
        
        return df_encoded
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       columns: Optional[List[str]] = None,
                       exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle outliers in numeric columns
        
        Args:
            df: Input DataFrame
            method: Method to handle outliers ('iqr', 'zscore', 'winsorize')
            columns: List of columns to process (None for all numeric columns)
            exclude_columns: Columns to exclude from outlier handling (e.g., target, binary features)
            
        Returns:
            DataFrame with outliers handled
        """
        df_processed = df.copy()
        
        if exclude_columns is None:
            exclude_columns = []
        
        if columns is None:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude binary columns (only 0 and 1) and specified columns
            numeric_cols = [col for col in numeric_cols 
                          if col not in exclude_columns 
                          and df_processed[col].nunique() > 2]
        else:
            numeric_cols = [col for col in columns if col not in exclude_columns]
        
        outlier_info = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                outlier_count = outliers.sum()
                
                # Cap outliers instead of removing
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                
                outlier_info[col] = {
                    'method': 'iqr',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers_found': outlier_count,
                    'outliers_capped': outlier_count
                }
                
            elif method == 'zscore':
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                
                # Cap outliers at 3 standard deviations
                mean_val = df_processed[col].mean()
                std_val = df_processed[col].std()
                df_processed[col] = df_processed[col].clip(
                    lower=mean_val - 3*std_val, 
                    upper=mean_val + 3*std_val
                )
                
                outlier_info[col] = {
                    'method': 'zscore',
                    'outliers_found': outlier_count,
                    'outliers_capped': outlier_count
                }
        
        self.outlier_info = outlier_info
        return df_processed
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Create interaction features between pairs of variables
        
        Args:
            df: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples for interactions
            
        Returns:
            DataFrame with interaction features
        """
        df_with_interactions = df.copy()
        
        if feature_pairs is None:
            # Default interaction features based on domain knowledge
            feature_pairs = [
                ('customer_age', 'months_on_book'),
                ('credit_limit', 'avg_utilization_ratio'),
                ('total_revolv_bal', 'avg_open_to_buy'),
                ('total_trans_amt', 'total_trans_ct'),
                ('months_inactive_12_mon', 'contacts_count_12_mon')
            ]
        
        interaction_features = []
        
        for feature1, feature2 in feature_pairs:
            if feature1 in df.columns and feature2 in df.columns:
                # Multiplication interaction
                interaction_name = f"{feature1}_x_{feature2}"
                df_with_interactions[interaction_name] = df[feature1] * df[feature2]
                interaction_features.append(interaction_name)
                
                # Ratio interaction (if denominator is not zero)
                if df[feature2].min() > 0:  # Avoid division by zero
                    ratio_name = f"{feature1}_div_{feature2}"
                    df_with_interactions[ratio_name] = df[feature1] / df[feature2]
                    interaction_features.append(ratio_name)
        
        self.interaction_features = interaction_features
        return df_with_interactions
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                 columns: Optional[List[str]] = None, 
                                 degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns
        
        Args:
            df: Input DataFrame
            columns: List of columns for polynomial features
            degree: Degree of polynomial features
            
        Returns:
            DataFrame with polynomial features
        """
        df_with_poly = df.copy()
        
        if columns is None:
            # Select numeric columns with reasonable variance
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numeric_cols if df[col].var() > 0]
        
        polynomial_features = []
        
        for col in columns:
            if col in df.columns and df[col].var() > 0:
                for d in range(2, degree + 1):
                    poly_name = f"{col}_pow_{d}"
                    df_with_poly[poly_name] = df[col] ** d
                    polynomial_features.append(poly_name)
        
        self.polynomial_features = polynomial_features
        return df_with_poly
    
    def scale_features(self, df: pd.DataFrame, method: str = 'robust', 
                      exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'robust', 'minmax')
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Select numeric columns to scale
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_scale = [col for col in numeric_cols if col not in exclude_columns]
        
        if not columns_to_scale:
            return df_scaled
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        self.scaled_columns = columns_to_scale
        
        return df_scaled
    
    def fit_preprocessing_pipeline(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit the complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Preprocessed DataFrame
        """
        print("Starting data preprocessing pipeline...")
        
        # Validate data
        validation_report = self.validate_data(df)
        print(f"Data validation completed. Shape: {validation_report['shape']}")
        
        if validation_report['issues']:
            print("Data issues found:")
            for issue in validation_report['issues']:
                print(f"  - {issue}")
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed if 'df_processed' in locals() else df)
        print("Missing values handled")
        
        # Encode categorical variables
        df_processed = self.encode_categorical_variables(df_processed, target_column)
        print("Categorical variables encoded")
        
        # Handle outliers (exclude target column)
        exclude_cols = [target_column] if target_column else []
        df_processed = self.handle_outliers(df_processed, exclude_columns=exclude_cols)
        print("Outliers handled")
        
        # Separate target column before feature engineering
        if target_column and target_column in df_processed.columns:
            target_series = df_processed[target_column].copy()
            df_features = df_processed.drop(columns=[target_column])
        else:
            target_series = None
            df_features = df_processed
        
        # Create interaction features (without target)
        df_features = self.create_interaction_features(df_features)
        print(f"Created {len(getattr(self, 'interaction_features', []))} interaction features")
        
        # Create polynomial features (without target)
        df_features = self.create_polynomial_features(df_features)
        print(f"Created {len(getattr(self, 'polynomial_features', []))} polynomial features")
        
        # Scale features (without target)
        df_features = self.scale_features(df_features)
        print("Features scaled")
        
        # Add target back
        if target_series is not None:
            df_processed = pd.concat([df_features, target_series], axis=1)
        else:
            df_processed = df_features
        
        # Store feature names
        self.feature_names = df_processed.columns.tolist()
        self.is_fitted = True
        
        print(f"Preprocessing completed. Final shape: {df_processed.shape}")
        
        return df_processed
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing pipeline to new data
        
        Args:
            df: New DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessing pipeline must be fitted before transforming new data")
        
        df_transformed = df.copy()
        
        # Apply missing value imputation
        if 'numeric' in self.imputers:
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
            df_transformed[numeric_cols] = self.imputers['numeric'].transform(df_transformed[numeric_cols])
        
        if 'categorical' in self.imputers:
            categorical_cols = df_transformed.select_dtypes(include=['object', 'category']).columns.tolist()
            df_transformed[categorical_cols] = self.imputers['categorical'].transform(df_transformed[categorical_cols])
        
        # Apply categorical encoding
        for col, encoding in self.encoders.items():
            if isinstance(encoding, dict):  # Ordinal encoding
                df_transformed[col] = df_transformed[col].map(encoding)
            else:  # One-hot encoding
                # Create dummy variables
                dummies = pd.get_dummies(df_transformed[col], prefix=col, drop_first=True)
                df_transformed = pd.concat([df_transformed, dummies], axis=1)
                df_transformed = df_transformed.drop(columns=[col])
        
        # Apply outlier handling (using stored bounds)
        if hasattr(self, 'outlier_info'):
            for col, info in self.outlier_info.items():
                if col in df_transformed.columns:
                    if info['method'] == 'iqr':
                        df_transformed[col] = df_transformed[col].clip(
                            lower=info['lower_bound'], 
                            upper=info['upper_bound']
                        )
        
        # Apply scaling
        if hasattr(self, 'scaled_columns'):
            for method, scaler in self.scalers.items():
                columns_to_scale = [col for col in self.scaled_columns if col in df_transformed.columns]
                if columns_to_scale:
                    df_transformed[columns_to_scale] = scaler.transform(df_transformed[columns_to_scale])
        
        return df_transformed
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of the preprocessing pipeline
        
        Returns:
            Dictionary with preprocessing summary
        """
        if not self.is_fitted:
            return {'status': 'Pipeline not fitted'}
        
        summary = {
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'scalers_applied': list(self.scalers.keys()),
            'imputers_applied': list(self.imputers.keys()),
            'encoders_applied': list(self.encoders.keys()),
            'interaction_features': getattr(self, 'interaction_features', []),
            'polynomial_features': getattr(self, 'polynomial_features', [])
        }
        
        if hasattr(self, 'outlier_info'):
            summary['outliers_handled'] = list(self.outlier_info.keys())
        
        return summary
