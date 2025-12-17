"""
Model Validation Framework

This module implements comprehensive model validation for imbalanced data,
including stratified cross-validation, statistical significance testing,
and model drift detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import StratifiedKFold, cross_val_score, validation_curve
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
from .metrics import CustomMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelValidator:
    """
    Comprehensive Model Validation Framework
    
    Provides robust validation methods specifically designed for imbalanced
    classification problems in credit card churn prediction.
    """
    
    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        """
        Initialize Model Validator
        
        Args:
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.metrics_calculator = CustomMetrics()
        
    def stratified_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, 
                                  metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Perform stratified cross-validation with multiple metrics
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with CV results for each metric
        """
        if metrics is None:
            metrics = ['f2_score', 'f1_score', 'roc_auc', 'precision', 'recall', 'g_mean']
        
        cv_scores = {}
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for metric in metrics:
            if metric in ['f2_score', 'g_mean', 'balanced_accuracy']:
                scorer = make_scorer(getattr(self.metrics_calculator, metric), greater_is_better=True)
            elif metric == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                scorer = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True)
            else:
                from sklearn.metrics import precision_score, recall_score, f1_score
                scorer = make_scorer(eval(metric), greater_is_better=True)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            
            cv_scores[metric] = {
                'scores': scores.tolist(),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        
        return cv_scores
    
    def nested_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, 
                              param_grid: Dict[str, List], metric: str = 'f2_score') -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased performance estimation
        
        Args:
            model: Model class (not fitted)
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for hyperparameter tuning
            metric: Metric to optimize
            
        Returns:
            Dictionary with nested CV results
        """
        from sklearn.model_selection import GridSearchCV
        
        # Create scorer
        if metric in ['f2_score', 'g_mean', 'balanced_accuracy']:
            scorer = make_scorer(getattr(self.metrics_calculator, metric), greater_is_better=True)
        else:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            scorer = make_scorer(eval(metric), greater_is_better=True)
        
        # Outer CV loop
        outer_cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        outer_scores = []
        best_params_list = []
        
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
            
            # Inner CV loop for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv, scoring=scorer, n_jobs=-1
            )
            
            # Fit on outer training set
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Evaluate on outer test set
            y_pred = grid_search.predict(X_test_outer)
            y_pred_proba = grid_search.predict_proba(X_test_outer)[:, 1]
            
            # Calculate metric score
            if metric in ['f2_score', 'g_mean', 'balanced_accuracy']:
                score = getattr(self.metrics_calculator, metric)(y_test_outer, y_pred)
            elif metric == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_test_outer, y_pred_proba)
            else:
                score = eval(metric)(y_test_outer, y_pred)
            
            outer_scores.append(score)
            best_params_list.append(grid_search.best_params_)
        
        return {
            'scores': outer_scores,
            'mean_score': float(np.mean(outer_scores)),
            'std_score': float(np.std(outer_scores)),
            'best_params': best_params_list
        }
    
    def statistical_significance_test(self, model1_scores: np.ndarray, model2_scores: np.ndarray, 
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance test between two models
        
        Args:
            model1_scores: CV scores from first model
            model2_scores: CV scores from second model
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_p_value = stats.wilcoxon(model1_scores, model2_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(model1_scores) - 1) * np.var(model1_scores, ddof=1) + 
                             (len(model2_scores) - 1) * np.var(model2_scores, ddof=1)) / 
                            (len(model1_scores) + len(model2_scores) - 2))
        cohens_d = (np.mean(model1_scores) - np.mean(model2_scores)) / pooled_std
        
        results = {
            'model1_mean': float(np.mean(model1_scores)),
            'model2_mean': float(np.mean(model2_scores)),
            'difference': float(np.mean(model1_scores) - np.mean(model2_scores)),
            't_statistic': float(t_stat),
            'p_value_t_test': float(p_value),
            'wilcoxon_statistic': float(w_stat),
            'p_value_wilcoxon': float(w_p_value),
            'cohens_d': float(cohens_d),
            'significant_t_test': p_value < alpha,
            'significant_wilcoxon': w_p_value < alpha,
            'effect_size': 'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.5 else 'large'
        }
        
        return results
    
    def learning_curves_analysis(self, model, X: pd.DataFrame, y: pd.Series, 
                               metric: str = 'f2_score', train_sizes: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Generate learning curves to analyze bias-variance tradeoff
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            metric: Metric to track
            train_sizes: Training set sizes to test
            
        Returns:
            Dictionary with learning curve results
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Create scorer
        if metric in ['f2_score', 'g_mean', 'balanced_accuracy']:
            scorer = getattr(self.metrics_calculator, metric)
        else:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            scorer = eval(metric)
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name='max_depth' if hasattr(model, 'max_depth') else 'C',
            param_range=[3, 5, 7, 9] if hasattr(model, 'max_depth') else [0.1, 1.0, 10.0],
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring=make_scorer(scorer, greater_is_better=True),
            n_jobs=-1
        )
        
        return {
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
            'val_scores_std': np.std(val_scores, axis=1).tolist(),
            'param_range': [3, 5, 7, 9] if hasattr(model, 'max_depth') else [0.1, 1.0, 10.0]
        }
    
    def plot_learning_curves(self, learning_curve_results: Dict[str, Any], 
                           figsize: Tuple[int, int] = (10, 6)):
        """
        Plot learning curves
        
        Args:
            learning_curve_results: Results from learning_curves_analysis
            figsize: Figure size
        """
        param_range = learning_curve_results['param_range']
        train_mean = learning_curve_results['train_scores_mean']
        train_std = learning_curve_results['train_scores_std']
        val_mean = learning_curve_results['val_scores_mean']
        val_std = learning_curve_results['val_scores_std']
        
        plt.figure(figsize=figsize)
        plt.plot(param_range, train_mean, 'o-', label='Training Score')
        plt.fill_between(param_range, 
                        np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std),
                        alpha=0.1)
        
        plt.plot(param_range, val_mean, 'o-', label='Validation Score')
        plt.fill_between(param_range,
                        np.array(val_mean) - np.array(val_std),
                        np.array(val_mean) + np.array(val_std),
                        alpha=0.1)
        
        plt.xlabel('Parameter Value')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def bootstrap_validation(self, model, X: pd.DataFrame, y: pd.Series, 
                           n_bootstrap: int = 1000, metric: str = 'f2_score') -> Dict[str, Any]:
        """
        Perform bootstrap validation for confidence intervals
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target variable
            n_bootstrap: Number of bootstrap samples
            metric: Metric to calculate
            
        Returns:
            Dictionary with bootstrap results
        """
        from sklearn.model_selection import train_test_split
        
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[indices]
            y_bootstrap = y.iloc[indices]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_bootstrap, y_bootstrap, test_size=0.2, random_state=self.random_state + i
            )
            
            # Train and evaluate
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_test)
            
            # Calculate metric
            if metric in ['f2_score', 'g_mean', 'balanced_accuracy']:
                score = getattr(self.metrics_calculator, metric)(y_test, y_pred)
            else:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                score = eval(metric)(y_test, y_pred)
            
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        return {
            'scores': bootstrap_scores.tolist(),
            'mean': float(np.mean(bootstrap_scores)),
            'std': float(np.std(bootstrap_scores)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'median': float(np.median(bootstrap_scores))
        }
    
    def model_stability_analysis(self, model, X: pd.DataFrame, y: pd.Series, 
                               n_iterations: int = 10) -> Dict[str, Any]:
        """
        Analyze model stability across multiple training runs
        
        Args:
            model: Model to analyze
            X: Feature matrix
            y: Target variable
            n_iterations: Number of training iterations
            
        Returns:
            Dictionary with stability analysis results
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, roc_auc_score
        
        scores_f1 = []
        scores_auc = []
        feature_importances = []
        
        for i in range(n_iterations):
            # Different train-test splits
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state + i
            )
            
            # Train model
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model_copy.predict(X_test)
            y_pred_proba = model_copy.predict_proba(X_test)[:, 1]
            
            scores_f1.append(f1_score(y_test, y_pred))
            scores_auc.append(roc_auc_score(y_test, y_pred_proba))
            
            # Feature importance (if available)
            if hasattr(model_copy, 'feature_importances_'):
                feature_importances.append(model_copy.feature_importances_)
        
        # Calculate stability metrics
        f1_std = np.std(scores_f1)
        auc_std = np.std(scores_auc)
        
        stability_results = {
            'f1_scores': scores_f1,
            'auc_scores': scores_auc,
            'f1_mean': float(np.mean(scores_f1)),
            'f1_std': float(f1_std),
            'auc_mean': float(np.mean(scores_auc)),
            'auc_std': float(auc_std),
            'f1_cv': float(f1_std / np.mean(scores_f1)) if np.mean(scores_f1) > 0 else 0,
            'auc_cv': float(auc_std / np.mean(scores_auc)) if np.mean(scores_auc) > 0 else 0
        }
        
        # Feature importance stability
        if feature_importances:
            feature_importances = np.array(feature_importances)
            stability_results['feature_importance_mean'] = np.mean(feature_importances, axis=0).tolist()
            stability_results['feature_importance_std'] = np.std(feature_importances, axis=0).tolist()
        
        return stability_results
    
    def comprehensive_validation(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform comprehensive validation on multiple models
        
        Args:
            models: Dictionary of models to validate
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with comprehensive validation results
        """
        validation_results = {}
        
        for model_name, model in models.items():
            print(f"Validating {model_name}...")
            
            # Basic CV validation
            cv_results = self.stratified_cross_validation(model, X, y)
            
            # Stability analysis
            stability_results = self.model_stability_analysis(model, X, y)
            
            # Bootstrap validation
            bootstrap_results = self.bootstrap_validation(model, X, y)
            
            validation_results[model_name] = {
                'cross_validation': cv_results,
                'stability': stability_results,
                'bootstrap': bootstrap_results
            }
        
        # Statistical significance tests between models
        model_names = list(models.keys())
        significance_tests = {}
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1_name = model_names[i]
                model2_name = model_names[j]
                
                model1_scores = np.array(validation_results[model1_name]['cross_validation']['f2_score']['scores'])
                model2_scores = np.array(validation_results[model2_name]['cross_validation']['f2_score']['scores'])
                
                significance_tests[f'{model1_name}_vs_{model2_name}'] = self.statistical_significance_test(
                    model1_scores, model2_scores
                )
        
        validation_results['significance_tests'] = significance_tests
        
        return validation_results
