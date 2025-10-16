"""
Custom Metrics for Imbalanced Data

This module implements custom evaluation metrics specifically designed for
imbalanced classification problems like credit card churn prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class CustomMetrics:
    """
    Custom Metrics for Credit Card Churn Prediction
    
    Implements business-focused metrics that are appropriate for imbalanced data
    and align with the cost structure of churn prediction.
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Initialize Custom Metrics
        
        Args:
            beta: Beta parameter for F-beta score (default 2.0 for F2-score)
        """
        self.beta = beta
    
    def f_beta_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate F-beta score (emphasizes recall over precision)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            F-beta score
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if precision + recall == 0:
            return 0.0
        
        f_beta = (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall)
        return f_beta
    
    def specificity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (True Negative Rate)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity
    
    def balanced_accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate balanced accuracy (average of sensitivity and specificity)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Balanced accuracy score
        """
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = self.specificity_score(y_true, y_pred)
        balanced_acc = (recall + specificity) / 2
        return balanced_acc
    
    def matthews_correlation_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Matthews Correlation Coefficient (MCC)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            MCC score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        mcc = numerator / denominator
        return mcc
    
    def g_mean_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Geometric Mean (G-mean)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            G-mean score
        """
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = self.specificity_score(y_true, y_pred)
        g_mean = np.sqrt(recall * specificity)
        return g_mean
    
    def precision_recall_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate Precision-Recall AUC (PR-AUC)
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            PR-AUC score
        """
        return average_precision_score(y_true, y_pred_proba)
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive set of metrics for model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with all metrics
        """
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = self.f_beta_score(y_true, y_pred)
        
        # Advanced metrics
        specificity = self.specificity_score(y_true, y_pred)
        balanced_acc = self.balanced_accuracy_score(y_true, y_pred)
        mcc = self.matthews_correlation_coefficient(y_true, y_pred)
        g_mean = self.g_mean_score(y_true, y_pred)
        
        # Probability-based metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = self.precision_recall_auc(y_true, y_pred_proba)
        
        # Additional derived metrics
        sensitivity = recall  # Same as recall
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'f2_score': f2,
            'balanced_accuracy': balanced_acc,
            'matthews_correlation': mcc,
            'g_mean': g_mean,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'negative_predictive_value': npv,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return metrics
    
    def calculate_class_wise_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each class separately
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with class-wise metrics
        """
        classes = np.unique(y_true)
        class_metrics = {}
        
        for cls in classes:
            # Binary classification for this class
            binary_y_true = (y_true == cls).astype(int)
            binary_y_pred = (y_pred == cls).astype(int)
            
            # Calculate metrics
            precision = precision_score(binary_y_true, binary_y_pred, zero_division=0)
            recall = recall_score(binary_y_true, binary_y_pred, zero_division=0)
            f1 = f1_score(binary_y_true, binary_y_pred, zero_division=0)
            f2 = self.f_beta_score(binary_y_true, binary_y_pred)
            
            # Confusion matrix for this class
            tn, fp, fn, tp = confusion_matrix(binary_y_true, binary_y_pred).ravel()
            
            class_metrics[f'class_{cls}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'f2_score': f2,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'support': tp + fn
            }
        
        return class_metrics
    
    def calculate_threshold_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate metrics across different decision thresholds
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: Array of thresholds to test
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.01)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             metric: str = 'f2_score') -> Tuple[float, float]:
        """
        Find optimal threshold based on specified metric
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f2_score', 'f1_score', 'g_mean', etc.)
            
        Returns:
            Optimal threshold and corresponding metric value
        """
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        if metric not in threshold_metrics.columns:
            raise ValueError(f"Unknown metric: {metric}")
        
        optimal_idx = threshold_metrics[metric].idxmax()
        optimal_threshold = threshold_metrics.iloc[optimal_idx]['threshold']
        optimal_value = threshold_metrics.iloc[optimal_idx][metric]
        
        return optimal_threshold, optimal_value
    
    def plot_metrics_by_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                metrics: List[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot multiple metrics across different thresholds
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metrics: List of metrics to plot
            figsize: Figure size
        """
        if metrics is None:
            metrics = ['f1_score', 'f2_score', 'precision', 'recall', 'g_mean', 'balanced_accuracy']
        
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            
            if metric in threshold_metrics.columns:
                axes[row, col].plot(threshold_metrics['threshold'], threshold_metrics[metric])
                axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                axes[row, col].set_xlabel('Threshold')
                axes[row, col].set_ylabel(metric.replace("_", " ").title())
                axes[row, col].grid(True)
                
                # Mark optimal threshold
                optimal_threshold, optimal_value = self.find_optimal_threshold(y_true, y_pred_proba, metric)
                axes[row, col].axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)
                axes[row, col].text(optimal_threshold, optimal_value, f'{optimal_threshold:.2f}', 
                                  rotation=90, va='bottom', ha='right')
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  figsize: Tuple[int, int] = (8, 6)):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = self.precision_recall_auc(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return precision, recall, thresholds
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      figsize: Tuple[int, int] = (8, 6)):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return fpr, tpr, thresholds
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix with annotations
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return cm
    
    def generate_metrics_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold used
            
        Returns:
            Dictionary with comprehensive metrics report
        """
        # Calculate all metrics
        comprehensive_metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        class_wise_metrics = self.calculate_class_wise_metrics(y_true, y_pred)
        
        # Find optimal thresholds for different metrics
        optimal_thresholds = {}
        key_metrics = ['f2_score', 'f1_score', 'g_mean', 'balanced_accuracy']
        
        for metric in key_metrics:
            optimal_threshold, optimal_value = self.find_optimal_threshold(y_true, y_pred_proba, metric)
            optimal_thresholds[metric] = {
                'threshold': optimal_threshold,
                'value': optimal_value
            }
        
        # Classification report
        classification_report_str = classification_report(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        report = {
            'comprehensive_metrics': comprehensive_metrics,
            'class_wise_metrics': class_wise_metrics,
            'optimal_thresholds': optimal_thresholds,
            'classification_report': classification_report_str,
            'confusion_matrix': cm.tolist(),
            'threshold_used': threshold,
            'data_summary': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true)),
                'negative_samples': int(np.sum(1 - y_true)),
                'positive_rate': float(np.mean(y_true))
            }
        }
        
        return report
