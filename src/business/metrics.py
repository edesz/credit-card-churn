"""
Business Metrics Module

This module implements business-specific metrics for credit card churn prediction,
including ROI calculations, profit optimization, and business KPI tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import precision_recall_curve, roc_curve


class BusinessMetrics:
    """
    Business Metrics Calculator for Churn Prediction
    
    Implements business-focused metrics that align with the cost structure
    and objectives of credit card churn prediction.
    """
    
    def __init__(self, intervention_cost: float = 50.0, success_rate: float = 0.4):
        """
        Initialize Business Metrics Calculator
        
        Args:
            intervention_cost: Cost per intervention
            success_rate: Success rate of interventions
        """
        self.intervention_cost = intervention_cost
        self.success_rate = success_rate
    
    def calculate_profit_per_customer(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                    clv: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calculate profit per customer based on prediction and intervention
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            threshold: Decision threshold
            
        Returns:
            Dictionary with profit calculations
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate profit for each prediction type
        profits = []
        
        for i in range(len(y_true)):
            true_label = y_true[i]
            pred_label = y_pred[i]
            customer_clv = clv.iloc[i]
            prob = y_pred_proba[i]
            
            if pred_label == 1:  # We predict churn
                if true_label == 1:  # True positive
                    # Save CLV with success rate, minus intervention cost
                    profit = customer_clv * self.success_rate - self.intervention_cost
                else:  # False positive
                    # Lose intervention cost
                    profit = -self.intervention_cost
            else:  # We predict no churn
                if true_label == 1:  # False negative
                    # Lose full CLV
                    profit = -customer_clv
                else:  # True negative
                    # No cost, no gain
                    profit = 0
            
            profits.append(profit)
        
        profits = np.array(profits)
        
        return {
            'total_profit': float(np.sum(profits)),
            'average_profit': float(np.mean(profits)),
            'profit_per_customer': profits,
            'profitable_customers': int(np.sum(profits > 0)),
            'losing_customers': int(np.sum(profits < 0))
        }
    
    def calculate_roi_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                            clv: pd.Series, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate ROI metrics for the churn prediction model
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            threshold: Decision threshold
            
        Returns:
            Dictionary with ROI metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate costs and revenues
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Intervention costs
        intervention_costs = (tp + fp) * self.intervention_cost
        
        # Revenue saved (successful interventions)
        clv_saved = np.sum(clv[(y_true == 1) & (y_pred == 1)]) * self.success_rate
        
        # Revenue lost (missed churners)
        clv_lost = np.sum(clv[(y_true == 1) & (y_pred == 0)])
        
        # Net profit
        net_profit = clv_saved - intervention_costs
        
        # ROI calculation
        roi = (net_profit / intervention_costs * 100) if intervention_costs > 0 else 0
        
        return {
            'intervention_costs': float(intervention_costs),
            'clv_saved': float(clv_saved),
            'clv_lost': float(clv_lost),
            'net_profit': float(net_profit),
            'roi_percentage': float(roi),
            'total_customers_targeted': int(tp + fp),
            'successful_interventions': int(tp * self.success_rate)
        }
    
    def calculate_customer_value_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                       clv: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calculate customer value preservation metrics
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            threshold: Decision threshold
            
        Returns:
            Dictionary with customer value metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Segment customers
        high_value_customers = clv >= clv.quantile(0.75)
        
        # Calculate metrics for different segments
        segments = {
            'all_customers': np.ones(len(y_true), dtype=bool),
            'high_value_customers': high_value_customers,
            'low_value_customers': ~high_value_customers
        }
        
        segment_metrics = {}
        
        for segment_name, segment_mask in segments.items():
            if not np.any(segment_mask):
                continue
                
            segment_y_true = y_true[segment_mask]
            segment_y_pred = y_pred[segment_mask]
            segment_clv = clv[segment_mask]
            segment_proba = y_pred_proba[segment_mask]
            
            # Calculate segment-specific metrics
            tp = np.sum((segment_y_true == 1) & (segment_y_pred == 1))
            fp = np.sum((segment_y_true == 0) & (segment_y_pred == 1))
            fn = np.sum((segment_y_true == 1) & (segment_y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Value-based metrics
            clv_at_risk = np.sum(segment_clv[segment_y_true == 1])
            clv_protected = np.sum(segment_clv[(segment_y_true == 1) & (segment_y_pred == 1)]) * self.success_rate
            clv_lost = np.sum(segment_clv[(segment_y_true == 1) & (segment_y_pred == 0)])
            
            segment_metrics[segment_name] = {
                'customers_count': int(np.sum(segment_mask)),
                'churners_count': int(np.sum(segment_y_true)),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'clv_at_risk': float(clv_at_risk),
                'clv_protected': float(clv_protected),
                'clv_lost': float(clv_lost),
                'protection_rate': float(clv_protected / clv_at_risk) if clv_at_risk > 0 else 0,
                'average_clv': float(segment_clv.mean()),
                'total_clv': float(segment_clv.sum())
            }
        
        return segment_metrics
    
    def calculate_threshold_optimization_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                               clv: pd.Series, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate business metrics across different thresholds
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            thresholds: Array of thresholds to test
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.01)
        
        results = []
        
        for threshold in thresholds:
            # Calculate all metrics for this threshold
            profit_metrics = self.calculate_profit_per_customer(y_true, y_pred_proba, clv, threshold)
            roi_metrics = self.calculate_roi_metrics(y_true, y_pred_proba, clv, threshold)
            
            # Combine results
            result = {
                'threshold': threshold,
                'total_profit': profit_metrics['total_profit'],
                'average_profit': profit_metrics['average_profit'],
                'roi_percentage': roi_metrics['roi_percentage'],
                'intervention_costs': roi_metrics['intervention_costs'],
                'clv_saved': roi_metrics['clv_saved'],
                'clv_lost': roi_metrics['clv_lost'],
                'net_profit': roi_metrics['net_profit'],
                'customers_targeted': roi_metrics['total_customers_targeted'],
                'successful_interventions': roi_metrics['successful_interventions']
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def find_business_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                      clv: pd.Series, objective: str = 'maximize_profit') -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal threshold based on business objectives
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            objective: 'maximize_profit', 'maximize_roi', or 'minimize_clv_loss'
            
        Returns:
            Optimal threshold and results
        """
        threshold_metrics = self.calculate_threshold_optimization_metrics(y_true, y_pred_proba, clv)
        
        if objective == 'maximize_profit':
            optimal_idx = threshold_metrics['total_profit'].idxmax()
        elif objective == 'maximize_roi':
            optimal_idx = threshold_metrics['roi_percentage'].idxmax()
        elif objective == 'minimize_clv_loss':
            optimal_idx = threshold_metrics['clv_lost'].idxmin()
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        optimal_threshold = threshold_metrics.iloc[optimal_idx]['threshold']
        optimal_results = threshold_metrics.iloc[optimal_idx].to_dict()
        
        return optimal_threshold, optimal_results
    
    def calculate_lift_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             clv: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calculate lift metrics for model performance
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            threshold: Decision threshold
            
        Returns:
            Dictionary with lift metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        
        # Calculate cumulative metrics
        n_customers = len(y_true)
        n_churners = np.sum(y_true)
        
        # Decile analysis
        decile_size = n_customers // 10
        decile_results = []
        
        for decile in range(10):
            start_idx = decile * decile_size
            end_idx = start_idx + decile_size if decile < 9 else n_customers
            
            decile_indices = sorted_indices[start_idx:end_idx]
            decile_y_true = y_true[decile_indices]
            decile_clv = clv.iloc[decile_indices]
            
            decile_churners = np.sum(decile_y_true)
            decile_clv = np.sum(decile_clv)
            
            decile_results.append({
                'decile': decile + 1,
                'customers': len(decile_indices),
                'churners': int(decile_churners),
                'churn_rate': float(decile_churners / len(decile_indices)),
                'total_clv': float(decile_clv),
                'average_clv': float(decile_clv / len(decile_indices))
            })
        
        # Calculate lift
        baseline_churn_rate = n_churners / n_customers
        lift_results = []
        
        for result in decile_results:
            lift = result['churn_rate'] / baseline_churn_rate if baseline_churn_rate > 0 else 0
            result['lift'] = lift
            lift_results.append(result)
        
        return {
            'decile_analysis': lift_results,
            'baseline_churn_rate': float(baseline_churn_rate),
            'top_decile_lift': float(lift_results[0]['lift']) if lift_results else 0,
            'top_three_deciles_lift': float(np.mean([r['lift'] for r in lift_results[:3]])) if len(lift_results) >= 3 else 0
        }
    
    def generate_business_report(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               clv: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive business metrics report
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            
        Returns:
            Dictionary with comprehensive business analysis
        """
        # Find optimal thresholds for different objectives
        profit_threshold, profit_results = self.find_business_optimal_threshold(y_true, y_pred_proba, clv, 'maximize_profit')
        roi_threshold, roi_results = self.find_business_optimal_threshold(y_true, y_pred_proba, clv, 'maximize_roi')
        loss_threshold, loss_results = self.find_business_optimal_threshold(y_true, y_pred_proba, clv, 'minimize_clv_loss')
        
        # Calculate comprehensive metrics at profit-optimal threshold
        profit_metrics = self.calculate_profit_per_customer(y_true, y_pred_proba, clv, profit_threshold)
        roi_metrics = self.calculate_roi_metrics(y_true, y_pred_proba, clv, profit_threshold)
        customer_value_metrics = self.calculate_customer_value_metrics(y_true, y_pred_proba, clv, profit_threshold)
        lift_metrics = self.calculate_lift_metrics(y_true, y_pred_proba, clv, profit_threshold)
        
        # Overall statistics
        total_clv = float(clv.sum())
        total_churners = int(np.sum(y_true))
        churn_rate = float(np.mean(y_true))
        
        report = {
            'overview': {
                'total_customers': len(y_true),
                'total_churners': total_churners,
                'churn_rate': churn_rate,
                'total_clv': total_clv,
                'average_clv': float(clv.mean()),
                'high_value_customers': int(np.sum(clv >= clv.quantile(0.75)))
            },
            'optimal_thresholds': {
                'profit_maximization': {
                    'threshold': profit_threshold,
                    'results': profit_results
                },
                'roi_maximization': {
                    'threshold': roi_threshold,
                    'results': roi_results
                },
                'clv_loss_minimization': {
                    'threshold': loss_threshold,
                    'results': loss_results
                }
            },
            'business_impact': {
                'recommended_threshold': profit_threshold,
                'total_profit': profit_metrics['total_profit'],
                'average_profit_per_customer': profit_metrics['average_profit'],
                'roi_percentage': roi_metrics['roi_percentage'],
                'intervention_costs': roi_metrics['intervention_costs'],
                'clv_saved': roi_metrics['clv_saved'],
                'clv_lost': roi_metrics['clv_lost'],
                'customers_targeted': roi_metrics['total_customers_targeted'],
                'successful_interventions': roi_metrics['successful_interventions']
            },
            'customer_segments': customer_value_metrics,
            'lift_analysis': lift_metrics,
            'recommendations': {
                'threshold': profit_threshold,
                'reasoning': 'Profit maximization provides the best balance of revenue preservation and cost management',
                'expected_impact': f"Expected net profit of ${profit_metrics['total_profit']:.2f} with ROI of {roi_metrics['roi_percentage']:.1f}%"
            }
        }
        
        return report
