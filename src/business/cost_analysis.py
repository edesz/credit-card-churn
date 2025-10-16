"""
Cost Analysis Module

This module implements comprehensive cost analysis for credit card churn prediction,
including false positive/negative costs, intervention strategies, and profit optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from .clv_calculator import CLVCalculator


class CostAnalyzer:
    """
    Cost Analysis for Credit Card Churn Prediction
    
    Analyzes the business costs associated with different prediction scenarios
    and optimizes decision thresholds based on profit maximization.
    """
    
    def __init__(self, clv_calculator: CLVCalculator):
        """
        Initialize Cost Analyzer
        
        Args:
            clv_calculator: CLVCalculator instance for revenue calculations
        """
        self.clv_calculator = clv_calculator
        
        # Cost parameters
        self.false_negative_cost_multiplier = 1.0  # Full CLV loss
        self.false_positive_cost_multiplier = 0.1  # 10% of intervention cost
        self.true_positive_savings_multiplier = 0.4  # Success rate
        
    def calculate_prediction_costs(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 clv: pd.Series) -> Dict[str, float]:
        """
        Calculate costs for different prediction outcomes
        
        Args:
            y_true: True churn labels
            y_pred: Predicted churn labels
            clv: Customer Lifetime Value
            
        Returns:
            Dictionary with cost breakdown
        """
        # Confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        # Calculate costs
        # False Negatives: Lost revenue from missed churners
        fn_clv_loss = np.sum(clv[(y_true == 1) & (y_pred == 0)])
        fn_cost = fn_clv_loss * self.false_negative_cost_multiplier
        
        # False Positives: Unnecessary intervention costs
        fp_cost = fp * self.clv_calculator.intervention_cost * self.false_positive_cost_multiplier
        
        # True Positives: Saved revenue minus intervention cost
        tp_clv_saved = np.sum(clv[(y_true == 1) & (y_pred == 1)])
        tp_intervention_cost = tp * self.clv_calculator.intervention_cost
        tp_savings = tp_clv_saved * self.true_positive_savings_multiplier
        tp_net_savings = tp_savings - tp_intervention_cost
        
        # True Negatives: No cost (correctly identified as non-churners)
        tn_cost = 0
        
        total_cost = fn_cost + fp_cost - tp_net_savings
        
        cost_breakdown = {
            'true_positives': {
                'count': int(tp),
                'clv_at_risk': float(tp_clv_saved),
                'intervention_cost': float(tp_intervention_cost),
                'savings': float(tp_savings),
                'net_savings': float(tp_net_savings)
            },
            'false_positives': {
                'count': int(fp),
                'unnecessary_cost': float(fp_cost)
            },
            'false_negatives': {
                'count': int(fn),
                'clv_lost': float(fn_clv_loss),
                'cost': float(fn_cost)
            },
            'true_negatives': {
                'count': int(tn),
                'cost': float(tn_cost)
            },
            'total_cost': float(total_cost),
            'total_interventions': int(tp + fp),
            'total_churners': int(tp + fn)
        }
        
        return cost_breakdown
    
    def calculate_threshold_costs(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                clv: pd.Series, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate costs for different decision thresholds
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            thresholds: Array of thresholds to test (optional)
            
        Returns:
            DataFrame with cost analysis for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.01)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            costs = self.calculate_prediction_costs(y_true, y_pred, clv)
            
            # Additional metrics
            precision = costs['true_positives']['count'] / (costs['true_positives']['count'] + costs['false_positives']['count']) if (costs['true_positives']['count'] + costs['false_positives']['count']) > 0 else 0
            recall = costs['true_positives']['count'] / (costs['true_positives']['count'] + costs['false_negatives']['count']) if (costs['true_positives']['count'] + costs['false_negatives']['count']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'total_cost': costs['total_cost'],
                'interventions': costs['total_interventions'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp_count': costs['true_positives']['count'],
                'fp_count': costs['false_positives']['count'],
                'fn_count': costs['false_negatives']['count'],
                'tn_count': costs['true_negatives']['count'],
                'clv_saved': costs['true_positives']['net_savings'],
                'clv_lost': costs['false_negatives']['cost'],
                'unnecessary_cost': costs['false_positives']['unnecessary_cost']
            })
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             clv: pd.Series, objective: str = 'minimize_cost') -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal threshold based on business objective
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            objective: 'minimize_cost', 'maximize_profit', or 'balanced'
            
        Returns:
            Optimal threshold and detailed results
        """
        threshold_costs = self.calculate_threshold_costs(y_true, y_pred_proba, clv)
        
        if objective == 'minimize_cost':
            optimal_idx = threshold_costs['total_cost'].idxmin()
        elif objective == 'maximize_profit':
            # Profit = CLV saved - total costs
            threshold_costs['profit'] = threshold_costs['clv_saved'] - threshold_costs['total_cost']
            optimal_idx = threshold_costs['profit'].idxmax()
        elif objective == 'balanced':
            # Balance between recall and precision
            threshold_costs['balanced_score'] = 2 * (threshold_costs['recall'] * threshold_costs['precision']) / (threshold_costs['recall'] + threshold_costs['precision'])
            optimal_idx = threshold_costs['balanced_score'].idxmax()
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        optimal_threshold = threshold_costs.iloc[optimal_idx]['threshold']
        optimal_results = threshold_costs.iloc[optimal_idx].to_dict()
        
        return optimal_threshold, optimal_results
    
    def analyze_cost_sensitivity(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               clv: pd.Series, cost_multipliers: Optional[Dict[str, List[float]]] = None) -> pd.DataFrame:
        """
        Analyze sensitivity of costs to different cost parameters
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            cost_multipliers: Dictionary with parameter ranges to test
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        if cost_multipliers is None:
            cost_multipliers = {
                'false_negative_cost_multiplier': [0.5, 0.75, 1.0, 1.25, 1.5],
                'false_positive_cost_multiplier': [0.05, 0.1, 0.15, 0.2, 0.25],
                'true_positive_savings_multiplier': [0.2, 0.3, 0.4, 0.5, 0.6]
            }
        
        results = []
        
        # Use optimal threshold for baseline
        optimal_threshold, _ = self.find_optimal_threshold(y_true, y_pred_proba, clv)
        y_pred_baseline = (y_pred_proba >= optimal_threshold).astype(int)
        baseline_costs = self.calculate_prediction_costs(y_true, y_pred_baseline, clv)
        
        # Test sensitivity to each parameter
        for param_name, values in cost_multipliers.items():
            for value in values:
                # Temporarily update parameter
                original_value = getattr(self, param_name)
                setattr(self, param_name, value)
                
                # Recalculate costs
                costs = self.calculate_prediction_costs(y_true, y_pred_baseline, clv)
                
                results.append({
                    'parameter': param_name,
                    'parameter_value': value,
                    'total_cost': costs['total_cost'],
                    'cost_change': costs['total_cost'] - baseline_costs['total_cost'],
                    'cost_change_pct': (costs['total_cost'] - baseline_costs['total_cost']) / abs(baseline_costs['total_cost']) * 100
                })
                
                # Restore original value
                setattr(self, param_name, original_value)
        
        return pd.DataFrame(results)
    
    def plot_cost_analysis(self, threshold_costs: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive cost analysis plots
        
        Args:
            threshold_costs: DataFrame from calculate_threshold_costs
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Cost Analysis by Decision Threshold', fontsize=16)
        
        # Plot 1: Total Cost vs Threshold
        axes[0, 0].plot(threshold_costs['threshold'], threshold_costs['total_cost'])
        axes[0, 0].set_title('Total Cost vs Threshold')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Total Cost')
        axes[0, 0].grid(True)
        
        # Plot 2: Precision and Recall
        axes[0, 1].plot(threshold_costs['threshold'], threshold_costs['precision'], label='Precision')
        axes[0, 1].plot(threshold_costs['threshold'], threshold_costs['recall'], label='Recall')
        axes[0, 1].set_title('Precision and Recall vs Threshold')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Number of Interventions
        axes[0, 2].plot(threshold_costs['threshold'], threshold_costs['interventions'])
        axes[0, 2].set_title('Number of Interventions vs Threshold')
        axes[0, 2].set_xlabel('Threshold')
        axes[0, 2].set_ylabel('Interventions')
        axes[0, 2].grid(True)
        
        # Plot 4: Cost Breakdown
        axes[1, 0].plot(threshold_costs['threshold'], threshold_costs['clv_saved'], label='CLV Saved')
        axes[1, 0].plot(threshold_costs['threshold'], threshold_costs['clv_lost'], label='CLV Lost')
        axes[1, 0].plot(threshold_costs['threshold'], threshold_costs['unnecessary_cost'], label='Unnecessary Cost')
        axes[1, 0].set_title('Cost Breakdown vs Threshold')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Confusion Matrix Elements
        axes[1, 1].plot(threshold_costs['threshold'], threshold_costs['tp_count'], label='TP')
        axes[1, 1].plot(threshold_costs['threshold'], threshold_costs['fp_count'], label='FP')
        axes[1, 1].plot(threshold_costs['threshold'], threshold_costs['fn_count'], label='FN')
        axes[1, 1].set_title('Confusion Matrix Elements vs Threshold')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot 6: F1 Score
        axes[1, 2].plot(threshold_costs['threshold'], threshold_costs['f1'])
        axes[1, 2].set_title('F1 Score vs Threshold')
        axes[1, 2].set_xlabel('Threshold')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_cost_report(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                           clv: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive cost analysis report
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            
        Returns:
            Dictionary with comprehensive cost analysis
        """
        # Find optimal thresholds for different objectives
        cost_optimal_threshold, cost_optimal_results = self.find_optimal_threshold(y_true, y_pred_proba, clv, 'minimize_cost')
        profit_optimal_threshold, profit_optimal_results = self.find_optimal_threshold(y_true, y_pred_proba, clv, 'maximize_profit')
        balanced_optimal_threshold, balanced_optimal_results = self.find_optimal_threshold(y_true, y_pred_proba, clv, 'balanced')
        
        # Calculate threshold costs for plotting
        threshold_costs = self.calculate_threshold_costs(y_true, y_pred_proba, clv)
        
        # Sensitivity analysis
        sensitivity_results = self.analyze_cost_sensitivity(y_true, y_pred_proba, clv)
        
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
                'average_clv': float(clv.mean())
            },
            'optimal_thresholds': {
                'cost_minimization': {
                    'threshold': cost_optimal_threshold,
                    'results': cost_optimal_results
                },
                'profit_maximization': {
                    'threshold': profit_optimal_threshold,
                    'results': profit_optimal_results
                },
                'balanced': {
                    'threshold': balanced_optimal_threshold,
                    'results': balanced_optimal_results
                }
            },
            'recommendations': {
                'recommended_threshold': profit_optimal_threshold,  # Default to profit maximization
                'reasoning': 'Profit maximization balances revenue preservation with intervention costs'
            },
            'sensitivity_analysis': sensitivity_results,
            'threshold_analysis': threshold_costs
        }
        
        return report
