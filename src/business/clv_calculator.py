"""
Customer Lifetime Value (CLV) Calculator

This module implements the CLV calculation framework as defined in the project proposal,
including revenue streams from interchange fees, interest income, and annual fees.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class CLVCalculator:
    """
    Customer Lifetime Value Calculator
    
    Implements the CLV calculation framework from the project proposal:
    - Interchange Revenue = Total_Trans_Amt * r (2%)
    - Interest Revenue = Total_Revolving_Bal * apr (18%)
    - Fee Revenue = Based on card category
    - Annual Revenue = Sum of all revenue streams
    - CLV = Annual Revenue * multiplier (T=3 years, d=0.9 loyalty factor)
    """
    
    def __init__(self, 
                 interchange_rate: float = 0.02,
                 apr: float = 0.18,
                 expected_tenure: float = 3.0,
                 loyalty_discount: float = 0.9,
                 intervention_cost: float = 50.0,
                 success_rate: float = 0.40,
                 cac: float = 200.0):
        """
        Initialize CLV Calculator with business parameters
        
        Args:
            interchange_rate: Bank's interchange fee rate (default 2%)
            apr: Annual percentage rate for revolving balance (default 18%)
            expected_tenure: Expected remaining tenure in years (default 3)
            loyalty_discount: Loyalty discount factor (default 0.9)
            intervention_cost: Cost to prevent churn (default $50)
            success_rate: Success rate of interventions (default 40%)
            cac: Customer acquisition cost (default $200)
        """
        self.interchange_rate = interchange_rate
        self.apr = apr
        self.expected_tenure = expected_tenure
        self.loyalty_discount = loyalty_discount
        self.intervention_cost = intervention_cost
        self.success_rate = success_rate
        self.cac = cac
        
        # Calculate multiplier for CLV
        self.multiplier = self._calculate_multiplier()
        
        # Card category fees
        self.card_fees = {
            'Blue': 0,
            'Silver': 50,
            'Gold': 100,
            'Platinum': 200
        }
    
    def _calculate_multiplier(self) -> float:
        """
        Calculate CLV multiplier based on tenure and loyalty discount
        
        Formula: multiplier = (1 - d^T) / (1 - d)
        where d = loyalty_discount, T = expected_tenure
        """
        if self.loyalty_discount == 1.0:
            return self.expected_tenure
        else:
            return (1 - self.loyalty_discount ** self.expected_tenure) / (1 - self.loyalty_discount)
    
    def calculate_interchange_revenue(self, total_trans_amt: pd.Series) -> pd.Series:
        """
        Calculate interchange revenue from transaction volume
        
        Args:
            total_trans_amt: Total transaction amount per customer
            
        Returns:
            Interchange revenue per customer
        """
        return total_trans_amt * self.interchange_rate
    
    def calculate_interest_revenue(self, total_revolving_bal: pd.Series) -> pd.Series:
        """
        Calculate interest revenue from revolving balance
        
        Args:
            total_revolving_bal: Total revolving balance per customer
            
        Returns:
            Interest revenue per customer
        """
        return total_revolving_bal * self.apr
    
    def calculate_fee_revenue(self, card_category: pd.Series) -> pd.Series:
        """
        Calculate annual fee revenue based on card category
        
        Args:
            card_category: Card category for each customer
            
        Returns:
            Annual fee revenue per customer
        """
        return card_category.map(self.card_fees)
    
    def calculate_annual_revenue(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate total annual revenue per customer
        
        Args:
            df: DataFrame with customer data including required columns
            
        Returns:
            Annual revenue per customer
        """
        # Ensure we have the required columns (support both cases)
        required_cols_map = {
            'Total_Trans_Amt': ['Total_Trans_Amt', 'total_trans_amt'],
            'Total_Revolving_Bal': ['Total_Revolving_Bal', 'total_revolv_bal', 'total_revolving_bal'],
            'Card_Category': ['Card_Category', 'card_category']
        }
        
        # Find actual column names
        actual_cols = {}
        for key, possible_names in required_cols_map.items():
            found = False
            for name in possible_names:
                if name in df.columns:
                    actual_cols[key] = name
                    found = True
                    break
            if not found:
                raise ValueError(f"Missing required column: {key} (tried: {possible_names})")
        
        # Calculate each revenue stream
        interchange_rev = self.calculate_interchange_revenue(df[actual_cols['Total_Trans_Amt']])
        interest_rev = self.calculate_interest_revenue(df[actual_cols['Total_Revolving_Bal']])
        fee_rev = self.calculate_fee_revenue(df[actual_cols['Card_Category']])
        
        # Total annual revenue
        annual_revenue = interchange_rev + interest_rev + fee_rev
        
        return annual_revenue
    
    def calculate_clv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Customer Lifetime Value for each customer
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            CLV per customer
        """
        annual_revenue = self.calculate_annual_revenue(df)
        clv = annual_revenue * self.multiplier
        
        return clv
    
    def calculate_expected_savings(self, y_pred_proba: np.ndarray, clv: pd.Series) -> np.ndarray:
        """
        Calculate expected savings from intervention
        
        Formula: Expected Savings = y_pred_proba * s * CLV - c
        where s = success_rate, c = intervention_cost
        
        Args:
            y_pred_proba: Probability of churn for each customer
            clv: Customer Lifetime Value for each customer
            
        Returns:
            Expected savings per customer
        """
        expected_savings = y_pred_proba * self.success_rate * clv - self.intervention_cost
        
        return expected_savings
    
    def calculate_roi(self, expected_savings: np.ndarray, intervention_costs: Optional[np.ndarray] = None) -> float:
        """
        Calculate Return on Investment
        
        Args:
            expected_savings: Expected savings array
            intervention_costs: Intervention costs array (optional)
            
        Returns:
            ROI as a percentage
        """
        if intervention_costs is None:
            # Assume all customers with positive expected savings get intervention
            intervention_costs = np.where(expected_savings > 0, self.intervention_cost, 0)
        
        total_savings = np.sum(expected_savings[expected_savings > 0])
        total_costs = np.sum(intervention_costs)
        
        if total_costs == 0:
            return 0.0
        
        roi = (total_savings - total_costs) / total_costs * 100
        
        return roi
    
    def optimize_threshold_by_profit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   clv: pd.Series) -> Tuple[float, Dict[str, Any]]:
        """
        Optimize prediction threshold to maximize profit
        
        Args:
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            clv: Customer Lifetime Value
            
        Returns:
            Optimal threshold and results dictionary
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        profits = []
        metrics = []
        
        for threshold in thresholds:
            # Predict churn based on threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate expected savings for predicted churners
            expected_savings = self.calculate_expected_savings(y_pred_proba, clv)
            
            # Calculate profit (only for customers we predict as churners)
            profit = np.sum(expected_savings * y_pred)
            
            # Calculate business metrics
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics.append({
                'threshold': threshold,
                'profit': profit,
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
            
            profits.append(profit)
        
        # Find optimal threshold
        optimal_idx = np.argmax(profits)
        optimal_threshold = thresholds[optimal_idx]
        optimal_metrics = metrics[optimal_idx]
        
        return optimal_threshold, optimal_metrics
    
    def analyze_customer_segments(self, df: pd.DataFrame, y_pred_proba: np.ndarray) -> pd.DataFrame:
        """
        Analyze different customer segments by CLV and churn probability
        
        Args:
            df: Customer data DataFrame
            y_pred_proba: Predicted churn probabilities
            
        Returns:
            DataFrame with segment analysis
        """
        # Calculate CLV
        clv = self.calculate_clv(df)
        
        # Create segments based on CLV and churn probability
        df_analysis = df.copy()
        df_analysis['CLV'] = clv
        df_analysis['Churn_Probability'] = y_pred_proba
        
        # Define segments
        high_clv_threshold = clv.quantile(0.75)
        high_risk_threshold = 0.5
        
        def get_segment(row):
            if row['CLV'] >= high_clv_threshold and row['Churn_Probability'] >= high_risk_threshold:
                return 'High Value High Risk'
            elif row['CLV'] >= high_clv_threshold and row['Churn_Probability'] < high_risk_threshold:
                return 'High Value Low Risk'
            elif row['CLV'] < high_clv_threshold and row['Churn_Probability'] >= high_risk_threshold:
                return 'Low Value High Risk'
            else:
                return 'Low Value Low Risk'
        
        df_analysis['Segment'] = df_analysis.apply(get_segment, axis=1)
        
        # Calculate segment statistics
        segment_stats = df_analysis.groupby('Segment').agg({
            'CLV': ['count', 'mean', 'sum'],
            'Churn_Probability': 'mean',
            'Total_Trans_Amt': 'mean',
            'Total_Revolving_Bal': 'mean'
        }).round(2)
        
        return segment_stats
    
    def generate_business_report(self, df: pd.DataFrame, y_true: np.ndarray, 
                               y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive business analysis report
        
        Args:
            df: Customer data DataFrame
            y_true: True churn labels
            y_pred_proba: Predicted churn probabilities
            
        Returns:
            Dictionary with business analysis results
        """
        # Calculate CLV for all customers
        clv = self.calculate_clv(df)
        
        # Calculate expected savings
        expected_savings = self.calculate_expected_savings(y_pred_proba, clv)
        
        # Optimize threshold
        optimal_threshold, optimal_metrics = self.optimize_threshold_by_profit(y_true, y_pred_proba, clv)
        
        # Calculate ROI
        roi = self.calculate_roi(expected_savings)
        
        # Segment analysis
        segment_stats = self.analyze_customer_segments(df, y_pred_proba)
        
        # Business impact calculations
        total_clv_at_risk = np.sum(clv[y_pred_proba >= optimal_threshold])
        total_intervention_cost = np.sum(expected_savings[y_pred_proba >= optimal_threshold] > 0) * self.intervention_cost
        total_expected_savings = np.sum(expected_savings[y_pred_proba >= optimal_threshold])
        
        report = {
            'clv_statistics': {
                'mean_clv': float(clv.mean()),
                'median_clv': float(clv.median()),
                'total_clv': float(clv.sum()),
                'high_value_customers': int((clv >= clv.quantile(0.75)).sum())
            },
            'business_impact': {
                'optimal_threshold': optimal_threshold,
                'customers_to_target': int(np.sum(y_pred_proba >= optimal_threshold)),
                'total_clv_at_risk': total_clv_at_risk,
                'total_intervention_cost': total_intervention_cost,
                'total_expected_savings': total_expected_savings,
                'net_profit': total_expected_savings - total_intervention_cost,
                'roi_percentage': roi
            },
            'optimal_metrics': optimal_metrics,
            'segment_analysis': segment_stats,
            'parameters': {
                'interchange_rate': self.interchange_rate,
                'apr': self.apr,
                'expected_tenure': self.expected_tenure,
                'loyalty_discount': self.loyalty_discount,
                'intervention_cost': self.intervention_cost,
                'success_rate': self.success_rate,
                'cac': self.cac
            }
        }
        
        return report
