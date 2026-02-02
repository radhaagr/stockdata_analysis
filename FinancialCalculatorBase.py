
"""
Financial Ratio Calculation Utilities
Reusable base class and utilities for safe financial calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from typing import Tuple, List

# ============================================================================
# BASE UTILITY CLASS - All Calculators Inherit This
# ============================================================================

class FinancialCalculatorBase:
    """
    Abstract base class for all financial ratio calculators
    Provides common utilities: safe_get, safe_divide, interpolation
    """
    
    def __init__(self, ticker: str, data: pd.DataFrame, 
                 min_threshold: float = 1e6, max_ratio: float = 100.0):
        """
        Args:
            ticker: Stock ticker symbol
            data: Financial data (balance sheet, income statement, etc.)
            min_threshold: Values below this treated as zero
            max_ratio: Maximum allowed ratio value (for capping)
        """
        self.ticker = ticker
        self.data = data
        self.MIN_THRESHOLD = min_threshold
        self.MAX_RATIO = max_ratio
    
    # ========================================================================
    # SAFE DATA ACCESS METHODS
    # ========================================================================
    
    def safe_get(self, key: str, default: float = 0.0) -> pd.Series:
        """
        Safely retrieve a row from financial data
        
        Args:
            key: Row label (e.g., 'Total Revenue', 'Current Assets')
            default: Default value if key doesn't exist
        
        Returns:
            pd.Series with data or default values
        """
        if key in self.data.index:
            return self.data.loc[key]
        else:
            # Return series of default values with same index as data columns
            return pd.Series(default, index=self.data.columns)
    
    def safe_divide(self, numerator: pd.Series, denominator: pd.Series,
                    max_value: Optional[float] = None,
                    strategy: str = 'nan') -> pd.Series:
        """
        Safely divide two series with comprehensive error handling
        
        Args:
            numerator: Numerator series
            denominator: Denominator series
            max_value: Maximum allowed value (default: self.MAX_RATIO)
            strategy: How to handle zero denominator
                     - 'nan': Mark as NaN (for later interpolation)
                     - 'cap': Cap at max_value
                     - 'zero': Set to 0
        
        Returns:
            pd.Series with safely calculated ratios
        """
        if max_value is None:
            max_value = self.MAX_RATIO
        
        result = pd.Series(index=numerator.index, dtype=float)
        
        for idx in numerator.index:
            num = numerator[idx]
            denom = denominator[idx]
            
            # Handle missing values
            if pd.isna(num) or pd.isna(denom):
                result[idx] = np.nan
                continue
            
            # Handle zero/near-zero denominator
            if abs(denom) < self.MIN_THRESHOLD:
                if strategy == 'nan':
                    result[idx] = np.nan
                elif strategy == 'cap':
                    result[idx] = max_value if num > 0 else -max_value
                elif strategy == 'zero':
                    result[idx] = 0.0
                else:
                    result[idx] = np.nan
                continue
            
            # Normal division
            value = num / denom
            
            # Handle infinity
            if np.isinf(value):
                result[idx] = max_value if value > 0 else -max_value
            else:
                # Optional: Cap at max_value
                if abs(value) > max_value:
                    result[idx] = max_value if value > 0 else -max_value
                else:
                    result[idx] = value
        
        return result
    
    # ========================================================================
    # DATA QUALITY & INTERPOLATION METHODS
    # ========================================================================
    
    def interpolate_missing(self, series: pd.Series, 
                           ratio_name: str,
                           default: Optional[float] = None) -> pd.Series:
        """
        Interpolate missing values in time series
        
        Strategy:
        1. Linear interpolation for gaps
        2. Forward fill for leading NaN
        3. Backward fill for trailing NaN
        4. Use default if still NaN
        
        Args:
            series: Series with potential NaN values
            ratio_name: Name of ratio (for default lookup)
            default: Default value if all else fails
        
        Returns:
            Series with NaN values filled
        """
        original_nan_count = series.isna().sum()
        
        if original_nan_count == 0:
            return series
        
        # Sort by date (assuming index is dates)
        series = series.sort_index()
        
        # Linear interpolation
        series = series.interpolate(method='linear', limit_direction='both')
        
        # Forward/backward fill
        series = series.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN, use default
        if series.isna().any():
            if default is None:
                # Use built-in defaults
                default = self._get_default_value(ratio_name)
            series = series.fillna(default)
        
        filled_count = original_nan_count - series.isna().sum()
        if filled_count > 0:
            print(f"       → Interpolated {filled_count} missing values")
        
        return series
    
    def _get_default_value(self, ratio_name: str) -> float:
        """
        Get reasonable default value for a ratio
        Override in subclasses for ratio-specific defaults
        """
        defaults = {
            # Liquidity ratios
            'Current_Ratio': 1.5,
            'Quick_Ratio': 1.2,
            'Cash_Ratio': 0.5,
            
            # Leverage ratios
            'Debt_to_Equity': 0.5,
            'Debt_to_Assets': 0.4,
            'Equity_Ratio': 0.6,
            
            # Profitability margins
            'Net_Profit_Margin': 15.0,
            'Gross_Profit_Margin': 40.0,
            'Operating_Margin': 20.0,
            
            # Efficiency ratios
            'ROE': 15.0,
            'ROA': 8.0,
            'Asset_Turnover': 1.0,
        }
        
        return defaults.get(ratio_name, 1.0)
    
    def check_missing_data(self, *keys: str) -> Tuple[bool, List[str]]:
        """
        Check if required data keys exist and have values
        
        Args:
            *keys: Row labels to check
        
        Returns:
            (all_present: bool, missing_keys: List[str])
        """
        missing = []
        
        for key in keys:
            if key not in self.data.index:
                missing.append(key)
            elif self.data.loc[key].isna().all():
                missing.append(key)
        
        return len(missing) == 0, missing
    
    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================
    
    def validate_range(self, series: pd.Series, 
                      min_value: float = None,
                      max_value: float = None) -> pd.Series:
        """
        Clip series to valid range
        
        Args:
            series: Series to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        
        Returns:
            Clipped series
        """
        if min_value is not None and max_value is not None:
            return series.clip(lower=min_value, upper=max_value)
        elif min_value is not None:
            return series.clip(lower=min_value)
        elif max_value is not None:
            return series.clip(upper=max_value)
        else:
            return series
    
    def validate_constraint(self, series1: pd.Series, series2: pd.Series,
                          constraint: str) -> Tuple[bool, int]:
        """
        Validate constraint between two series
        
        Args:
            series1: First series
            series2: Second series
            constraint: Type of constraint ('<=', '>=', '==', 'sum_to_1')
        
        Returns:
            (all_valid: bool, violation_count: int)
        """
        violations = 0
        
        for idx in series1.index:
            val1 = series1[idx]
            val2 = series2[idx]
            
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            if constraint == '<=':
                if val1 > val2:
                    violations += 1
            elif constraint == '>=':
                if val1 < val2:
                    violations += 1
            elif constraint == '==':
                if abs(val1 - val2) > 1e-6:
                    violations += 1
            elif constraint == 'sum_to_1':
                if abs(val1 + val2 - 1.0) > 1e-2:
                    violations += 1
        
        return violations == 0, violations
    
    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================
    
    @abstractmethod
    def calculate_all_ratios(self) -> Dict[str, pd.Series]:
        """
        Calculate all ratios for this calculator
        Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate data quality report
        Must be implemented by subclasses
        """
        pass
