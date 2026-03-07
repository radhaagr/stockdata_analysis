"""
Smart Zero Imputation Strategy
================================

BEFORE imputing zeros, identify:
1. Tickers completely missing this column (all dates NaN)
2. Tickers partially missing this column (some dates NaN)
3. Tickers with all data present

Different handling for each case
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SmartZeroImputation:
    """
    Intelligently impute zeros only where appropriate
    
    Process:
    1. Analyze missing patterns BY TICKER (across all dates)
    2. Classify tickers into 3 groups
    3. Handle each group differently
    4. Add flags to track imputation
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        df: DataFrame with multi-index (ticker, date) or ticker in index
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.analysis = {}
        
    def analyze_missing_by_ticker(self, column: str) -> Dict:
        """
        Analyze missing data patterns for a column by ticker
        
        Returns:
        --------
        Dict with:
        - completely_missing: Tickers with 100% missing
        - partially_missing: Tickers with some data
        - fully_present: Tickers with no missing
        """
        
        print(f"\n{'='*80}")
        print(f"ANALYZING MISSING PATTERN: {column}")
        print(f"{'='*80}")
        
        if 'ticker' in self.df.columns:
            # If ticker is a column
            grouped = self.df.groupby('ticker')[column]
        elif isinstance(self.df.index, pd.MultiIndex):
            # If ticker is in multi-index
            grouped = self.df.groupby(level='ticker')[column]
        else:
            # If ticker is the main index
            grouped = self.df[column]
            if not isinstance(grouped.index, pd.Index):
                grouped = pd.Series({idx: self.df.loc[idx, column] 
                                    for idx in self.df.index})
        
        analysis = {
            'completely_missing': [],    # 100% NaN for all dates
            'partially_missing': [],     # Some NaN, some data
            'fully_present': [],         # No NaN
            'completely_missing_count': 0,
            'partially_missing_count': 0,
            'fully_present_count': 0,
        }
        
        print(f"\nAnalyzing {len(grouped.groups) if hasattr(grouped, 'groups') else 'tickers'}...")
        
        for ticker, ticker_data in grouped:
            missing_count = ticker_data.isnull().sum()
            total_count = len(ticker_data)
            missing_pct = missing_count / total_count * 100
            
            if missing_count == total_count:
                # 100% missing
                analysis['completely_missing'].append(ticker)
                analysis['completely_missing_count'] += 1
                status = "COMPLETELY MISSING"
            elif missing_count > 0:
                # Partially missing
                analysis['partially_missing'].append(ticker)
                analysis['partially_missing_count'] += 1
                status = f"PARTIALLY MISSING ({missing_pct:.1f}%)"
            else:
                # Fully present
                analysis['fully_present'].append(ticker)
                analysis['fully_present_count'] += 1
                status = "FULLY PRESENT"
            
            print(f"  {ticker:10s}: {status:35s} ({missing_count}/{total_count} missing)")
        
        # Print summary
        print(f"\n{'─'*80}")
        print(f"SUMMARY FOR '{column}':")
        print(f"{'─'*80}")
        print(f"Completely missing (all dates NaN): {analysis['completely_missing_count']} tickers")
        if analysis['completely_missing']:
            print(f"  {', '.join(analysis['completely_missing'][:10])}{'...' if len(analysis['completely_missing']) > 10 else ''}")
        
        print(f"\nPartially missing (some dates NaN): {analysis['partially_missing_count']} tickers")
        if analysis['partially_missing']:
            print(f"  {', '.join(analysis['partially_missing'][:10])}{'...' if len(analysis['partially_missing']) > 10 else ''}")
        
        print(f"\nFully present (no missing): {analysis['fully_present_count']} tickers")
        
        return analysis
    
    def get_strategy_recommendation(self, column: str, analysis: Dict) -> Dict:
        """
        Recommend strategy for each group
        """
        
        print(f"\n{'='*80}")
        print(f"STRATEGY RECOMMENDATIONS: {column}")
        print(f"{'='*80}")
        
        strategy = {
            'completely_missing': {
                'action': 'DECIDE BY DOMAIN KNOWLEDGE',
                'reason': 'Ticker has NO data for this column across ALL dates',
                'options': [
                    {
                        'choice': 'ZERO',
                        'description': 'Fill with 0 if this ticker legitimately has zero',
                        'example': f'{column} = 0 (e.g., non-financial has no interestIncome)',
                        'confidence': 'MEDIUM-HIGH',
                        'code': 'df.loc[ticker, column] = 0.0'
                    },
                    {
                        'choice': 'CROSS-SECTIONAL MEDIAN',
                        'description': 'Use median from similar companies',
                        'example': f'Use median {column} of same industry',
                        'confidence': 'MEDIUM',
                        'code': 'df.loc[ticker, column] = industry_median'
                    },
                    {
                        'choice': 'DROP WITH FLAG',
                        'description': 'Keep column but flag as unreliable for this ticker',
                        'example': f'{column}_is_completely_missing[ticker] = 1',
                        'confidence': 'HIGH',
                        'code': 'df[f"{column}_ticker_completely_missing"] = flag'
                    },
                    {
                        'choice': 'EXCLUDE TICKER',
                        'description': 'Remove ticker from analysis entirely',
                        'example': 'df = df[df.ticker != ticker_with_no_data]',
                        'confidence': 'SAFE',
                        'code': 'df = df.drop(ticker, level=0)'
                    }
                ]
            },
            'partially_missing': {
                'action': 'FORWARD/BACKWARD FILL or CROSS-SECTIONAL',
                'reason': 'Ticker has some data but not all dates',
                'options': [
                    {
                        'choice': 'FORWARD FILL',
                        'description': 'Use previous year value to fill current year',
                        'example': '2022 missing → Use 2021 value',
                        'confidence': 'HIGH',
                        'code': 'df.loc[ticker].fillna(method="ffill", limit=1)'
                    },
                    {
                        'choice': 'BACKWARD FILL',
                        'description': 'Use next year value to fill current year',
                        'example': '2021 missing → Use 2022 value',
                        'confidence': 'MEDIUM',
                        'code': 'df.loc[ticker].fillna(method="bfill", limit=1)'
                    },
                    {
                        'choice': 'ZERO FILL',
                        'description': 'Fill missing with 0 (only if partial is legitimate)',
                        'example': f'{column} = 0 (company didn\'t have this in that year)',
                        'confidence': 'MEDIUM',
                        'code': 'df.loc[ticker, column].fillna(0.0)'
                    },
                    {
                        'choice': 'INTERPOLATE',
                        'description': 'Estimate missing value between known values',
                        'example': '2021: 100, 2022: missing, 2023: 200 → 2022 ≈ 150',
                        'confidence': 'LOW-MEDIUM',
                        'code': 'df.loc[ticker].interpolate(method="linear")'
                    }
                ]
            },
            'fully_present': {
                'action': 'NO IMPUTATION NEEDED',
                'reason': 'Ticker has complete data for all dates',
                'options': [
                    {
                        'choice': 'NO ACTION',
                        'description': 'Use data as-is',
                        'example': 'All years have values',
                        'confidence': 'PERFECT',
                        'code': 'pass'
                    }
                ]
            }
        }
        
        return strategy
    
    def print_recommendations(self, column: str, analysis: Dict):
        """Print detailed recommendations"""
        recommendations = self.get_strategy_recommendation(column, analysis)
        
        for group, rec in recommendations.items():
            print(f"\n{group.upper().replace('_', ' ')}:")
            print(f"  Action: {rec['action']}")
            print(f"  Reason: {rec['reason']}")
            print(f"\n  Options:")
            
            for i, opt in enumerate(rec['options'], 1):
                print(f"\n    {i}. {opt['choice']}")
                print(f"       Description: {opt['description']}")
                print(f"       Example: {opt['example']}")
                print(f"       Confidence: {opt['confidence']}")
                print(f"       Code: {opt['code']}")


# ==================== IMPLEMENTATION ====================

class SmartZeroImputer:
    """
    Execute smart zero imputation with full tracking
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.imputation_log = {}
        
    def smart_impute_zero(self, column: str, 
                         strategy_map: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Impute zeros intelligently by ticker
        
        Parameters:
        -----------
        column : str
            Column to impute
        strategy_map : dict
            Mapping of {ticker: strategy} for custom handling
            Example: {'AAPL': 'forward_fill', 'MSFT': 'zero', 'GOOGL': 'skip'}
        """
        
        print(f"\n{'='*80}")
        print(f"SMART ZERO IMPUTATION: {column}")
        print(f"{'='*80}")
        
        # Analyze
        analyzer = SmartZeroImputation(self.df)
        analysis = analyzer.analyze_missing_by_ticker(column)
        
        imputation_log = {
            'column': column,
            'completely_missing_count': 0,
            'partially_missing_count': 0,
            'fully_present_count': 0,
            'imputed_rows': 0,
            'flags_created': False,
            'actions': []
        }
        
        # Get DataFrame grouping
        if 'ticker' in self.df.columns:
            grouped_col = self.df.groupby('ticker')[column]
        elif isinstance(self.df.index, pd.MultiIndex):
            grouped_col = self.df.groupby(level='ticker')[column]
        else:
            print("⚠ Warning: Cannot determine ticker grouping")
            return self.df, imputation_log
        
        # STEP 1: Handle COMPLETELY MISSING tickers
        print(f"\n{'─'*80}")
        print(f"STEP 1: Handle COMPLETELY MISSING tickers ({len(analysis['completely_missing'])})")
        print(f"{'─'*80}")
        
        for ticker in analysis['completely_missing']:
            if strategy_map and ticker in strategy_map:
                strategy = strategy_map[ticker]
            else:
                # Default: Ask user or use domain knowledge
                # For now: Zero (can be overridden)
                strategy = 'zero'
            
            if strategy == 'zero':
                # Fill with zero
                mask = self._get_ticker_mask(ticker)
                self.df.loc[mask, column] = 0.0
                print(f"  ✓ {ticker}: Filled with 0 (all {mask.sum()} rows)")
                imputation_log['actions'].append({
                    'ticker': ticker,
                    'group': 'completely_missing',
                    'action': 'zero_fill',
                    'rows_imputed': mask.sum()
                })
                imputation_log['completely_missing_count'] += 1
                imputation_log['imputed_rows'] += mask.sum()
                
            elif strategy == 'skip':
                # Leave as NaN, will handle later or skip
                print(f"  ⊘ {ticker}: Skipped (will handle later)")
                imputation_log['actions'].append({
                    'ticker': ticker,
                    'group': 'completely_missing',
                    'action': 'skipped',
                    'rows_imputed': 0
                })
                
            elif strategy == 'drop':
                # Remove ticker
                mask = self._get_ticker_mask(ticker)
                self.df = self.df.drop(self.df[mask].index)
                print(f"  ✗ {ticker}: Removed ({mask.sum()} rows dropped)")
                imputation_log['actions'].append({
                    'ticker': ticker,
                    'group': 'completely_missing',
                    'action': 'dropped',
                    'rows_imputed': 0
                })
        
        # STEP 2: Handle PARTIALLY MISSING tickers
        print(f"\n{'─'*80}")
        print(f"STEP 2: Handle PARTIALLY MISSING tickers ({len(analysis['partially_missing'])})")
        print(f"{'─'*80}")
        
        for ticker in analysis['partially_missing']:
            if strategy_map and ticker in strategy_map:
                strategy = strategy_map[ticker]
            else:
                # Default: Forward fill (most appropriate for time-series)
                strategy = 'forward_fill'
            
            if strategy == 'forward_fill':
                # Forward fill for this ticker
                mask = self._get_ticker_mask(ticker)
                before = self.df.loc[mask, column].isnull().sum()
                self.df.loc[mask, column] = self.df.loc[mask, column].fillna(method='ffill')
                after = self.df.loc[mask, column].isnull().sum()
                filled = before - after
                
                print(f"  ⬆  {ticker}: Forward filled {filled} rows")
                imputation_log['actions'].append({
                    'ticker': ticker,
                    'group': 'partially_missing',
                    'action': 'forward_fill',
                    'rows_imputed': filled
                })
                imputation_log['partially_missing_count'] += 1
                imputation_log['imputed_rows'] += filled
                
            elif strategy == 'backward_fill':
                # Backward fill for this ticker
                mask = self._get_ticker_mask(ticker)
                before = self.df.loc[mask, column].isnull().sum()
                self.df.loc[mask, column] = self.df.loc[mask, column].fillna(method='bfill')
                after = self.df.loc[mask, column].isnull().sum()
                filled = before - after
                
                print(f"  ⬇  {ticker}: Backward filled {filled} rows")
                imputation_log['actions'].append({
                    'ticker': ticker,
                    'group': 'partially_missing',
                    'action': 'backward_fill',
                    'rows_imputed': filled
                })
                imputation_log['partially_missing_count'] += 1
                imputation_log['imputed_rows'] += filled
                
            elif strategy == 'zero':
                # Fill remaining with zero
                mask = self._get_ticker_mask(ticker)
                before = self.df.loc[mask, column].isnull().sum()
                self.df.loc[mask, column] = self.df.loc[mask, column].fillna(0.0)
                filled = before - self.df.loc[mask, column].isnull().sum()
                
                print(f"  0  {ticker}: Filled {filled} rows with 0")
                imputation_log['actions'].append({
                    'ticker': ticker,
                    'group': 'partially_missing',
                    'action': 'zero_fill',
                    'rows_imputed': filled
                })
                imputation_log['partially_missing_count'] += 1
                imputation_log['imputed_rows'] += filled
        
        # STEP 3: Handle FULLY PRESENT tickers
        print(f"\n{'─'*80}")
        print(f"STEP 3: FULLY PRESENT tickers ({len(analysis['fully_present'])})")
        print(f"{'─'*80}")
        print(f"  ✓ No action needed ({len(analysis['fully_present'])} tickers with complete data)")
        imputation_log['fully_present_count'] = len(analysis['fully_present'])
        
        # STEP 4: Create tracking flags
        print(f"\n{'─'*80}")
        print(f"STEP 4: Create missing flags for tracking")
        print(f"{'─'*80}")
        
        flag_col = f'{column}_was_missing'
        self.df[flag_col] = self.original_df[column].isnull().astype(int)
        
        print(f"  ✓ Created flag column: {flag_col}")
        print(f"    Total flagged: {self.df[flag_col].sum()} rows")
        imputation_log['flags_created'] = True
        
        return self.df, imputation_log
    
    def _get_ticker_mask(self, ticker: str) -> pd.Series:
        """Get boolean mask for a specific ticker"""
        if 'ticker' in self.df.columns:
            return self.df['ticker'] == ticker
        elif isinstance(self.df.index, pd.MultiIndex):
            return self.df.index.get_level_values('ticker') == ticker
        else:
            # Assume ticker is the main index
            return pd.Series([idx == ticker for idx in self.df.index], 
                           index=self.df.index)
    
    def print_imputation_summary(self, log: Dict):
        """Print summary of imputation"""
        print(f"\n{'='*80}")
        print(f"IMPUTATION SUMMARY: {log['column']}")
        print(f"{'='*80}")
        
        print(f"\nTicker Groups Processed:")
        print(f"  Completely missing: {log['completely_missing_count']}")
        print(f"  Partially missing: {log['partially_missing_count']}")
        print(f"  Fully present: {log['fully_present_count']}")
        
        print(f"\nTotal rows imputed: {log['imputed_rows']}")
        print(f"Tracking flags created: {log['flags_created']}")
        
        print(f"\nDetailed actions:")
        for action in log['actions'][:10]:
            print(f"  • {action['ticker']:10s} ({action['group']:20s}): "
                  f"{action['action']:15s} → {action['rows_imputed']:3d} rows")
        
        if len(log['actions']) > 10:
            print(f"  ... and {len(log['actions']) - 10} more actions")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    
    print("\n" + "█"*80)
    print("█" + " "*15 + "SMART ZERO IMPUTATION BY TICKER" + " "*33 + "█")
    print("█"*80)
    
    print("""
WORKFLOW:
=========

1. ANALYZE MISSING PATTERNS
   ├─ Completely missing (all dates): Decide by domain knowledge
   ├─ Partially missing (some dates): Use time-series fill
   └─ Fully present: No action needed

2. APPLY SMART IMPUTATION
   ├─ Completely missing → Zero or Skip or Drop
   ├─ Partially missing → Forward fill or Backward fill or Zero
   └─ Fully present → Keep as-is

3. CREATE TRACKING FLAGS
   └─ Track which values were imputed for transparency

4. LOG & VERIFY
   └─ Document all imputation actions


EXAMPLE CODE:
=============

# Create imputer
imputer = SmartZeroImputer(df)

# Define custom strategy for each column
strategy_map = {
    'AAPL': 'forward_fill',      # Has some data, fill gaps
    'MSFT': 'zero',              # No data, fill with zero
    'GOOGL': 'skip',             # No data, keep as NaN
    'AMZN': 'drop',              # No data, remove ticker
}

# Impute with strategy
df_clean, log = imputer.smart_impute_zero('researchAndDevelopment', 
                                          strategy_map=strategy_map)

# Print results
imputer.print_imputation_summary(log)

# Analyze before imputing
analyzer = SmartZeroImputation(df)
analysis = analyzer.analyze_missing_by_ticker('interestIncome')
analyzer.print_recommendations('interestIncome', analysis)
    """)
