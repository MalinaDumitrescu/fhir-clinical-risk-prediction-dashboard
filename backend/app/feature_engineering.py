"""
Feature cleaning and engineering for the FHIR risk model.

Responsibilities:
1. Remove constant/useless features (condition_count)
2. Add temporal features (count, std, slope, etc.)
3. Create missingness indicators
4. Report feature changes
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats


def detect_constant_features(X, threshold=1):
    """
    Detect features with only 1 unique value (or very few).
    
    Parameters
    ----------
    X : DataFrame
        Feature matrix
    threshold : int
        Consider feature constant if n_unique <= threshold
        
    Returns
    -------
    list : column names that are constant
    """
    constant_features = []
    for col in X.columns:
        nunique = X[col].nunique()
        if nunique <= threshold:
            constant_features.append(col)
    return constant_features


def remove_constant_features(X, features_to_remove=None):
    """
    Remove constant features.
    
    Parameters
    ----------
    X : DataFrame
        Feature matrix
    features_to_remove : list, optional
        Specific features to remove. If None, auto-detect.
        
    Returns
    -------
    X_clean : DataFrame
        Features with constants removed
    removed_features : list
        Features that were removed
    """
    if features_to_remove is None:
        features_to_remove = detect_constant_features(X)
    
    if not features_to_remove:
        return X.copy(), []
    
    # Preserve non-target features
    removal_list = [f for f in features_to_remove if f not in ['target_long_icu_stay', 'y']]
    
    if removal_list:
        print(f"Removing {len(removal_list)} constant features:")
        for f in removal_list:
            print(f"  - {f}")
    
    return X.drop(columns=removal_list, errors='ignore'), removal_list


class TemporalFeatureEngineer:
    """
    Create temporal and distributional features from raw observations.
    
    This is a placeholder for the improved feature generation.
    In practice, this would:
    1. Load raw FHIR observations for each patient
    2. Filter to first 24 hours
    3. Compute summary statistics
    4. Create trend features
    
    For now, it documents what COULD be done.
    """
    
    # Features that could be added if raw observation data available:
    POSSIBLE_FEATURES = {
        'heart_rate': [
            'heart_rate_count_24h',           # number of measurements
            'heart_rate_std_24h',             # variability
            'heart_rate_median_24h',          # robust central tendency
            'heart_rate_range_24h',           # max - min
            'heart_rate_slope_24h',           # trend over time
            'heart_rate_first_24h',           # first measurement
            'heart_rate_last_24h',            # last measurement
            'heart_rate_delta_24h',           # last - first
        ],
        'systolic_bp': [
            'systolic_bp_count_24h',
            'systolic_bp_std_24h',
            'systolic_bp_median_24h',
            'systolic_bp_range_24h',
            'systolic_bp_slope_24h',
        ],
        'respiratory_rate': [
            'respiratory_rate_count_24h',
            'respiratory_rate_std_24h',
            'respiratory_rate_range_24h',
            'respiratory_rate_slope_24h',
        ],
        'creatinine': [
            'creatinine_count_24h',
            'creatinine_first_24h',
            'creatinine_last_24h',
            'creatinine_delta_24h',  # worsening kidney function?
        ],
        'lactate': [
            'lactate_count_24h',
            'lactate_max_24h',  # already available
            'lactate_trend_24h',  # rising = concerning
        ],
    }
    
    @staticmethod
    def add_missingness_indicators(X, high_missingness_threshold=0.30):
        """
        Add binary indicators for high-missingness features.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix
        high_missingness_threshold : float
            Consider feature high-missingness if > this proportion
            
        Returns
        -------
        X_with_indicators : DataFrame
            Original features + missingness indicators
        indicators_added : list
            Names of new indicator columns
        """
        X_copy = X.copy()
        indicators_added = []
        
        for col in X.columns:
            missing_pct = X[col].isna().sum() / len(X)
            if missing_pct > high_missingness_threshold:
                indicator_col = f'{col}_is_missing'
                X_copy[indicator_col] = X[col].isna().astype(int)
                indicators_added.append(indicator_col)
        
        if indicators_added:
            print(f"Added {len(indicators_added)} missingness indicators:")
            for ind in indicators_added:
                print(f"  - {ind}")
        
        return X_copy, indicators_added
    
    @staticmethod
    def report_changes(original_features, new_features, removed_features, added_features):
        """
        Generate a summary report of feature engineering changes.
        
        Returns
        -------
        dict : summary statistics
        """
        return {
            'original_count': len(original_features),
            'final_count': len(new_features),
            'removed_count': len(removed_features),
            'added_count': len(added_features),
            'removed_features': removed_features,
            'added_features': added_features,
            'net_change': len(new_features) - len(original_features),
        }


def prepare_features_for_training(df, remove_constants=True, add_indicators=True):
    """
    Clean and prepare features for training.
    
    Parameters
    ----------
    df : DataFrame
        Raw feature matrix with target
    remove_constants : bool
        Remove constant features
    add_indicators : bool
        Add missingness indicators for high-missingness features
        
    Returns
    -------
    X_clean : DataFrame
        Prepared feature matrix
    y : Series
        Target variable
    changes : dict
        Summary of changes made
    """
    y = df['target_long_icu_stay'].copy()
    X = df.drop(columns=['target_long_icu_stay'])
    
    original_features = list(X.columns)
    removed_features = []
    added_features = []
    
    # Remove constant features
    if remove_constants:
        X, removed = remove_constant_features(X)
        removed_features.extend(removed)
    
    # Add missingness indicators
    if add_indicators:
        X, added = TemporalFeatureEngineer.add_missingness_indicators(X)
        added_features.extend(added)
    
    final_features = list(X.columns)
    
    changes = TemporalFeatureEngineer.report_changes(
        original_features, final_features, removed_features, added_features
    )
    
    return X, y, changes
