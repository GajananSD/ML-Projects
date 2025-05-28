from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    A custom transformer that caps outliers in numeric data using the IQR method.
    
    This transformer:
    1. During fit: Calculates lower and upper bounds for each feature based on IQR
    2. During transform: Caps values outside these bounds to the bound values
    
    Works with pandas DataFrames and follows the scikit-learn transformer API.
    """

    def fit(self, X, y=None):
        """
        Calculate the IQR bounds for each feature in the input data.
    
        Parameters:
        -----------
        X : pandas DataFrame (Input features to calculate bounds from)
        y : None (Ignored, present for scikit-learn API compatibility)
            
        Returns:
        --------
        self : OutlierCapper (The fitted transformer object)
        """
        # Initialize empty list to store bounds for each feature
        self.bounds = []

        for i in range(X.shape[1]):
            Q1 = X.iloc[:, i].quantile(0.25)    # First quartile (25th percentile)
            Q3 = X.iloc[:, i].quantile(0.75)    # Third quartile (75th percentile)
            IQR = Q3 - Q1                       # Interquartile range
            # Calculate lower and upper bounds (1.5*IQR is the typical threshold)
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            # Store bounds for this feature
            self.bounds.append((lower, upper))
        return self

    def transform(self, X):
        """
        Cap outliers in the input data using the bounds calculated during fit.
        
        Parameters:
        -----------
        X : pandas DataFrame  (Input features to transform)
            
        Returns:
        --------
        X : pandas DataFrame (Transformed data with outliers capped)
        """
        X = X.copy()
        for i in range(X.shape[1]):
            lower, upper = self.bounds[i]   # Get bounds for this feature
            # Clip values - anything below lower bound becomes lower, anything above upper becomes upper
            X.iloc[:, i] = X.iloc[:, i].clip(lower, upper)
        return X
