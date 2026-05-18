"""Data processor."""

import pandas as pd
import numpy as np


class RealEstateProcessor:
    """
    Load, clean, and transform real estate data.
    """
    
    def __init__(self, filepath: str) -> None:
        self._df = pd.read_csv(filepath)
        print(f"loaded {len(self._df)} rows from {filepath}")
    
    def clean(self):
        """Clean data: remove bad prices, outliers, impute missing."""
        initial_rows = len(self._df)
        
        # Step 1: Remove invalid prices (<=0)
        self._df = self._df[self._df['price'] > 0].copy()
        print(f"  Removed {initial_rows - len(self._df)} rows with price <= 0")
        initial_rows = len(self._df)
        
        # Step 2: Remove price outliers using IQR method
        Q1 = self._df['price'].quantile(0.25)
        Q3 = self._df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self._df = self._df[
            (self._df['price'] >= lower_bound) & (self._df['price'] <= upper_bound)
        ].copy()
        print(f"  Removed {initial_rows - len(self._df)} rows with price outliers")
        initial_rows = len(self._df)
        
        # Step 3: Impute missing area_sqm using city + rooms median
        # Group by city and rooms, calculate median area
        area_impute = self._df.groupby(['city', 'rooms'])['area_sqm'].transform('median')
        self._df['area_sqm'] = self._df['area_sqm'].fillna(area_impute)
        
        # For any remaining missing areas, use city median
        city_area_median = self._df.groupby('city')['area_sqm'].transform('median')
        self._df['area_sqm'] = self._df['area_sqm'].fillna(city_area_median)
        
        missing_before = self._df['area_sqm'].isna().sum()
        if missing_before > 0:
            # Last resort: use global median
            self._df['area_sqm'].fillna(self._df['area_sqm'].median(), inplace=True)
            print(f"  Imputed missing area_sqm values")
        
        # Step 4: Impute missing price using city median
        price_impute = self._df.groupby('city')['price'].transform('median')
        self._df['price'] = self._df['price'].fillna(price_impute)
        
        # Step 5: Remove any remaining NaN rows
        rows_before = len(self._df)
        self._df = self._df.dropna()
        print(f"  Removed {rows_before - len(self._df)} rows with remaining NaN values")
        
        print(f"cleaned: {len(self._df)} rows")
    
    def feature_engineer(self):
        """Add features: price_per_sqm, year, month."""
        self._df['listing_date'] = pd.to_datetime(self._df['listing_date'])
        self._df['price_per_sqm'] = self._df['price'] / self._df['area_sqm']
        self._df['year'] = self._df['listing_date'].dt.year
        self._df['month'] = self._df['listing_date'].dt.month
        print("added features: price_per_sqm, year, month")
    
    def get_data(self) -> pd.DataFrame:
        """
        Return the cleaned and engineered DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Processed real estate data.
        """
        return self._df.copy()
    
    def get_market_summary(self) -> pd.DataFrame:
        """
        Market stats aggregated by city.
        """
        summary = self._df.groupby('city').agg({
            'price_per_sqm': ['mean', 'median'],
            'price': 'mean',
            'property_id': 'count'
        }).round(2)
        
        # Flatten multi-level column names
        summary.columns = [
            'avg_price_per_sqm',
            'median_price_per_sqm',
            'avg_price',
            'total_volume'
        ]
        
        # Sort by avg_price_per_sqm descending
        summary = summary.sort_values('avg_price_per_sqm', ascending=False)
        
        return summary
