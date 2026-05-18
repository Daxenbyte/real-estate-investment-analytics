"""Data generator."""

import pandas as pd
import numpy as np


def generate_raw_data(n_rows=5000, seed=42):
    """Generate synthetic data with dirty values."""
    np.random.seed(seed)
    
    # Generate base data
    property_ids = np.arange(1, n_rows + 1)
    
    # Generate listing dates over the past 2 years
    start_date = pd.Timestamp.now() - pd.Timedelta(days=730)
    listing_dates = [
        start_date + pd.Timedelta(days=int(x))
        for x in np.random.randint(0, 730, size=n_rows)
    ]
    
    # Generate prices with intentional dirty data:
    # - ~97% valid prices (50k-500k USD)
    # - ~3% negative or zero prices (bad data)
    prices = np.random.uniform(50000, 500000, size=n_rows)
    bad_price_indices = np.random.choice(
        n_rows, size=int(0.03 * n_rows), replace=False
    )
    prices[bad_price_indices] = np.random.choice([-10000, 0, 1], size=len(bad_price_indices))
    
    # Add missing price values (~5%)
    missing_price_indices = np.random.choice(
        n_rows, size=int(0.05 * n_rows), replace=False
    )
    prices[missing_price_indices] = np.nan
    
    # Add a few extreme outliers (10M+ prices)
    outlier_indices = np.random.choice(
        n_rows, size=max(1, int(0.01 * n_rows)), replace=False
    )
    prices[outlier_indices] = np.random.uniform(10_000_000, 50_000_000, size=len(outlier_indices))
    
    # Generate area in square meters with missing values (~7%)
    area_sqm = np.random.uniform(30, 300, size=n_rows)
    missing_area_indices = np.random.choice(
        n_rows, size=int(0.07 * n_rows), replace=False
    )
    area_sqm[missing_area_indices] = np.nan
    
    # Generate city names
    cities = np.random.choice(
        ['Warsaw', 'Krakow', 'Wroclaw', 'Gdansk'], size=n_rows
    )
    
    # Generate number of rooms (1-5)
    rooms = np.random.randint(1, 6, size=n_rows)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'property_id': property_ids,
        'listing_date': listing_dates,
        'price': prices,
        'area_sqm': area_sqm,
        'city': cities,
        'rooms': rooms,
    })
    
    return df


def save_to_csv(df, filename='raw_real_estate_data.csv'):
    df.to_csv(filename, index=False)
    print(f"saved to {filename}: {len(df)} rows, {len(df.columns)} cols")
