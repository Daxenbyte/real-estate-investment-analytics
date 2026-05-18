"""
Monte Carlo simulations for price forecasting.
"""

import numpy as np


class MarketSimulator:
    """
    Monte Carlo simulator with geometric brownian motion.
    """
    
    def __init__(self, drift: float = 0.05, volatility: float = 0.15) -> None:
        self.drift = drift
        self.volatility = volatility
    
    def run_monte_carlo(
        self, 
        current_prices: np.ndarray, 
        years: int, 
        simulations: int,
        seed: int = 42
    ) -> np.ndarray:
        """
        Run GBM simulations. Returns 3D array (simulations, years+1, properties).
        """
        np.random.seed(seed)
        
        n_properties = len(current_prices)
        n_steps = years  # Number of steps per simulation
        dt = 1 / 252  # Daily resampling (252 trading days per year)
        
        # Pre-allocate output array
        # Index: [simulation, time_step, property]
        price_paths = np.zeros((simulations, n_steps + 1, n_properties), dtype=np.float64)
        
        # Initialize first time step with current prices
        price_paths[:, 0, :] = current_prices[np.newaxis, :]
        
        # Precompute drift and volatility coefficients for all steps
        drift_component = (self.drift - 0.5 * self.volatility ** 2) * dt
        volatility_component = self.volatility * np.sqrt(dt)
        
        # Generate all random numbers upfront (vectorized)
        # Shape: (simulations, n_steps, n_properties)
        random_shocks = np.random.standard_normal((simulations, n_steps, n_properties))
        
        # Vectorized GBM: S_{t+1} = S_t * exp(drift_term + volatility_term * Z)
        # Loop over time steps (cannot be fully vectorized due to sequential dependency)
        for t in range(n_steps):
            exponent = drift_component + volatility_component * random_shocks[:, t, :]
            price_paths[:, t + 1, :] = price_paths[:, t, :] * np.exp(exponent)
        
        return price_paths
    
    def calculate_roi(
        self,
        purchase_prices: np.ndarray,
        future_prices: np.ndarray,
        rental_yield: float = 0.05,
        years: int = 10
    ) -> np.ndarray:
        """
        Calculate ROI including rental income.
        Returns 2D array (simulations, properties).
        """
        # Extract final prices (last time step) from trajectories
        # Shape: (simulations, n_properties)
        sale_prices = future_prices[:, -1, :]
        
        # Calculate total rental income
        rental_income = purchase_prices[np.newaxis, :] * rental_yield * years
        
        # Vectorized ROI calculation
        # Numerator: (Sale - Purchase + Rental) for all simulations and properties
        numerator = (sale_prices - purchase_prices[np.newaxis, :]) + rental_income
        
        # Denominator: Purchase price for all simulations
        denominator = purchase_prices[np.newaxis, :]
        
        # ROI as percentage
        roi_pct = (numerator / denominator) * 100
        
        return roi_pct
    
    def get_statistics(
        self,
        roi_array: np.ndarray,
        confidence_level: float = 0.95
    ):
        """
        Calculate ROI statistics: mean, median, std, percentiles.
        """
        # Calculate mean across simulations (aggregate per property)
        mean_roi_per_property = roi_array.mean(axis=0)
        
        stats = {
            'mean': np.mean(roi_array),
            'median': np.median(roi_array),
            'std': np.std(roi_array),
            'p5': np.percentile(roi_array, 5),
            'p25': np.percentile(roi_array, 25),
            'p50': np.percentile(roi_array, 50),
            'p75': np.percentile(roi_array, 75),
            'p95': np.percentile(roi_array, 95),
            'custom_percentile': np.percentile(roi_array, confidence_level * 100),
            'min': np.min(roi_array),
            'max': np.max(roi_array),
            'mean_per_property': mean_roi_per_property  # For detailed city-level analysis
        }
        
        return stats
