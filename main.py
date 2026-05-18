"""
Main Orchestrator Script

Coordinates the entire real estate investment analysis workflow:
1. Generates synthetic housing market data
2. Cleans and preprocesses the data
3. Runs Monte Carlo simulations for each city
4. Calculates ROI and generates business insights
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_generator import generate_raw_data, save_to_csv
from data_processor import RealEstateProcessor
from simulator import MarketSimulator


def load_config(config_path: str = 'config.json') -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_simulation_for_city(
    city_prices: np.ndarray,
    simulator: MarketSimulator,
    config: dict,
    city_name: str
) -> dict:
    """
    Run Monte Carlo simulation for a specific city and calculate ROI.
    
    Parameters
    ----------
    city_prices : np.ndarray
        Current prices for properties in the city.
    simulator : MarketSimulator
        Initialized simulator with drift and volatility.
    config : dict
        Configuration dictionary.
    city_name : str
        Name of the city (for reporting).
    
    Returns
    -------
    dict
        Results dictionary with simulation outcomes and statistics.
    """
    sim_config = config['simulation']
    years = sim_config['years']
    simulations = sim_config['simulations']
    rental_yield = sim_config['rental_yield']
    confidence_level = sim_config['confidence_level']
    
    # Run Monte Carlo
    print(f"\n  Running {simulations} simulations for {city_name} "
          f"({len(city_prices)} properties)...")
    future_prices = simulator.run_monte_carlo(
        city_prices,
        years=years,
        simulations=simulations
    )
    
    # Calculate ROI
    roi_array = simulator.calculate_roi(
        city_prices,
        future_prices,
        rental_yield=rental_yield,
        years=years
    )
    
    # Get statistics
    roi_stats = simulator.get_statistics(roi_array, confidence_level=confidence_level)
    
    # Calculate price growth statistics
    final_prices = future_prices[:, -1, :]  # (simulations, n_properties)
    price_growth = (final_prices - city_prices[np.newaxis, :]) / city_prices[np.newaxis, :] * 100
    
    growth_stats = {
        'mean_growth_pct': np.mean(price_growth),
        'median_growth_pct': np.median(price_growth),
        'confidence_interval_low': np.percentile(price_growth, (1 - confidence_level) / 2 * 100),
        'confidence_interval_high': np.percentile(price_growth, (1 + confidence_level) / 2 * 100),
        'p5': np.percentile(price_growth, 5),
        'p95': np.percentile(price_growth, 95),
    }
    
    return {
        'city': city_name,
        'n_properties': len(city_prices),
        'current_avg_price': np.mean(city_prices),
        'future_avg_price': np.mean(final_prices),
        'roi_stats': roi_stats,
        'growth_stats': growth_stats,
    }


def print_market_summary(summary_df: pd.DataFrame) -> None:
    print("\nMarket summary by city:")
    print(summary_df.to_string())


def print_simulation_results(all_results: list, confidence_level: float) -> None:
    print("\nSimulation results:")
    
    for result in all_results:
        city = result['city']
        roi_stats = result['roi_stats']
        growth_stats = result['growth_stats']
        current_avg = result['current_avg_price']
        future_avg = result['future_avg_price']
        confidence_pct = int(confidence_level * 100)
        
        print(f"\n{city}")
        print(f"  Properties: {result['n_properties']}")
        print(f"  Current avg: ${current_avg:,.0f}")
        print(f"  Future avg (10y): ${future_avg:,.0f}")
        print(f"  Price growth median: {growth_stats['median_growth_pct']:+.2f}%")
        print(f"  Price growth {confidence_pct}% CI: "
              f"{growth_stats['confidence_interval_low']:.2f}% - "
              f"{growth_stats['confidence_interval_high']:.2f}%")
        print(f"  ROI median: {roi_stats['median']:.2f}%")
        print(f"  ROI mean: {roi_stats['mean']:.2f}% (std {roi_stats['std']:.2f}%)")


def print_investment_recommendations(all_results: list) -> None:
    print("\nRanked by expected ROI:")
    
    sorted_by_roi = sorted(
        all_results,
        key=lambda x: x['roi_stats']['median'],
        reverse=True
    )
    
    for i, result in enumerate(sorted_by_roi, 1):
        city = result['city']
        median_roi = result['roi_stats']['median']
        print(f"  {i}. {city}: ROI {median_roi:.2f}%")
    
    best_city = sorted_by_roi[0]
    worst_city = sorted_by_roi[-1]
    all_median_rois = [r['roi_stats']['median'] for r in all_results]
    
    print(f"\nBest: {best_city['city']} ({best_city['roi_stats']['median']:.2f}%)")
    print(f"Worst: {worst_city['city']} ({worst_city['roi_stats']['median']:.2f}%)")
    print(f"Average: {np.mean(all_median_rois):.2f}%")


def main():
    config = load_config('config.json')
    
    print("step 1: generating data...")
    gen_config = config['data_generation']
    df_raw = generate_raw_data(
        n_rows=gen_config['n_rows'],
        seed=gen_config['seed']
    )
    
    csv_filename = gen_config['csv_filename']
    save_to_csv(df_raw, csv_filename)
    
    print("step 2: cleaning data...")
    processor = RealEstateProcessor(csv_filename)
    processor.clean()
    processor.feature_engineer()
    df_clean = processor.get_data()
    
    # Print market summary
    market_summary = processor.get_market_summary()
    print_market_summary(market_summary)
    
    print("step 3: running simulations...")
    sim_config = config['simulation']
    simulator = MarketSimulator(
        drift=sim_config['drift'],
        volatility=sim_config['volatility']
    )
    
    all_results = []
    cities = df_clean['city'].unique()
    
    for city in sorted(cities):
        city_data = df_clean[df_clean['city'] == city]
        city_prices = city_data['price'].values
        
        result = run_simulation_for_city(
            city_prices,
            simulator,
            config,
            city
        )
        all_results.append(result)
    
    print_simulation_results(all_results, sim_config['confidence_level'])
    print_investment_recommendations(all_results)
    
    print("Analysis complete.")


if __name__ == '__main__':
    main()
