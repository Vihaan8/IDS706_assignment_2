#!/usr/bin/env python3
"""
Standalone Performance Benchmark: Pandas vs Polars

This script runs focused performance tests comparing pandas and Polars
operations on the Data Analyst dataset.

Usage:
    python performance_benchmark.py

Author: Vihaan Manchanda
Date: September 6, 2025
"""

import pandas as pd
import polars as pl
import time
import numpy as np
from salary_analysis import (
    extract_salary, extract_salary_polars,
    clean_data, clean_data_polars
)


def detailed_benchmark():
    """Run detailed performance benchmarks"""
    print("🚀 DETAILED PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Load data for both
    print("Loading datasets...")
    try:
        df_pandas = pd.read_csv("DataAnalyst.csv")
        df_polars = pl.read_csv("DataAnalyst.csv")
        print(f"Dataset size: {df_pandas.shape[0]} rows, {df_pandas.shape[1]} columns")
    except FileNotFoundError:
        print("ERROR: DataAnalyst.csv not found!")
        return
    
    results = {}
    
    # Test 1: Data Loading
    print("\n1. 📁 Data Loading Performance")
    print("-" * 30)
    
    # Pandas loading (multiple runs for accuracy)
    pandas_load_times = []
    for i in range(5):
        start = time.time()
        _ = pd.read_csv("DataAnalyst.csv")
        pandas_load_times.append(time.time() - start)
    
    # Polars loading
    polars_load_times = []
    for i in range(5):
        start = time.time()
        _ = pl.read_csv("DataAnalyst.csv")
        polars_load_times.append(time.time() - start)
    
    pandas_load_avg = np.mean(pandas_load_times)
    polars_load_avg = np.mean(polars_load_times)
    
    print(f"Pandas loading: {pandas_load_avg:.4f}s ± {np.std(pandas_load_times):.4f}s")
    print(f"Polars loading: {polars_load_avg:.4f}s ± {np.std(polars_load_times):.4f}s")
    
    results['loading'] = {
        'pandas': pandas_load_avg,
        'polars': polars_load_avg,
        'speedup': pandas_load_avg / polars_load_avg
    }
    
    # Test 2: Data Cleaning
    print("\n2. 🧹 Data Cleaning Performance")
    print("-" * 30)
    
    # Pandas cleaning
    pandas_clean_times = []
    for i in range(3):
        df_test = df_pandas.copy()
        start = time.time()
        _ = clean_data(df_test)
        pandas_clean_times.append(time.time() - start)
    
    # Polars cleaning  
    polars_clean_times = []
    for i in range(3):
        df_test = df_polars.clone()
        start = time.time()
        _ = clean_data_polars(df_test)
        polars_clean_times.append(time.time() - start)
    
    pandas_clean_avg = np.mean(pandas_clean_times)
    polars_clean_avg = np.mean(polars_clean_times)
    
    print(f"Pandas cleaning: {pandas_clean_avg:.4f}s ± {np.std(pandas_clean_times):.4f}s")
    print(f"Polars cleaning: {polars_clean_avg:.4f}s ± {np.std(polars_clean_times):.4f}s")
    
    results['cleaning'] = {
        'pandas': pandas_clean_avg,
        'polars': polars_clean_avg,
        'speedup': pandas_clean_avg / polars_clean_avg
    }
    
    # Test 3: GroupBy Operations
    print("\n3. 📊 GroupBy Performance")
    print("-" * 30)
    
    # Prepare cleaned data
    df_clean_pandas = clean_data(df_pandas.copy())
    df_clean_polars = clean_data_polars(df_polars.clone())
    
    # Pandas groupby
    pandas_groupby_times = []
    for i in range(5):
        start = time.time()
        _ = df_clean_pandas.groupby("Industry")["salary"].agg(["mean", "count"])
        pandas_groupby_times.append(time.time() - start)
    
    # Polars groupby
    polars_groupby_times = []
    for i in range(5):
        start = time.time()
        _ = df_clean_polars.group_by("Industry").agg([
            pl.col("salary").mean().alias("mean"),
            pl.col("salary").count().alias("count")
        ])
        polars_groupby_times.append(time.time() - start)
    
    pandas_groupby_avg = np.mean(pandas_groupby_times)
    polars_groupby_avg = np.mean(polars_groupby_times)
    
    print(f"Pandas groupby: {pandas_groupby_avg:.4f}s ± {np.std(pandas_groupby_times):.4f}s")
    print(f"Polars groupby: {polars_groupby_avg:.4f}s ± {np.std(polars_groupby_times):.4f}s")
    
    results['groupby'] = {
        'pandas': pandas_groupby_avg,
        'polars': polars_groupby_avg,
        'speedup': pandas_groupby_avg / polars_groupby_avg
    }
    
    # Test 4: Memory Usage
    print("\n4. 💾 Memory Usage Comparison")
    print("-" * 30)
    
    pandas_memory = df_clean_pandas.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    # Note: Polars memory usage is harder to measure directly
    print(f"Pandas memory usage: {pandas_memory:.2f} MB")
    print("Polars memory usage: Not directly measurable (generally more efficient)")
    
    # Summary
    print("\n5. 📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    total_pandas = results['loading']['pandas'] + results['cleaning']['pandas'] + results['groupby']['pandas']
    total_polars = results['loading']['polars'] + results['cleaning']['polars'] + results['groupby']['polars']
    overall_speedup = total_pandas / total_polars
    
    print(f"Loading speedup:   {results['loading']['speedup']:.2f}x")
    print(f"Cleaning speedup:  {results['cleaning']['speedup']:.2f}x") 
    print(f"GroupBy speedup:   {results['groupby']['speedup']:.2f}x")
    print(f"Overall speedup:   {overall_speedup:.2f}x")
    
    if overall_speedup > 1:
        print(f"\n🎉 Polars is {overall_speedup:.2f}x faster overall!")
    else:
        print(f"\n📊 Pandas is {1/overall_speedup:.2f}x faster overall")
    
    print(f"\nTotal processing time:")
    print(f"  Pandas: {total_pandas:.4f}s")
    print(f"  Polars: {total_polars:.4f}s")
    
    # Interpretation
    print("\n6. 🔍 INTERPRETATION")
    print("=" * 50)
    print("For this dataset size (~2K records):")
    
    if overall_speedup > 2:
        print("• Polars shows significant performance advantage")
        print("• Benefits would be even greater with larger datasets")
    elif overall_speedup > 1.2:
        print("• Polars shows moderate performance advantage")
        print("• Good foundation for scaling to larger datasets")
    elif overall_speedup > 0.8:
        print("• Performance is roughly equivalent")
        print("• Polars overhead might affect small datasets")
    else:
        print("• Pandas performs better on this small dataset")
        print("• Polars advantages emerge with larger datasets")
    
    print("• Polars typically excels with 100K+ row datasets")
    print("• Memory efficiency improvements are not captured in timing")
    

if __name__ == "__main__":
    detailed_benchmark()