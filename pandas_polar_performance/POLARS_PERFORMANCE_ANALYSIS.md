# Polars Performance Analysis

This project includes **Polars** implementations alongside the existing pandas code for performance comparison. The core functionality remains unchanged, but you can see how Polars performs compared to pandas.

## Key Additions

### Functions Added
- `load_data_polars()` - Load CSV using Polars
- `extract_salary_polars()` - Polars-compatible salary extraction
- `clean_data_polars()` - Data cleaning using Polars operations
- `analyze_company_size_polars()` - Company size analysis with Polars
- `analyze_industry_polars()` - Industry analysis with Polars
- `performance_comparison()` - **Main performance comparison function**

### Enhanced Testing
- Added tests for Polars functions
- Added consistency tests between pandas and Polars results

## Performance Comparison Features

The `performance_comparison()` function benchmarks:

1. **Data Loading Time**: CSV reading performance
2. **Data Cleaning Time**: Complex transformations and filtering
3. **GroupBy Operations**: Aggregation performance
4. **Total Processing Time**: Overall workflow comparison

## How to Run

### Standard Analysis
```bash
make all
```

### Performance Difference Results

```
==================================================
PERFORMANCE COMPARISON: PANDAS vs POLARS
==================================================

1. Data Loading Performance:
   Pandas loading time: 0.058 seconds
   Polars loading time: 0.039 seconds

2. Data Cleaning Performance:
   Pandas cleaning time: 0.003 seconds
   Polars cleaning time: 0.011 seconds

3. GroupBy Operations Performance:
   Pandas groupby time: 0.001 seconds
   Polars groupby time: 0.007 seconds

4. Performance Summary:
   Total Pandas time: 0.063 seconds
   Total Polars time: 0.057 seconds
   🚀 Polars is 1.11x faster than Pandas!
==================================================
```
**Key Findings:** Polars achieved a modest 1.11x speedup overall, primarily due to faster CSV loading, while pandas performed better on the smaller data processing operations. This demonstrates that for small datasets (~2K records), the performance differences are minimal, but Polars shows promise for scaling to larger datasets where its advantages would be more pronounced.
```