#!/usr/bin/env python3
# Author: Vihaan Manchanda
# Date: 2024-06-15
# Updated: 2024-06-15
# Version: 3.0 -> Added more tests and made the test file more comprehensive. 
# New version includes unit, integration, system, and performance tests.


"""
Unit, Integration, System, and Performance tests for salary_analysis.py

Usage:
    make test
    python -m pytest test_salary_analysis.py -v
    python -m pytest test_salary_analysis.py -v --cov=salary_analysis

"""

import unittest
import pandas as pd
import polars as pl
import numpy as np
import time
from unittest.mock import patch
import warnings
warnings.filterwarnings("ignore")

from salary_analysis import (
    load_data,
    extract_salary,
    clean_data,
    clean_data_polars,
    analyze_company_size,
    analyze_industry,
    build_ml_model
)


class TestUnitTests(unittest.TestCase):
    """Unit tests for individual functions"""
    
    def test_data_loading(self):
        """Test data loading with mock CSV"""
        with patch('salary_analysis.pd.read_csv') as mock_read:
            mock_df = pd.DataFrame({
                'Salary Estimate': ['$50K-$70K'] * 10,
                'Rating': [4.2] * 10,
                'Size': ['51-200 employees'] * 10,
                'Industry': ['Technology'] * 10
            })
            mock_read.return_value = mock_df
            
            df = load_data()
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 10)
    
    def test_salary_extraction(self):
        """Test salary extraction from string"""
        self.assertEqual(extract_salary("$50K-$70K"), 60000)
        self.assertEqual(extract_salary("80K-100K"), 90000)
        self.assertTrue(pd.isna(extract_salary("Invalid")))
    
    def test_data_filtering(self):
        """Test data cleaning and filtering"""
        test_data = pd.DataFrame({
            'Salary Estimate': ['$10K-$15K', '$50K-$70K', '$250K-$300K'],
            'Rating': [4.2, 3.8, 4.5],
            'Size': ['51-200 employees'] * 3,
            'Industry': ['Tech'] * 3
        })
        
        df_cleaned = clean_data(test_data)
        # Should only keep middle salary (50-70K) due to 20K-200K filter
        self.assertEqual(len(df_cleaned), 1)
        self.assertEqual(df_cleaned['salary'].iloc[0], 60000)
    
    def test_machine_learning_model(self):
        """Test ML model building"""
        np.random.seed(42)
        test_data = pd.DataFrame({
            'Size': ['51-200 employees'] * 60 + ['1001-5000 employees'] * 60,
            'rating': np.random.uniform(2.0, 5.0, 120),
            'salary': np.random.uniform(40000, 120000, 120),
            'Industry': ['Tech'] * 120
        })
        
        result = build_ml_model(test_data)
        self.assertIsNotNone(result)
        self.assertIn('importance', result.columns)


class TestIntegrationTests(unittest.TestCase):
    """Integration tests for component interactions"""
    
    def test_load_clean_analyze_pipeline(self):
        """Test data loading -> cleaning -> analysis integration"""
        test_data = pd.DataFrame({
            'Salary Estimate': ['$50K-$70K'] * 20 + ['$80K-$100K'] * 20,
            'Rating': [4.2] * 40,
            'Size': ['51-200 employees'] * 20 + ['1001-5000 employees'] * 20,
            'Industry': ['Technology'] * 20 + ['Healthcare'] * 20
        })
        
        with patch('salary_analysis.pd.read_csv') as mock_read:
            mock_read.return_value = test_data
            
            # Load -> Clean -> Analyze
            df = load_data()
            df_clean = clean_data(df)
            size_result = analyze_company_size(df_clean)
            
            self.assertIsNotNone(df)
            self.assertIn('salary', df_clean.columns)
            self.assertIsNotNone(size_result)
            self.assertEqual(len(size_result), 2)
    
    def test_pandas_polars_consistency(self):
        """Test that pandas and polars produce consistent results"""
        test_data = {
            'Salary Estimate': ['$50K-$70K'] * 10 + ['$80K-$100K'] * 10,
            'Rating': [4.2] * 20,
            'Size': ['51-200 employees'] * 10 + ['1001-5000 employees'] * 10,
            'Industry': ['Technology'] * 10 + ['Healthcare'] * 10
        }
        
        df_pandas = pd.DataFrame(test_data)
        df_polars = pl.DataFrame(test_data)
        
        clean_pandas = clean_data(df_pandas)
        clean_polars = clean_data_polars(df_polars)
        
        # Should produce same number of records
        self.assertEqual(len(clean_pandas), clean_polars.height)
    
    def test_groupby_with_ml(self):
        """Test analysis functions feeding into ML model"""
        test_data = pd.DataFrame({
            'salary': [60000] * 30 + [90000] * 30,
            'Size': ['51-200 employees'] * 30 + ['1001-5000 employees'] * 30,
            'Industry': ['Technology'] * 30 + ['Healthcare'] * 30,
            'rating': np.random.uniform(3.0, 5.0, 60)
        })
        
        # Analyze then build model
        size_result = analyze_company_size(test_data)
        industry_result = analyze_industry(test_data)
        ml_result = build_ml_model(test_data)
        
        self.assertIsNotNone(size_result)
        self.assertIsNotNone(industry_result)
        self.assertIsNotNone(ml_result)


class TestSystemTests(unittest.TestCase):
    """End-to-end system tests"""
    
    def test_full_analysis_pipeline(self):
        """Test complete system from CSV to ML insights"""
        np.random.seed(42)
        
        # Create realistic test dataset
        test_data = pd.DataFrame({
            'Salary Estimate': ['$50K-$70K'] * 30 + ['$70K-$90K'] * 30 + 
                              ['$90K-$110K'] * 30 + ['$110K-$130K'] * 30,
            'Rating': np.random.uniform(2.0, 5.0, 120),
            'Size': ['51-200 employees'] * 30 + ['201-500 employees'] * 30 + 
                    ['501-1000 employees'] * 30 + ['1001-5000 employees'] * 30,
            'Industry': ['Technology'] * 40 + ['Healthcare'] * 40 + 
                        ['Finance'] * 20 + ['Retail'] * 20,
            'Company Name': [f'Company_{i}' for i in range(120)]
        })
        
        with patch('salary_analysis.pd.read_csv') as mock_read:
            mock_read.return_value = test_data
            
            # Complete pipeline
            df = load_data()
            self.assertEqual(len(df), 120)
            
            df_clean = clean_data(df)
            self.assertTrue(len(df_clean) > 0)
            self.assertTrue(all(df_clean['salary'] >= 20000))
            self.assertTrue(all(df_clean['salary'] <= 200000))
            
            size_analysis = analyze_company_size(df_clean)
            self.assertEqual(len(size_analysis), 4)
            
            industry_analysis = analyze_industry(df_clean)
            self.assertEqual(len(industry_analysis), 4)
            
            ml_model = build_ml_model(df_clean)
            self.assertIsNotNone(ml_model)
            self.assertTrue(ml_model['importance'].sum() > 0.9)
    
    def test_edge_cases_handling(self):
        """Test system handles edge cases gracefully"""
        # Empty dataset with required columns
        empty_df = pd.DataFrame({
            'Salary Estimate': [],
            'Rating': [],
            'Size': [],
            'Industry': []
        })
        clean_empty = clean_data(empty_df)
        self.assertEqual(len(clean_empty), 0)
        
        # All invalid salaries
        invalid_df = pd.DataFrame({
            'Salary Estimate': ['Invalid'] * 5,
            'Rating': [4.0] * 5,
            'Size': ['51-200 employees'] * 5,
            'Industry': ['Tech'] * 5
        })
        clean_invalid = clean_data(invalid_df)
        self.assertEqual(len(clean_invalid), 0)
        
        # Single record
        single_df = pd.DataFrame({
            'Salary Estimate': ['$50K-$70K'],
            'salary': [60000],
            'Rating': [4.2],
            'rating': [4.2],
            'Size': ['51-200 employees'],
            'Industry': ['Technology']
        })
        size_result = analyze_company_size(single_df)
        self.assertIsNotNone(size_result)
        self.assertEqual(len(size_result), 1)


class TestPerformanceTests(unittest.TestCase):
    """Performance comparison tests"""
    
    def test_pandas_vs_polars_speed(self):
        """Test that both pandas and polars complete in reasonable time"""
        # Create test data
        test_data_dict = {
            'Salary Estimate': ['$50K-$70K'] * 100,
            'Rating': [4.2] * 100,
            'Size': ['51-200 employees'] * 50 + ['1001-5000 employees'] * 50,
            'Industry': ['Technology'] * 50 + ['Healthcare'] * 50
        }
        
        # Time pandas
        df_pandas = pd.DataFrame(test_data_dict)
        start = time.time()
        clean_pandas = clean_data(df_pandas.copy())
        pandas_time = time.time() - start
        
        # Time polars
        df_polars = pl.DataFrame(test_data_dict)
        start = time.time()
        clean_polars = clean_data_polars(df_polars.clone())
        polars_time = time.time() - start
        
        # Both should complete quickly (under 1 second for small data)
        self.assertLess(pandas_time, 1.0)
        self.assertLess(polars_time, 1.0)
        
        # Print performance comparison
        print(f"\nPandas: {pandas_time:.4f}s, Polars: {polars_time:.4f}s")
        if polars_time < pandas_time:
            print(f"Polars is {pandas_time/polars_time:.2f}x faster")
        else:
            print(f"Pandas is {polars_time/pandas_time:.2f}x faster")


if __name__ == '__main__':
    unittest.main(verbosity=2)