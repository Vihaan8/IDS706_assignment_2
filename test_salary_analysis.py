#!/usr/bin/env python3
"""
Simple unit tests for salary_analysis.py

Usage:
    python -m pytest -vv --cov=salary_analysis test_salary_analysis.py
"""

import unittest
import pandas as pd
import polars as pl
import numpy as np
from salary_analysis import (
    extract_salary,
    extract_salary_polars,
    clean_data,
    clean_data_polars,
)


class TestSalaryAnalysis(unittest.TestCase):
    """Simple test cases for salary analysis functions"""

    def test_extract_salary_valid_ranges(self):
        """Test salary extraction with valid salary ranges"""
        # Test K notation ranges
        result1 = extract_salary("$50K-$70K (Glassdoor est.)")
        self.assertEqual(result1, 60000)

        # Test without $ signs
        result2 = extract_salary("80K-100K")
        self.assertEqual(result2, 90000)

    def test_extract_salary_polars_valid_ranges(self):
        """Test Polars salary extraction with valid salary ranges"""
        # Test K notation ranges
        result1 = extract_salary_polars("$50K-$70K (Glassdoor est.)")
        self.assertEqual(result1, 60000)

        # Test without $ signs
        result2 = extract_salary_polars("80K-100K")
        self.assertEqual(result2, 90000)

    def test_extract_salary_invalid_inputs(self):
        """Test salary extraction with invalid inputs"""
        # Test NaN input
        result1 = extract_salary(np.nan)
        self.assertTrue(pd.isna(result1))

        # Test invalid string
        result2 = extract_salary("No salary information")
        self.assertTrue(pd.isna(result2))

    def test_extract_salary_polars_invalid_inputs(self):
        """Test Polars salary extraction with invalid inputs"""
        # Test None input
        result1 = extract_salary_polars(None)
        self.assertIsNone(result1)

        # Test invalid string
        result2 = extract_salary_polars("No salary information")
        self.assertIsNone(result2)

    def test_clean_data_basic(self):
        """Test basic data cleaning functionality with pandas"""
        # Create simple test data
        sample_data = {
            "Salary Estimate": ["$50K-$70K", "$80K-$100K", "Invalid"],
            "Rating": [4.2, 3.8, -1],
            "Size": ["51-200 employees", "1001-5000 employees", "-1"],
            "Industry": ["Technology", "Healthcare", "Finance"],
        }
        df_sample = pd.DataFrame(sample_data)

        # Test cleaning
        df_cleaned = clean_data(df_sample)

        # Check that salary column was added
        self.assertIn("salary", df_cleaned.columns)

        # Check that some valid salaries were extracted
        self.assertTrue(df_cleaned["salary"].notna().any())

    def test_clean_data_polars_basic(self):
        """Test basic data cleaning functionality with Polars"""
        # Create simple test data
        sample_data = {
            "Salary Estimate": ["$50K-$70K", "$80K-$100K", "Invalid"],
            "Rating": [4.2, 3.8, -1],
            "Size": ["51-200 employees", "1001-5000 employees", "-1"],
            "Industry": ["Technology", "Healthcare", "Finance"],
        }
        df_sample = pl.DataFrame(sample_data)

        # Test cleaning
        df_cleaned = clean_data_polars(df_sample)

        # Check that salary column was added
        self.assertIn("salary", df_cleaned.columns)

        # Check that some valid salaries were extracted
        self.assertTrue(df_cleaned["salary"].is_not_null().any())

    def test_consistency_pandas_polars(self):
        """Test that pandas and polars produce consistent results"""
        # Create test data
        sample_data = {
            "Salary Estimate": ["$50K-$70K", "$80K-$100K", "$60K-$90K"],
            "Rating": [4.2, 3.8, 4.5],
            "Size": ["51-200 employees", "1001-5000 employees", "201-500 employees"],
            "Industry": ["Technology", "Healthcare", "Finance"],
        }

        # Test with pandas
        df_pandas = pd.DataFrame(sample_data)
        df_cleaned_pandas = clean_data(df_pandas)

        # Test with polars
        df_polars = pl.DataFrame(sample_data)
        df_cleaned_polars = clean_data_polars(df_polars)

        # Convert polars to pandas for comparison
        df_cleaned_polars_pd = df_cleaned_polars.to_pandas()

        # Check that both have same number of records
        self.assertEqual(len(df_cleaned_pandas), len(df_cleaned_polars_pd))

        # Check that salary means are approximately equal
        pandas_mean = df_cleaned_pandas["salary"].mean()
        polars_mean = df_cleaned_polars_pd["salary"].mean()
        self.assertAlmostEqual(pandas_mean, polars_mean, places=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
