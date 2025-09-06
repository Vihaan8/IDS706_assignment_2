#!/usr/bin/env python3
"""
Simple unit tests for salary_analysis.py

Usage:
    python -m pytest -vv --cov=salary_analysis test_salary_analysis.py
"""

import unittest
import pandas as pd
import numpy as np
from salary_analysis import extract_salary, clean_data


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

    def test_extract_salary_invalid_inputs(self):
        """Test salary extraction with invalid inputs"""
        # Test NaN input
        result1 = extract_salary(np.nan)
        self.assertTrue(pd.isna(result1))

        # Test invalid string
        result2 = extract_salary("No salary information")
        self.assertTrue(pd.isna(result2))

    def test_clean_data_basic(self):
        """Test basic data cleaning functionality"""
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
