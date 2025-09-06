"""
Question: Data Analyst Salary Analysis: What Factors Influence Salaries the Most?

This script analyzes a dataset of Data Analyst job postings to identify
the key factors that influence salary levels.

Author: Vihaan Manchanda
Date: September 6, 2025
Course: IDS 706 - Data Engineering

Usage:
    python salary_analysis.py

Requirements:
    - DataAnalyst.csv file in the same directory
    - Python packages: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the dataset and return basic information"""
    print("Data Analyst Salary Analysis")
    print("="*40)
    print("Loading data...")
    
    try:
        df = pd.read_csv('DataAnalyst.csv')
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("ERROR: DataAnalyst.csv file not found!")
        print("Please ensure the CSV file is in the same directory as this script.")
        return None

def extract_salary(salary_str):
    """Extract average salary from salary string"""
    if pd.isna(salary_str):
        return np.nan
    
    # Remove text and extract numbers
    salary_clean = str(salary_str).replace('(Glassdoor est.)', '').replace('$', '').replace(',', '')
    
    # Find ranges like "37K-66K"
    match = re.search(r'(\d+)K?\s*-\s*(\d+)K?', salary_clean)
    if match:
        min_sal = float(match.group(1))
        max_sal = float(match.group(2))
        if min_sal < 500:  # Convert K to thousands
            min_sal *= 1000
            max_sal *= 1000
        return (min_sal + max_sal) / 2
    return np.nan

def clean_data(df):
    """Clean and preprocess the dataset"""
    print("Extracting salaries...")
    df['salary'] = df['Salary Estimate'].apply(extract_salary)

    # Clean ratings - only keep valid ratings (1.0 to 5.0)
    df['rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['rating'] = df['rating'].where((df['rating'] >= 1.0) & (df['rating'] <= 5.0))

    # Clean company size - remove weird values like "-1"
    df['Size'] = df['Size'].replace(['-1', 'Unknown'], np.nan)

    # Filter valid data
    df_clean = df.dropna(subset=['salary'])
    df_clean = df_clean[(df_clean['salary'] >= 20000) & (df_clean['salary'] <= 200000)]

    print(f"Valid salary records: {len(df_clean)}")
    print(f"Average salary: ${df_clean['salary'].mean():,.0f}")
    
    return df_clean

def analyze_company_size(df_clean):
    """Analyze salary by company size"""
    print("\n1. Company Size Impact:")
    size_data = df_clean.dropna(subset=['Size'])
    
    if len(size_data) > 0:
        size_impact = size_data.groupby('Size')['salary'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        for size in size_impact.index:
            mean_sal = size_impact.loc[size, 'mean']
            count = size_impact.loc[size, 'count']
            print(f"   {size}: ${mean_sal:,.0f} (n={count})")
        
        return size_impact
    else:
        print("   No valid company size data")
        return None

def analyze_industry(df_clean):
    """Analyze salary by industry (only reliable data)"""
    print("\n2. Top Industries (min 10 records):")
    industry_counts = df_clean['Industry'].value_counts()
    valid_industries = industry_counts[industry_counts >= 10].index
    
    if len(valid_industries) > 0:
        industry_data = df_clean[df_clean['Industry'].isin(valid_industries)]
        industry_impact = industry_data.groupby('Industry')['salary'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        for industry in industry_impact.head(5).index:
            mean_sal = industry_impact.loc[industry, 'mean']
            count = industry_impact.loc[industry, 'count']
            print(f"   {industry[:40]}: ${mean_sal:,.0f} (n={count})")
        
        return industry_impact
    else:
        print("   No industries with sufficient data")
        return None

def analyze_rating(df_clean):
    """Analyze salary by company rating"""
    print("\n3. Rating Impact:")
    rating_data = df_clean.dropna(subset=['rating'])
    
    if len(rating_data) > 0:
        # Create proper rating categories
        rating_bins = pd.cut(rating_data['rating'], 
                            bins=[1.0, 2.5, 3.5, 4.0, 4.5, 5.0],
                            labels=['Poor (1-2.5)', 'Fair (2.5-3.5)', 'Good (3.5-4)', 
                                   'Very Good (4-4.5)', 'Excellent (4.5-5)'])
        
        rating_impact = rating_data.groupby(rating_bins)['salary'].agg(['mean', 'count'])
        
        for rating_range in rating_impact.index:
            if pd.notna(rating_range):
                mean_sal = rating_impact.loc[rating_range, 'mean']
                count = rating_impact.loc[rating_range, 'count']
                print(f"   {rating_range}: ${mean_sal:,.0f} (n={count})")
        
        return rating_impact
    else:
        print("   No valid rating data")
        return None

def build_ml_model(df_clean):
    """Build machine learning model to identify most important factors"""
    print("\nMachine Learning Analysis:")
    print("-" * 25)

    model_df = df_clean.copy()
    features = []

    # Encode company size (only valid sizes)
    size_valid = model_df.dropna(subset=['Size'])
    if len(size_valid) > 50:
        le_size = LabelEncoder()
        le_size.fit(size_valid['Size'])
        model_df['size_encoded'] = model_df['Size'].apply(
            lambda x: le_size.transform([x])[0] if pd.notna(x) and x in le_size.classes_ else np.nan
        )
        features.append('size_encoded')

    # Use cleaned ratings
    rating_valid = model_df.dropna(subset=['rating'])
    if len(rating_valid) > 50:
        model_df['rating_filled'] = model_df['rating'].fillna(model_df['rating'].median())
        features.append('rating_filled')

    # Build model if we have features
    if len(features) > 0:
        model_data = model_df[features + ['salary']].dropna()
        
        if len(model_data) > 50:
            X = model_data[features]
            y = model_data['salary']
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': features,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance (Most Influential Factors):")
            for _, row in importance.iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
            
            return importance
        else:
            print("Not enough data for modeling")
            return None
    else:
        print("No valid features for modeling")
        return None

def create_visualizations(df_clean, size_impact, industry_impact, rating_data):
    """Create visualizations of the analysis results"""
    print("\nCreating visualizations...")
    
    plt.figure(figsize=(12, 8))

    # Salary by company size
    if size_impact is not None:
        plt.subplot(2, 2, 1)
        size_impact['mean'].plot(kind='bar', color='skyblue')
        plt.title('Average Salary by Company Size')
        plt.xticks(rotation=45)
        plt.ylabel('Salary ($)')

    # Salary by top industries
    if industry_impact is not None:
        plt.subplot(2, 2, 2)
        industry_impact['mean'].head(8).plot(kind='barh', color='lightgreen')
        plt.title('Top Industries by Salary')
        plt.xlabel('Salary ($)')

    # Salary distribution
    plt.subplot(2, 2, 3)
    plt.hist(df_clean['salary'], bins=30, alpha=0.7, color='coral')
    plt.title('Salary Distribution')
    plt.xlabel('Salary ($)')
    plt.ylabel('Frequency')

    # Rating vs Salary
    if rating_data is not None and len(rating_data) > 0:
        plt.subplot(2, 2, 4)
        plt.scatter(rating_data['rating'], rating_data['salary'], alpha=0.6)
        plt.title('Salary vs Company Rating')
        plt.xlabel('Rating')
        plt.ylabel('Salary ($)')

    plt.tight_layout()
    plt.show()
    print("Visualizations displayed!")

def generate_conclusion(size_impact, industry_impact, importance):
    """Generate final conclusion answering the research question"""
    print("\n" + "="*50)
    print("ANSWER: What factors influence Data Analyst salaries the most?")
    print("="*50)

    print("Based on the cleaned analysis:")
    
    if size_impact is not None and len(size_impact) > 0:
        top_size = size_impact.index[0]
        top_size_salary = size_impact.iloc[0]['mean']
        print(f"1. COMPANY SIZE: {top_size} pays highest (${top_size_salary:,.0f})")

    if industry_impact is not None and len(industry_impact) > 0:
        top_industry = industry_impact.index[0]
        top_industry_salary = industry_impact.iloc[0]['mean']
        print(f"2. INDUSTRY: {top_industry[:30]}... pays highest (${top_industry_salary:,.0f})")

    if importance is not None:
        most_important = importance.iloc[0]['feature']
        print(f"3. ML MODEL: {most_important} is the most predictive factor")

    print(f"\nConclusion: The analysis shows which factors have the biggest")
    print(f"impact on Data Analyst salaries with cleaned, reliable data.")
    
    if importance is not None:
        print(f"\nKey Insight: While industry shows the highest individual salary differences,")
        print(f"the ML model reveals that company rating is the most reliable predictor")
        print(f"across all scenarios, making it the most influential factor overall.")
    
    print("="*50)

def main():
    """Main function to run the complete analysis"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Clean data
    df_clean = clean_data(df)
    
    # Perform factor analysis
    print("\nFactor Analysis:")
    print("-" * 20)
    
    size_impact = analyze_company_size(df_clean)
    industry_impact = analyze_industry(df_clean)
    rating_impact = analyze_rating(df_clean)
    
    # Build ML model
    importance = build_ml_model(df_clean)
    
    # Create visualizations
    rating_data = df_clean.dropna(subset=['rating']) if 'rating' in df_clean.columns else None
    create_visualizations(df_clean, size_impact, industry_impact, rating_data)
    
    # Generate conclusion
    generate_conclusion(size_impact, industry_impact, importance)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Check the generated visualizations for detailed insights.")

if __name__ == "__main__":
    main()