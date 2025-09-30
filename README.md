<div align="center">

# Data Analyst Salary Analysis

**Identifying the Key Factors That Drive Data Analyst Compensation**

[![CI](https://github.com/Vihaan8/IDS706_assignment_2/actions/workflows/ci.yaml/badge.svg)](https://github.com/Vihaan8/IDS706_assignment_2/actions/workflows/ci.yaml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-Passing-success)

**Course:** Data Engineering Systems (IDS 706) | **Institution:** Duke University


</div>

## Table of Contents

- [Project Overview](#-project-overview)
- [Project Files](#-project-files)
- [Features](#-features)
- [CI/CD Pipeline](#cicd-pipeline)
- [Code Refactoring](#-code-refactoring)
- [Dependencies](#-dependencies)
- [Usage](#-usage)
- [Setup & Testing](#-setup--testing)
- [Testing](#-testing)
- [Analysis Workflow](#-analysis-workflow)
- [Performance Analysis](#-performance-analysis)
- [Key Findings](#-key-findings)
- [Troubleshooting](#-troubleshooting)  


## 📊 Project Overview
**Research Question: "What factors influence Data Analyst salaries the most?"**

This repository contains a **Data Analyst Salary Analysis** project for **Data Engineering Systems (IDS 706)** mini assignment. The project analyzes a dataset of Data Analyst job postings to identify the key factors that influence salary levels, demonstrating data science workflows including data cleaning, exploratory analysis, machine learning, and visualization.

The Dataset is publically available on Kaggle - https://www.kaggle.com/datasets/andrewmvd/data-analyst-jobs/data. Thanks to @andrewmvd on Kaggle!

---

## 🏆 Key Findings

> **TL;DR:** Company rating is the most influential factor for Data Analyst salaries, even more than company size or industry—a surprising result that challenges conventional wisdom.

### 📈 Quick Results Summary

| Metric | Value | Insight |
|--------|-------|---------|
| **Dataset Size** | 2,252 jobs | After cleaning from 2,253 raw records |
| **Average Salary** | $72,123 | Median: $70,000 |
| **Salary Range** | $20K - $200K | Filtered for realistic ranges |
| **Top Predictor** | Company Rating | 0.579 importance (ML model) |
| **Highest Paying Industry** | Biotech & Pharma | $83,106 avg (+15% vs overall) |
| **Optimal Company Size** | 5K-10K employees | $74,201 avg |
| **Performance Gain** | Polars 2.3x faster | For data processing operations |

### 🎓 Implications for Job Seekers

1. **Prioritize company culture** — Rating predicts salary better than obvious factors like company size
2. **Industry selection matters** — Top sectors command $10K+ premiums
3. **Company size is overrated** — Minimal salary variation across different company sizes ($5K range)
4. **Look beyond surface metrics** — The most predictive factors aren't always the most visually obvious

---

## 📁 Project Files

```
salary-analysis/
├── DataAnalyst.csv                
├── Dockerfile                   
├── Makefile                       
├── README.md                        
├── requirements.txt                  
├── salary_analysis.py               
├── test_salary_analysis.py           
├── pandas_polar_performance/   
│   ├── performance_benchmark.py
│   └── POLARS_PERFORMANCE_ANALYSIS.md
└── .devcontainer/
    └── devcontainer.json    
```

## ✨ Features
- **Python 3.11** environment setup
- **Data cleaning and preprocessing** with pandas
- **Exploratory Data Analysis** with statistical grouping
- **Machine Learning** with Random Forest for feature importance
- **Data visualization** with matplotlib
- **Unit testing** with pytest
- **Code formatting** with black
- **Code linting** with flake8
- **Makefile** for automated workflow
- **Comprehensive test suite** (unit, integration, system, and performance tests with coverage)
- **Reproducible Docker image** for consistent execution
- **VS Code Dev Container** for full development environment setup

## 🔧 Code Refactoring

The codebase has been refactored to improve readability, maintainability, and follow best practices:

### Rename Variables for Clarity
Using VS Code's F2 rename feature, variables were renamed throughout the codebase for better semantic meaning:

![Variable Renaming 1](https://github.com/Vihaan8/IDS706_assignment_2/blob/main/results/refactor1.png)

![Variable Renaming 2](https://github.com/Vihaan8/IDS706_assignment_2/blob/main/results/refactor2.png)

- `df` → `raw_dataframe` / `cleaned_salary_data` for clearer data pipeline understanding
- `extract_salary()` → `parse_salary_range()` for more accurate function naming
- All parameter and variable names updated consistently across the codebase

### Extract Methods for Modularity
Complex code blocks were extracted into separate, reusable functions using VS Code's extract method feature:

![Extract Method 1](https://github.com/Vihaan8/IDS706_assignment_2/blob/main/results/refactor3.png)

![Extract Method 2](https://github.com/Vihaan8/IDS706_assignment_2/blob/main/results/refactor4.png)

- `filter_valid_salary_range()` - Extracted salary filtering logic for reusability
- `encode_company_size()` - Separated feature engineering from model building
- Improved code organization and testability

## 📦 Dependencies

The project uses the following Python packages (defined in `requirements.txt`):

- **pandas>=1.3.0** - Data manipulation and analysis
- **numpy>=1.21.0** - Numerical computing
- **matplotlib>=3.5.0** - Data visualization
- **scikit-learn>=1.0.0** - Machine learning algorithms
- **pytest>=6.0.0** - Testing framework
- **pytest-cov>=3.0.0** - Coverage reporting
- **black==25.1.0** - Code formatting
- **flake8>=4.0.0** - Code linting

## 🚀 Usage

The project uses a Makefile for automated workflow management:

### Available Make Commands
- `make install` - Install and upgrade dependencies
- `make format` - Format code using black
- `make lint` - Lint code using flake8
- `make tests` - Run all tests with coverage
- `make run` - Execute the salary analysis
- `make clean` - Remove cache files
- `make all` - Run complete workflow (install, format, lint, test, run)

### Example Usage
```bash
# Complete analysis workflow
make all

# Individual commands (if needed)
make install    # Install dependencies
make tests       # Run tests
make format     # Format code
make lint       # Check code quality
make run        # Execute analysis
```

## ⚙️ Setup & Testing

### Local (optional)
Run locally with Python 3.11:
```bash
make install
make tests
```

### Docker (recommended)
Build the Docker image:
```bash
docker build -t salary-analysis:dev .
```
Run tests inside the container (with coverage report in `htmlcov/`):
```bash
docker run --rm -it -v "$PWD":/app -w /app salary-analysis:dev   bash -lc "make tests"
```

### VS Code Dev Container 
The Dev Container extends the Docker setup by letting VS Code open directly inside the container - giving a consistent Python environment, dependencies, and tools without needing to install them locally.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)  
2. Install [VS Code](https://code.visualstudio.com/)  
3. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)  
4. Open this repo in VS Code.  
5. Press **View → Command Palette…** → search for **"Dev Containers: Reopen in Container"**.  
6. Once inside, open a terminal and run:
   ```bash
   make tests
   ```

## 🧪 Testing

This project includes comprehensive tests:

- **Unit tests**: verify core functions (loading, filtering, salary extraction, ML).  
- **Integration tests**: validate the full pipeline from loading → cleaning → analyzing.  
- **System tests**: edge cases and end-to-end workflows with multiple datasets.  
- **Performance tests**: compare execution speed between pandas and polars.

Run all tests (with verbose output, timing, and coverage report):
```bash
make tests
```
Results:
- Coverage summary in the terminal.
- Full HTML coverage report in `htmlcov/index.html`.

Here's an example run of the full test suite with coverage:

![Testing Results](https://github.com/Vihaan8/IDS706_assignment_2/blob/main/results/tests_results_combined.png)

## 🔄 Analysis Workflow

The analysis follows a structured data science pipeline:

1. **Data Loading & Cleaning**
   - Extracts salary ranges from text format
   - Cleans company ratings and sizes
   - Filters realistic salary ranges ($20K-$200K)

2. **Exploratory Data Analysis**
   - Analyzes salary by company size
   - Examines industry impact on compensation
   - Studies company rating effects

3. **Machine Learning**
   - Uses Random Forest Regressor
   - Identifies most predictive factors
   - Ranks feature importance

4. **Visualization**
   - Creates salary distribution plots
   - Shows factor comparisons
   - Displays correlation analysis

5. **Results & Conclusions**
   - Answers the research question
   - Provides data-driven insights

## ⚡ Performance Analysis
This project includes a comprehensive performance comparison between pandas and Polars for data processing operations. The analysis benchmarks data loading, cleaning, and grouping operations to evaluate the efficiency of modern data tools. For detailed performance results, benchmarking methodology, and insights, see the complete analysis in pandas_polar_performance/POLARS_PERFORMANCE_ANALYSIS.md.

## 🔍 Key Findings

The analysis reveals that **company rating** is the most influential factor for Data Analyst salaries, even more than company size or industry. This finding emerges from both statistical analysis and visual examination of the data patterns.

### Primary Statistical Results:
- **Average Data Analyst Salary**: $72,123 across 2,252 job postings
- **Most Important Factor**: Company rating (0.579 importance score from Random Forest)
- **Secondary Factor**: Company size (0.421 importance score)

### Detailed Factor Analysis:
- **Company Size**: Mid-large companies (5,001-10,000 employees) pay highest ($74,201)
- **Industry**: Biotech & Pharmaceuticals leads with $83,106 average salary
- **Company Rating**: Surprisingly, "Poor" rated companies pay highest ($75,035), indicating complex market dynamics

### Visualization Insights:

To validate my statistical findings and uncover patterns not immediately apparent in the numbers, I created four complementary visualizations:

![Alt Text](https://github.com/Vihaan8/IDS706_assignment_2/blob/main/results/Vis_results_figure_1.png)

**Upper Left - Average Salary by Company Size**: This bar chart reveals that salary differences across company sizes are surprisingly minimal (all within ~$5K range), contradicting the common assumption that larger companies always pay significantly more. The visual confirms my statistical finding that company size has modest predictive power.

**Upper Right - Top Industries by Salary**: This horizontal bar chart clearly illustrates the industry hierarchy, showing Biotech & Pharmaceuticals with a substantial $10K+ premium over average. The visual spacing between industries demonstrates why industry choice appears impactful in individual cases, even though my ML model ranked it as secondary to company rating.

**Lower Left - Salary Distribution**: This histogram confirms my data quality with a normal distribution centered around $70-80K. The shape validates that my salary extraction and filtering processes captured realistic market ranges without artificial clustering or outliers skewing results.

**Lower Right - Salary vs Company Rating**: This scatter plot was crucial for understanding why company rating emerged as the top predictor. While individual points appear scattered, the ML model detected subtle patterns across the 2,252 data points that aren't obvious to the human eye, explaining the apparent contradiction between visual assessment and statistical importance.

### How I Reached My Conclusions:

1. **Statistical Analysis**: Random Forest algorithm processed all factors simultaneously, revealing company rating as the strongest predictor despite visual scatter
2. **Visual Validation**: Charts confirmed that while industry shows dramatic individual differences, company rating's predictive power operates across all industries and sizes
3. **Data Integration**: The combination of statistical modeling and visual analysis revealed that rating's influence is consistent but subtle, making it more reliable than the visually obvious but variable industry effects

### Implications for Job Seekers:
- **Prioritize company culture**: Rating predicts salary better than obvious factors like company size
- **Industry selection matters**: Visual evidence shows clear $10K+ premiums in top sectors
- **Company size is overrated**: Minimal salary variation across different company sizes
- **Look beyond surface metrics**: The most predictive factors may not be the most visually obvious

## 🛠️ Troubleshooting

**Dataset not found:**
```
ERROR: DataAnalyst.csv file not found!
```
**Solution:** Ensure `DataAnalyst.csv` is in the project directory

**Make command issues:**
```
make: command not found
```
**Solution:** Use Git Bash (Windows), or install make via package manager

**Dependencies missing:**
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution:** Run `make all` to install all dependencies

---

**Author:** Vihaan Manchanda  
**Date:** September 28, 2025  
**Repository:** https://github.com/Vihaan8/IDS706_assignment_2.git