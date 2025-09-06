# Mini-Assignment 2 - Data Analyst Salary Analysis

## Project Overview
**Research Question: "What factors influence Data Analyst salaries the most?"**

This repository contains a **Data Analyst Salary Analysis** project for **Data Engineering Systems (IDS 706)** mini assignment. The project analyzes a dataset of Data Analyst job postings to identify the key factors that influence salary levels, demonstrating data science workflows including data cleaning, exploratory analysis, machine learning, and visualization.


The Dataset is publically available on Kaggle - https://www.kaggle.com/datasets/andrewmvd/data-analyst-jobs/data. Thanks to @andrewmvd on Kaggle!


## 📁 Project Files

```
📦 salary-analysis/
├── 📄 salary_analysis.py       # Main analysis script
├── 📄 test_salary_analysis.py  # Unit tests for the analysis
├── 📄 requirements.txt         # Required Python packages
├── 📄 Makefile                # Build automation and commands
├── 📄 README.md               # This documentation
└── 📄 DataAnalyst.csv         # Dataset 
```

## Features
- **Python 3.11** environment setup
- **Data cleaning and preprocessing** with pandas
- **Exploratory Data Analysis** with statistical grouping
- **Machine Learning** with Random Forest for feature importance
- **Data visualization** with matplotlib
- **Unit testing** with pytest
- **Code formatting** with black
- **Code linting** with flake8
- **Makefile** for automated workflow
- **Test coverage** reporting


## Dependencies

The project uses the following Python packages (defined in `requirements.txt`):

- **pandas>=1.3.0** - Data manipulation and analysis
- **numpy>=1.21.0** - Numerical computing
- **matplotlib>=3.5.0** - Data visualization
- **scikit-learn>=1.0.0** - Machine learning algorithms
- **pytest>=6.0.0** - Testing framework
- **pytest-cov>=3.0.0** - Coverage reporting
- **black>=22.0.0** - Code formatting
- **flake8>=4.0.0** - Code linting


## Setup Instructions

### Prerequisites
- **Python 3.7+** installed


### Installation
1. **Clone project repository** 
   ```bash
   git clone https://github.com/Vihaan8/IDS706_assignment_2.git
   ```

3. **Run the analysis:**
   ```bash
   make all
   ```


## Usage

The project uses a Makefile for automated workflow management:

### Available Make Commands
- `make install` - Install and upgrade dependencies
- `make format` - Format code using black
- `make lint` - Lint code using flake8
- `make test` - Run unit tests with coverage
- `make run` - Execute the salary analysis
- `make clean` - Remove cache files
- `make all` - Run complete workflow (install, format, lint, test, run)

### Example Usage
```bash
# Complete analysis workflow
make all

# Individual commands (if needed)
make install    # Install dependencies
make test       # Run tests
make format     # Format code
make lint       # Check code quality
make run        # Execute analysis
```
## Testing

The project includes unit tests that verify:
- Salary extraction from various text formats
- Data cleaning functionality
- Invalid input handling
- Core function behavior

Run tests with: `make test`


## Analysis Workflow

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


## Key Findings

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

To validate  statistical findings and uncover patterns not immediately apparent in the numbers, I created four complementary visualizations:

![Alt Text](https://github.com/Vihaan8/IDS706_assignment_2/blob/main/results/Vis_results_figure_1.png)

**Upper Left - Average Salary by Company Size**: This bar chart reveals that salary differences across company sizes are surprisingly minimal (all within ~$5K range), contradicting the common assumption that larger companies always pay significantly more. The visual confirms our statistical finding that company size has modest predictive power.

**Upper Right - Top Industries by Salary**: This horizontal bar chart clearly illustrates the industry hierarchy, showing Biotech & Pharmaceuticals with a substantial $10K+ premium over average. The visual spacing between industries demonstrates why industry choice appears impactful in individual cases, even though our ML model ranked it as secondary to company rating.

**Lower Left - Salary Distribution**: This histogram confirms our data quality with a normal distribution centered around $70-80K. The shape validates that our salary extraction and filtering processes captured realistic market ranges without artificial clustering or outliers skewing results.

**Lower Right - Salary vs Company Rating**: This scatter plot was crucial for understanding why company rating emerged as the top predictor. While individual points appear scattered, the ML model detected subtle patterns across the 2,252 data points that aren't obvious to the human eye, explaining the apparent contradiction between visual assessment and statistical importance.

### How We Reached Our Conclusions:

1. **Statistical Analysis**: Random Forest algorithm processed all factors simultaneously, revealing company rating as the strongest predictor despite visual scatter
2. **Visual Validation**: Charts confirmed that while industry shows dramatic individual differences, company rating's predictive power operates across all industries and sizes
3. **Data Integration**: The combination of statistical modeling and visual analysis revealed that rating's influence is consistent but subtle, making it more reliable than the visually obvious but variable industry effects

### Implications for Job Seekers:
- **Prioritize company culture**: Rating predicts salary better than obvious factors like company size
- **Industry selection matters**: Visual evidence shows clear $10K+ premiums in top sectors
- **Company size is overrated**: Minimal salary variation across different company sizes
- **Look beyond surface metrics**: The most predictive factors may not be the most visually obvious



## Troubleshooting

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

## Course Information

**Course:** Data Engineering Systems (IDS 706)  
**Assignment:** Mini Project - Data Analysis  
**Institution:** Duke University  
**Focus:** Data science workflow with professional development practices

---

**Author:** Vihaan Manchanda 
**Date:** September 6, 2025
**Repository:** https://github.com/Vihaan8/IDS706_assignment_2.git