# Makefile for Data Analyst Salary Analysis

# Install dependencies (updated to include polars)
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Format code with black
format:
	black *.py

# Lint code with flake8
lint:
	flake8 *.py --ignore=E501,W503

# Run tests with coverage
test:
	python -m pytest -vv --cov=salary_analysis test_salary_analysis.py

# Run the main analysis (includes performance comparison)
run:
	python salary_analysis.py

# Run standalone performance benchmark 
benchmark:
	python pandas_polar_performance/performance_benchmark.py

# Test polars installation
test-polars:
	python -c "import polars as pl; print(f'✅ Polars {pl.__version__} installed successfully')"

# Clean cache files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -f .coverage

# Run complete workflow (includes new polars functionality)
all: install format lint test run

# Run extended analysis with detailed benchmarking
all-extended: install format lint test run benchmark

# Help command
help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies (includes polars)"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with flake8"
	@echo "  make test         - Run unit tests with coverage"
	@echo "  make run          - Run main analysis (includes pandas vs polars)"
	@echo "  make benchmark    - Run detailed performance benchmark"
	@echo "  make test-polars  - Test polars installation"
	@echo "  make clean        - Remove cache files"
	@echo "  make all          - Run complete workflow"
	@echo "  make all-extended - Run workflow + detailed benchmarking"
	@echo "  make help         - Show this help message"

.PHONY: install format lint test run benchmark test-polars clean all all-extended help