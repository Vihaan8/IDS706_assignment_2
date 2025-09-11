# Makefile for Data Analyst Salary Analysis Project
# Author: Vihaan Manchanda
# Date: 2024-06-15
# Updated: 2024-06-15
# Version 3.0 -> Added more tests and made the Makefile more comprehensive.


.PHONY: all install test test-basic test-complete test-coverage test-all clean run benchmark help

# Default target
all: install test run

# Install dependencies
install:
	@echo "📦 Installing required packages..."
	pip install -r requirements.txt
	pip install pytest pytest-cov

# Run basic unit tests
test-basic:
	@echo "🧪 Running basic unit tests..."
	python -m pytest test_salary_analysis.py -v

# Run comprehensive test suite
test-complete:
	@echo "🧪 Running comprehensive test suite..."
	python -m pytest test_salary_analysis_complete.py -v

# Run all tests with coverage
test-coverage:
	@echo "📊 Running all tests with coverage..."
	python -m pytest test_salary_analysis.py \
		--cov=salary_analysis \
		--cov-report=term-missing \
		--cov-report=html

# Run all tests 
test: test-coverage

# Run all tests using the test runner
test-all:
	@echo "🚀 Running complete test suite with runner..."
	python run_tests.py

tests:
	@echo "🧪 Running all tests (verbose, with coverage)…"
	python -m pytest \
		-vv -rA -s --maxfail=1 --durations=10 \
		--cov=salary_analysis --cov-report=term-missing --cov-report=html \
		test_salary_analysis.py
	@echo "📈 Coverage HTML report: htmlcov/index.html"

# Run the main analysis
run:
	@echo "📊 Running salary analysis..."
	python salary_analysis.py

# Run performance benchmark
benchmark:
	@echo "⚡ Running performance benchmark..."
	python performance_benchmark.py

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.pyc
	rm -rf */__pycache__
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Run specific test categories
test-unit:
	@echo "🧪 Running unit tests only..."
	python -m pytest test_salary_analysis_complete.py::TestSalaryExtraction -v
	python -m pytest test_salary_analysis_complete.py::TestDataCleaning -v

test-integration:
	@echo "🧪 Running integration tests only..."
	python -m pytest test_salary_analysis_complete.py::TestSystemIntegration -v

test-ml:
	@echo "🧪 Running ML tests only..."
	python -m pytest test_salary_analysis_complete.py::TestMachineLearning -v

test-edge:
	@echo "🧪 Running edge case tests only..."
	python -m pytest test_salary_analysis_complete.py::TestEdgeCases -v

# Quick test (fast subset of tests)
test-quick:
	@echo "⚡ Running quick test subset..."
	python -m pytest test_salary_analysis.py -v -k "not performance"

# Continuous Integration target
ci: install test-coverage
	@echo "✅ CI pipeline complete"

# Development workflow
dev: install test-quick run
	@echo "✅ Development cycle complete"

# Full validation
validate: install test-all benchmark
	@echo "✅ Full validation complete"

# Help command
help:
	@echo "📚 Available commands:"
	@echo "  make all          - Install, test, and run analysis"
	@echo "  make install      - Install required packages"
	@echo "  make tests        - Run all tests with verbose output and coverage"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make test-basic   - Run unit tests only"
	@echo "  make test-complete - Run all tests"
	@echo "  make test-coverage - Run tests with coverage report"
	@echo "  make test-all     - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-system  - Run system tests only"
	@echo "  make test-performance - Run performance tests only"
	@echo "  make test-ml      - Run machine learning tests only"
	@echo "  make test-edge    - Run edge case tests only"
	@echo "  make test-quick   - Run quick test subset (no performance)"
	@echo "  make run          - Run the main analysis"
	@echo "  make benchmark    - Run performance benchmark"
	@echo "  make clean        - Clean up generated files"
	@echo "  make ci           - Run CI pipeline"
	@echo "  make dev          - Run development workflow"
	@echo "  make validate     - Run full validation"
	@echo "  make help         - Show this help message"