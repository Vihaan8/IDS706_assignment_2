install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

# lint:
# 	flake8 salary_analysis.py

test:
	python -m pytest -vv --cov=salary_analysis test_salary_analysis.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

run: 
	python salary_analysis.py
	
all: install format test run