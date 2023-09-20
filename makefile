lint:
	isort narps_open/pipelines/*X19V*.py tests/pipelines/*X19V.py tests/conftest.py
	black examples/notebooks/*X19V*.ipynb narps_open/pipelines/*X19V*.py tests/pipelines/*X19V.py tests/conftest.py
	flake8 narps_open/pipelines/*X19V*.py tests/pipelines/*X19V.py tests/conftest.py