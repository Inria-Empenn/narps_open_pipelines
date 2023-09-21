lint:
	isort narps_open/pipelines/*R9K3*.py tests/pipelines/*R9K3.py
	black examples/notebooks/*R9K3*.ipynb narps_open/pipelines/*R9K3*.py tests/pipelines/*R9K3.py
	flake8 narps_open/pipelines/*R9K3*.py tests/pipelines/*R9K3.py