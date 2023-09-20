lint:
	isort narps_open/pipelines/team_X19V_new.py narps_open/pipelines/team_X19V.py tests/pipelines/test_team_X19V.py
	black narps_open/pipelines/team_X19V_new.py narps_open/pipelines/team_X19V.py tests/pipelines/test_team_X19V.py
	flake8 narps_open/pipelines/team_X19V_new.py narps_open/pipelines/team_X19V.py tests/pipelines/test_team_X19V.py