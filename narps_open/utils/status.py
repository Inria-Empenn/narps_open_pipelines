#!/usr/bin/python
# coding: utf-8

""" Generate a table with status information about the pipelines """

from json import dumps
import requests

from narps_open.data.description import TeamDescription
from narps_open.pipelines import implemented_pipelines

# Get opened issues
request_url = 'https://api.github.com/repos/Inria-Empenn/narps_open_pipelines/issues'
issues_response = requests.get(request_url, timeout=2)

# Get pipeline files from the repository
request_url = 'https://api.github.com/repositories/501245714/contents/narps_open/pipelines'
pipelines_response = requests.get(request_url, timeout=2)

# List the teams of having a file in the repository
teams_having_pipeline = []
for json in pipelines_response.json():
    pipeline_file = json['name']
    if 'team' in pipeline_file:
        teams_having_pipeline.append(pipeline_file.split('_')[1][0:4])

# Print the header of the table
print('| team_id | status <br>:green_circle: completed <br>:orange_circle: work in progress <br>:red_circle: not started | softwares used | fmriprep used ? | related issues |')
print('| --- |--- | --- | --- | --- |')

# Get data from the team descriptions
for team_id, pipeline_class in implemented_pipelines.items():
    description = TeamDescription(team_id)
    team_softwares = description.categorized_for_analysis['analysis_SW']

    # Get issues related to the team
    issues = []
    for json in issues_response.json():
        issue_title = json['title']
        issue_description = json['body']
        if team_id in issue_title or team_id in issue_description :
            issues.append(f"[{json['number']}]({json['html_url']})")

    # Derive the satus of the pipeline
    if len(issues) == 0 and pipeline_class is not None:
        status = ':green_circle:'
    elif len(issues) != 0:
        status = ':orange_circle:'
    elif team_id in teams_having_pipeline:
        status = ':orange_circle:'
    else:
        status = ':red_circle:'

    print('|', team_id, '|', status, '|', team_softwares, '|', description.preprocessing['used_fmriprep_data'], '|', *issues, '|')
