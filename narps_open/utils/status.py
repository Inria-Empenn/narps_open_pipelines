#!/usr/bin/python
# coding: utf-8

""" Generate a table with status information about the pipelines """

from os.path import join, basename
from json import dumps
from argparse import ArgumentParser
from glob import glob

from requests import get
from importlib_resources import files

from narps_open.utils.description import TeamDescription
from narps_open.pipelines import implemented_pipelines

def get_opened_issues():
    """ Return a list of opened issues for the NARPS Open Pipelines project """
    request_url = 'https://api.github.com/repos/Inria-Empenn/narps_open_pipelines/issues'
    response = get(request_url, timeout = 2)
    response.raise_for_status()

    return response.json()

def get_teams_with_pipeline_files():
    """ Return a set of teams having a file for their pipeline in the repository """
    teams_having_pipeline = set()

    # List all files in the narps_open.pipelines package
    pipeline_files = glob(join(files('narps_open.pipelines'), 'team*.py'))

    # Identify teams from pipelines
    for file in pipeline_files:
        teams_having_pipeline.add(basename(file)[5:9])

    return teams_having_pipeline

class PipelineStatusReport():
    """ An object that  """

    def __init__(self):
        self.contents = {}

    def __str__(self):
        return dumps(self.contents, indent = 4)

    def generate(self):
        """ Generate the report by adding information into self.contents dictionary """

        opened_issues = get_opened_issues()
        teams_having_pipeline = get_teams_with_pipeline_files()

        # Loop through teams
        for team_id, pipeline_class in implemented_pipelines.items():

            self.contents[team_id] = {}

            # Get software used in the pipeline, from the team description
            description = TeamDescription(team_id)
            self.contents[team_id]['softwares'] = \
                description.categorized_for_analysis['analysis_SW']
            self.contents[team_id]['fmriprep'] = description.preprocessing['used_fmriprep_data']

            # Get issues related to the team
            issues = {}
            for issue in opened_issues:
                if team_id in issue['title'] or team_id in issue['body']:
                    issues[issue['number']] = issue['html_url']
            self.contents[team_id]['issues'] = issues

            # Derive the status of the pipeline
            has_issues = len(issues) > 0
            is_implemeted = pipeline_class is not None
            has_file = team_id in teams_having_pipeline

            if is_implemeted and not has_file:
                raise AttributeError(f'Pipeline {team_id} referred as implemented with no file')
            elif not is_implemeted and not has_issues and not has_file:
                self.contents[team_id]['status'] = 'idle'
            elif is_implemeted and has_file and not has_issues:
                self.contents[team_id]['status'] = 'done'
            else:
                self.contents[team_id]['status'] = 'progress'

    def markdown(self):
        """ Return a string representing the report as markdown format """

        # Create header
        output_markdown = '# Work progress status for each pipeline\n'
        output_markdown += 'The *status* column tells whether the work on the pipeline is :\n'
        output_markdown += '<br>:red_circle: not started yet\n'
        output_markdown += '<br>:orange_circle: in progress\n'
        output_markdown += '<br>:green_circle: completed\n'
        output_markdown += '<br>The *softwares* column gives a simplified version of what can be '
        output_markdown += 'found in the [team descriptions](/docs/description.md) under the '
        output_markdown += '`general.software` column.\n'

        # Start table
        output_markdown += '| team_id | status | softwares used | fmriprep used ? |'
        output_markdown += ' related issues |\n'
        output_markdown += '| --- |:---:| --- | --- | --- |\n'

        # Add table contents
        for team_key, team_values in self.contents.items():
            output_markdown += f'| {team_key} '

            status = ''
            if team_values['status'] == 'done':
                status = ':green_circle:'
            elif team_values['status'] == 'progress':
                status = ':orange_circle:'
            else:
                status = ':red_circle:'

            output_markdown += f'| {status} '
            output_markdown += f'| {team_values["softwares"]} '
            output_markdown += f'| {team_values["fmriprep"]} '

            issues = ''
            for issue_number, issue_url in team_values['issues'].items():
                issues += f'[{issue_number}]({issue_url}), '

            output_markdown += f'| {issues} |\n'

        return output_markdown

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description='Get a work progress status report for pipelines.')
    formats = parser.add_mutually_exclusive_group(required = False)
    formats.add_argument('--json', action='store_true', help='output the report as JSON')
    formats.add_argument('--md', action='store_true', help='output the report as Markdown')
    arguments = parser.parse_args()

    report = PipelineStatusReport()
    report.generate()

    if arguments.md:
        print(report.markdown())
    else:
        print(report)
