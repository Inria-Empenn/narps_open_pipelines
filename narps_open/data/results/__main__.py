#!/usr/bin/python
# coding: utf-8

""" Provide a command-line interface for the package narps_open.data.results """

from argparse import ArgumentParser

from narps_open.data.results import ResultsCollection
from narps_open.pipelines import implemented_pipelines

# Parse arguments
parser = ArgumentParser(description='Get Neurovault collection of results from NARPS teams.')
group = parser.add_mutually_exclusive_group(required = True)
group.add_argument('-t', '--teams', nargs='+', type=str, action='extend',
    help='a list of team IDs')
group.add_argument('-a', '--all', action='store_true', help='download results from all teams')
arguments = parser.parse_args()

if arguments.all:
    for team_id, _ in implemented_pipelines.items():
        ResultsCollection(team_id).download()
else:
    for team in arguments.teams:
        ResultsCollection(team).download()
