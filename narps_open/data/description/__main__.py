#!/usr/bin/python
# coding: utf-8

""" Provide a command-line interface for the package narps_open.data.description """

from argparse import ArgumentParser
from json import dumps

from narps_open.data.description import TeamDescription
from narps_open.pipelines import implemented_pipelines

def main():
    """ Entry-point for the command line tool narps_description """

    # Parse arguments
    parser = ArgumentParser(description='Get description of a NARPS pipeline.')
    parser.add_argument('-t', '--team', type=str, required=True,
        help='the team ID', choices=implemented_pipelines.keys())
    parser.add_argument('-d', '--dictionary', type=str, required=False,
        choices=[
            'general',
            'exclusions',
            'preprocessing',
            'analysis',
            'categorized_for_analysis',
            'derived',
            'comments'
            ],
        help='the sub dictionary of team description')
    formats = parser.add_mutually_exclusive_group(required = False)
    formats.add_argument('--json', action='store_true', help='output team description as JSON')
    formats.add_argument('--md', action='store_true', help='output team description as Markdown')
    arguments = parser.parse_args()

    # Initialize a TeamDescription
    information = TeamDescription(team_id = arguments.team)

    # Output description
    if arguments.md and arguments.dictionary is not None:
        print('Sub dictionaries cannot be exported as Markdown yet.')
        print('Print the whole description instead.')
    elif arguments.md:
        print(information.markdown())
    else:
        if arguments.dictionary == 'general':
            print(dumps(information.general, indent = 4))
        elif arguments.dictionary == 'exclusions':
            print(dumps(information.exclusions, indent = 4))
        elif arguments.dictionary == 'preprocessing':
            print(dumps(information.preprocessing, indent = 4))
        elif arguments.dictionary == 'analysis':
            print(dumps(information.analysis, indent = 4))
        elif arguments.dictionary == 'categorized_for_analysis':
            print(dumps(information.categorized_for_analysis, indent = 4))
        elif arguments.dictionary == 'derived':
            print(dumps(information.derived, indent = 4))
        elif arguments.dictionary == 'comments':
            print(dumps(information.comments, indent = 4))
        else:
            print(dumps(information, indent = 4))

if __name__ == '__main__':
    main()
