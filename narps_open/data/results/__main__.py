#!/usr/bin/python
# coding: utf-8

""" Provide a command-line interface for the package narps_open.data.results """

from argparse import ArgumentParser

from narps_open.data.results import ResultsCollectionFactory
from narps_open.pipelines import implemented_pipelines

def main():
    """ Entry-point for the command line tool narps_results """

    # Parse arguments
    parser = ArgumentParser(description='Get Neurovault collection of results from NARPS teams.')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('-t', '--teams', nargs='+', type=str, action='extend',
        help='a list of team IDs', choices=implemented_pipelines.keys())
    group.add_argument('-a', '--all', action='store_true', help='download results from all teams')
    parser.add_argument('-r', '--rectify', action='store_true', default = False, required = False,
        help='rectify the results')
    arguments = parser.parse_args()

    factory = ResultsCollectionFactory()

    if arguments.all:
        for team_id, _ in implemented_pipelines.items():
            collection = factory.get_collection(team_id)
            collection.download()
            if arguments.rectify:
                collection.rectify()
    else:
        for team in arguments.teams:
            collection = factory.get_collection(team)
            collection.download()
            if arguments.rectify:
                collection.rectify()

if __name__ == '__main__':
    main()
