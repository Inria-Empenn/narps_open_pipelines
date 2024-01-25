#!/usr/bin/python
# coding: utf-8

""" Provide a command-line interface for the package narps_open.pipelines """

from argparse import ArgumentParser

from narps_open.pipelines import get_implemented_pipelines

def main():
    """ Entry-point for the command line tool narps_open_pipeline """

    # Parse arguments
    parser = ArgumentParser(description='Get description of a NARPS pipeline.')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='verbose mode')
    arguments = parser.parse_args()

    # Print header
    print('NARPS Open Pipelines')

    # Print general information about NARS Open Pipelines
    print('A codebase reproducing the 70 pipelines of the NARPS study (Botvinik-Nezer et al., 2020) shared as an open resource for the community.')

    # Print pipelines
    implemented_pipelines = get_implemented_pipelines()
    print(f'There are currently {len(implemented_pipelines)} implemented pipelines: {implemented_pipelines}')

if __name__ == '__main__':
    main()
