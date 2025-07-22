#!/usr/bin/python
# coding: utf-8

""" This module allows to compare pipeline output with original team results """

import sys
from argparse import ArgumentParser

import pytest

from narps_open.pipelines import get_implemented_pipelines

def main():
    """ Entry-point for the command line tool narps_open_tester """

    # Parse arguments
    parser = ArgumentParser(description='Test the pipelines from NARPS.')
    parser.add_argument('-t', '--team', type=str, required=True,
        help='the team ID', choices=get_implemented_pipelines())
    parser.add_argument('--config', type=str, required=False,
        help='custom configuration file to be used')
    arguments = parser.parse_args()

    pytest_arguments = [
        '-s',
        '-q',
        '-x',
        f'tests/pipelines/test_team_{arguments.team}.py',
        '-m',
        'pipeline_test']

    if 'config' in arguments:
        pytest_arguments.append(f'--narps_open_config={arguments.config}')

    sys.exit(pytest.main(pytest_arguments))

if __name__ == '__main__':
    main()
