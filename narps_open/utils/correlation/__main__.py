#!/usr/bin/python
# coding: utf-8

""" A command line tool for the narps_open.utils.correlation module """

from os.path import join
from argparse import ArgumentParser

from narps_open.data.results import ResultsCollection
from narps_open.utils.configuration import Configuration
from narps_open.utils.correlation import get_correlation_coefficient
from narps_open.pipelines import get_implemented_pipelines
from narps_open.runner import PipelineRunner

def main():
    """ Entry-point for the command line tool narps_open_correlations """

    # Parse arguments
    parser = ArgumentParser(description = 'Compare reproduced files to original results.')
    parser.add_argument('-t', '--team', type = str, required = True,
        help = 'the team ID', choices = get_implemented_pipelines())
    parser.add_argument('-n', '--nsubjects', type = int, required = True,
        help='the number of subjects to be selected')
    parser.add_argument('--config', type=str, required=False,
        help='custom configuration file to be used')
    arguments = parser.parse_args()

    # Init configuration
    if arguments.config:
        Configuration('custom').config_file = arguments.config

    # Initialize pipeline
    runner = PipelineRunner(arguments.team)
    runner.pipeline.directories.dataset_dir = Configuration()['directories']['dataset']
    runner.pipeline.directories.results_dir = Configuration()['directories']['reproduced_results']
    runner.pipeline.directories.set_output_dir_with_team_id(arguments.team)
    runner.pipeline.directories.set_working_dir_with_team_id(arguments.team)
    runner.nb_subjects = arguments.nsubjects

    # Indices and keys to the unthresholded maps
    indices = list(range(1, 18, 2))

    # Retrieve the paths to the reproduced files
    reproduced_files = runner.pipeline.get_hypotheses_outputs()
    reproduced_files = [reproduced_files[i] for i in indices]

    # Retrieve the paths to the results files
    collection = ResultsCollection(arguments.team)
    file_keys = [f'hypo{h}_unthresh.nii.gz' for h in range(1,10)]
    results_files = [join(collection.directory, k) for k in file_keys]

    # Compute the correlation coefficients
    print([
        get_correlation_coefficient(reproduced_file, results_file)
        for reproduced_file, results_file in zip(reproduced_files, results_files)
        ])

if __name__ == '__main__':
    main()
