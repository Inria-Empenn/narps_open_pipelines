#!/usr/bin/python
# coding: utf-8

"""
conftest.py file will be automatically launched before running
pytest on (a) test file(s) in the same directory.
"""

from os.path import join, exists

from pytest import helpers

from narps_open.runner import PipelineRunner
from narps_open.utils.correlation import get_correlation_coefficient

@helpers.register
def test_pipeline(
    team_id: str,
    references_dir: str,
    dataset_dir: str,
    results_dir: str,
    nb_subjects: int = 4
    ):
    """ This pytest helper allows to launch a pipeline over a given number of subjects

    Arguments:
        - team_id: str, the ID of the team (allows to identify which pipeline to run)
        - references_dir: str, the path to the directory where results from the teams are
        - dataset_dir: str, the path to the ds001734 dataset
        - results_dir: str, the path where to store the results
        - nb_subjects: int, the number of subject to run the pipeline with

    Returns:
        - list(float) the correlation coefficients between the following
        (reference and computed) files:

    This function can be used as follows:
        results = pytest.helpers.test_pipeline('2T6S', '/references/', '/data/', '/output/', 4)
        assert statistics.mean(results) > .003

    TODO : how to keep intermediate files of the low level for the next numbers of subjects ?
        - keep intermediate levels : boolean in PipelineRunner
    """

    # Initialize the pipeline
    """
    runner = PipelineRunner(team_id)
    runner.random_nb_subjects = nb_subjects
    runner.pipeline.directories.dataset_dir = dataset_dir
    runner.pipeline.directories.results_dir = results_dir
    runner.pipeline.directories.set_output_dir_with_team_id(team_id)
    runner.pipeline.directories.set_working_dir_with_team_id(team_id)
    runner.start()
    """

    # Retrieve the paths to the computed files
    output_files = [
        join(
            #runner.pipeline.directories.output_dir,
            '/home/bclenet/output',
            'NARPS-reproduction',
            f'team_2T6S_nsub_{nb_subjects}_hypo{hypothesis}_unthresholded.nii')
            #f'team-2T6S_nsub-{nb_subjects}_hypo-{hypothesis}_unthresholded.nii')
        for hypothesis in range(1, 10)
        ]

    # Retrieve the paths to the reference files
    reference_files = [
        join(
            references_dir,
            f'NARPS-{team_id}',
            f'hypo{hypothesis}_unthresholded.nii.gz')
        for hypothesis in range(1, 10)
        ]

    # Add 'revised' to files when needed
    for index, reference_file in enumerate(reference_files):
        if not exists(reference_file):
            reference_files[index] = reference_file.replace('.nii.gz', '_revised.nii.gz')
        if not exists(reference_files[index]):
            raise FileNotFoundError(reference_files[index] + ' does not exist.')

    # Example paths for reference data 2T6S
    #    'https://neurovault.org/collections/4881/NARPS-2T6S/hypo1_unthresh'

    # Compute the correlation coefficients
    return [
        get_correlation_coefficient(output_file, reference_file)
        for output_file, reference_file in zip(output_files, reference_files)
        ]
