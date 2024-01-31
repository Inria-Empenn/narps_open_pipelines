#!/usr/bin/python
# coding: utf-8

"""
conftest.py file will be automatically launched before running
pytest on (a) test file(s) in the same directory.
"""

from os import remove
from os.path import join, isfile
from shutil import rmtree

from pytest import helpers
from pathvalidate import is_valid_filepath

from narps_open.pipelines import Pipeline
from narps_open.runner import PipelineRunner
from narps_open.utils import get_subject_id
from narps_open.utils.correlation import get_correlation_coefficient
from narps_open.utils.configuration import Configuration
from narps_open.data.results import ResultsCollection

# Init configuration, to ensure it is in testing mode
Configuration(config_type='testing')

@helpers.register
def test_pipeline_outputs(pipeline: Pipeline, number_of_outputs: list):
    """ Test the outputs of a Pipeline.
        Arguments:
        - pipeline, Pipeline: the pipeline to test
        - number_of_outputs, list: a list containing the expected number of outputs for each
          stage of the pipeline (preprocessing, run_level, subject_level, group_level, hypotheses)

        Return: True if the outputs are in sufficient number and each ones name is valid,
          False otherwise.
    """
    assert len(number_of_outputs) == 5
    for outputs, number in zip([
        pipeline.get_preprocessing_outputs(),
        pipeline.get_run_level_outputs(),
        pipeline.get_subject_level_outputs(),
        pipeline.get_group_level_outputs(),
        pipeline.get_hypotheses_outputs()], number_of_outputs):

        assert len(outputs) == number
        for output in outputs:
            assert is_valid_filepath(output, platform = 'auto')
            assert not any(c in output for c in ['{', '}'])

@helpers.register
def test_pipeline_execution(
    team_id: str,
    nb_subjects: int = 4
    ):
    """ This pytest helper allows to launch a pipeline over a given number of subjects

    Arguments:
        - team_id: str, the ID of the team (allows to identify which pipeline to run)
        - nb_subjects: int, the number of subject to run the pipeline with

    Returns:
        - list(float) the correlation coefficients between the following
        (result and reproduced) files:

    This function can be used as follows:
        results = pytest.helpers.test_pipeline('2T6S', 4)
        assert statistics.mean(results) > .003

    TODO : how to keep intermediate files of the low level for the next numbers of subjects ?
        - keep intermediate levels : boolean in PipelineRunner
    """
    # A list of number of subject to iterate over
    nb_subjects_list = list(range(
        Configuration()['testing']['pipelines']['nb_subjects_per_group'],
        nb_subjects,
        Configuration()['testing']['pipelines']['nb_subjects_per_group'])
        )
    nb_subjects_list.append(nb_subjects)

    # Initialize the pipeline
    runner = PipelineRunner(team_id)
    runner.pipeline.directories.dataset_dir = Configuration()['directories']['dataset']
    runner.pipeline.directories.results_dir = Configuration()['directories']['reproduced_results']
    runner.pipeline.directories.set_output_dir_with_team_id(team_id)
    runner.pipeline.directories.set_working_dir_with_team_id(team_id)

    # Run first level by (small) sub-groups of subjects
    for subjects in nb_subjects_list:
        runner.nb_subjects = subjects

        # Run as long as there are missing files after first level (with a max number of trials)
        # TODO : this is a workaround
        for _ in range(Configuration()['runner']['nb_trials']):

            # Get missing subjects
            missing_subjects = set()
            for file in runner.get_missing_first_level_outputs():
                subject_id = get_subject_id(file)
                if subject_id is not None:
                    missing_subjects.add(subject_id)

            # Leave if no missing subjects
            if not missing_subjects:
                break

            # Start pipeline
            runner.subjects = missing_subjects
            try: # This avoids errors in the workflow to make the test fail
                runner.start(True, False)
            except(RuntimeError) as err:
                print('RuntimeError: ', err)

    # Check missing files for the last time
    missing_files = runner.get_missing_first_level_outputs()
    if missing_files:
        print('Missing files:', missing_files)
        raise Exception('There are missing files for first level analysis.')

    # Start pipeline for the group level only
    runner.nb_subjects = nb_subjects
    runner.start(False, True)

    # Indices and keys to the unthresholded maps
    indices = list(range(1, 18, 2))

    # Retrieve the paths to the reproduced files
    reproduced_files = runner.pipeline.get_hypotheses_outputs()
    reproduced_files = [reproduced_files[i] for i in indices]

    # Retrieve the paths to the results files
    collection = ResultsCollection(team_id)
    results_files = [join(collection.directory, f) for f in sorted(collection.files.keys())]
    results_files = [results_files[i] for i in indices]

    # Compute the correlation coefficients
    return [
        get_correlation_coefficient(reproduced_file, results_file)
        for reproduced_file, results_file in zip(reproduced_files, results_files)
        ]

@helpers.register
def test_correlation_results(values: list, nb_subjects: int) -> bool:
    """ This pytest helper returns True if all values in `values` are greater than
        expected values. It returns False otherwise.

        Arguments:
        - values, list of 9 floats: a list of correlation values for the 9 hypotheses of NARPS
        - nb_subjects, int: the number of subject used to compute the correlation values
    """
    scores = Configuration()['testing']['pipelines']['correlation_thresholds']
    if nb_subjects < 21:
        expected = [scores[0] for _ in range(9)]
    elif nb_subjects < 41:
        expected = [scores[1] for _ in range(9)]
    elif nb_subjects < 61:
        expected = [scores[2] for _ in range(9)]
    elif nb_subjects < 81:
        expected = [scores[3] for _ in range(9)]
    else:
        expected = [scores[4] for _ in range(9)]

    return False not in [v > e for v, e in zip(values, expected)]

@helpers.register
def test_pipeline_evaluation(team_id: str):
    """ Test the execution of a Pipeline and compare with results.
        Arguments:
        - team_id, str: the id of the team for which to test the pipeline

        Return: True if the correlation coefficients between reproduced data and results
            meet the expectations, False otherwise.
    """

    # Remove previous computations
    reproduced_dir = join(
        Configuration()['directories']['reproduced_results'],
        f'NARPS-{team_id}-reproduced'
        )
    rmtree(reproduced_dir, ignore_errors=True)

    file_name = f'test_pipeline-{team_id}.txt'
    if isfile(file_name):
        remove(file_name)

    for subjects in [20, 40, 60, 80, 108]:
        # Execute pipeline
        results = test_pipeline_execution(team_id, subjects)

        # Compute correlation with results
        passed = test_correlation_results(results, subjects)

        # Write values in a file
        with open(file_name, 'a', encoding = 'utf-8') as file:
            file.write(f'| {team_id} | {subjects} subjects | ')
            file.write('success' if passed else 'failure')
            file.write(f' | {[round(i, 2) for i in results]} |\n')

        if not passed:
            break
