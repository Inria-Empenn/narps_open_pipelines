#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_T54A' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_T54A.py
    pytest -q test_team_T54A.py -k <selected_test>
"""
from os import mkdir
from os.path import exists, join
from shutil import rmtree

from pytest import helpers, mark, fixture
from numpy import isclose
from nipype import Workflow
from nipype.interfaces.base import Bunch

from narps_open.pipelines.team_T54A import PipelineTeamT54A
from narps_open.utils.configuration import Configuration

TEMPORARY_DIR = join(Configuration()['directories']['test_runs'], 'test_T54A')

@fixture
def remove_test_dir():
    """ A fixture to remove temporary directory created by tests """

    rmtree(TEMPORARY_DIR, ignore_errors = True)
    mkdir(TEMPORARY_DIR)
    yield # test runs here
    rmtree(TEMPORARY_DIR, ignore_errors = True)

def compare_float_2d_arrays(array_1, array_2):
    """ Assert array_1 and array_2 are close enough """

    assert len(array_1) == len(array_2)
    for reference_array, test_array in zip(array_1, array_2):
        assert len(reference_array) == len(test_array)
        assert isclose(reference_array, test_array).all()

class TestPipelinesTeamT54A:
    """ A class that contains all the unit tests for the PipelineTeamT54A class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineTeamT54A object """

        pipeline = PipelineTeamT54A()

        # 1 - check the parameters
        assert pipeline.fwhm == 4.0
        assert pipeline.team_id == 'T54A'
        assert pipeline.contrast_list == ['1', '2']
        assert pipeline.run_level_contrasts == [
            ('gain', 'T', ['trial', 'gain', 'loss'], [0, 1, 0]),
            ('loss', 'T', ['trial', 'gain', 'loss'], [0, 0, 1])
            ]

        # 2 - check workflows
        assert pipeline.get_preprocessing() is None
        assert isinstance(pipeline.get_run_level_analysis(), Workflow)
        assert isinstance(pipeline.get_subject_level_analysis(), Workflow)
        group_level = pipeline.get_group_level_analysis()

        assert len(group_level) == 3
        for sub_workflow in group_level:
            assert isinstance(sub_workflow, Workflow)

    @staticmethod
    @mark.unit_test
    def test_outputs():
        """ Test the expected outputs of a PipelineTeamT54A object """
        pipeline = PipelineTeamT54A()
        # 1 - 1 subject outputs
        pipeline.subject_list = ['001']
        helpers.test_pipeline_outputs(pipeline, [0, 9*4*1, 5*2*1, 8*2*2 + 4, 18])

        # 2 - 4 subjects outputs
        pipeline.subject_list = ['001', '002', '003', '004']
        helpers.test_pipeline_outputs(pipeline, [0, 9*4*4, 5*2*4, 8*2*2 + 4, 18])

    @staticmethod
    @mark.unit_test
    def test_subject_information():
        """ Test the get_subject_information method """

        # Get test files
        test_file = join(Configuration()['directories']['test_data'], 'pipelines', 'events.tsv')
        test_file_2 = join(Configuration()['directories']['test_data'],
            'pipelines', 'events_resp.tsv')

        # Prepare several scenarii
        info_missed = PipelineTeamT54A.get_subject_information(test_file)
        info_ok = PipelineTeamT54A.get_subject_information(test_file_2)

        # Compare bunches to expected
        bunch = info_missed[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'gain', 'loss', 'difficulty', 'response', 'missed']
        compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [6.459, 14.123, 29.615, 38.723],
            [19.535]
            ])
        compare_float_2d_arrays(bunch.durations, [
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [0.0, 0.0, 0.0, 0.0],
            [0.0]
            ])
        compare_float_2d_arrays(bunch.amplitudes, [
            [1.0, 1.0, 1.0, 1.0],
            [14.0, 34.0, 10.0, 16.0],
            [6.0, 14.0, 15.0, 17.0],
            [1.0, 3.0, 10.0, 9.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0]
            ])
        assert bunch.regressor_names == None
        assert bunch.regressors == None

        bunch = info_ok[0]
        assert isinstance(bunch, Bunch)
        assert bunch.conditions == ['trial', 'gain', 'loss', 'difficulty', 'response']
        compare_float_2d_arrays(bunch.onsets, [
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [4.071, 11.834, 27.535, 36.435],
            [6.459, 14.123, 29.615, 38.723]
            ])
        compare_float_2d_arrays(bunch.durations, [
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [2.388, 2.289, 2.08, 2.288],
            [0.0, 0.0, 0.0, 0.0]
            ])
        compare_float_2d_arrays(bunch.amplitudes, [
            [1.0, 1.0, 1.0, 1.0],
            [14.0, 34.0, 10.0, 16.0],
            [6.0, 14.0, 15.0, 17.0],
            [1.0, 3.0, 10.0, 9.0],
            [1.0, 1.0, 1.0, 1.0]
            ])
        assert bunch.regressor_names == None
        assert bunch.regressors == None

    @staticmethod
    @mark.unit_test
    def test_parameters_file(remove_test_dir):
        """ Test the get_parameters_file method """

        confounds_file_path = join(
            Configuration()['directories']['test_data'], 'pipelines', 'confounds.tsv')

        PipelineTeamT54A.get_parameters_file(
            confounds_file_path,
            'fake_subject_id',
            'fake_run_id',
            TEMPORARY_DIR
            )

        # Check parameter file was created
        assert exists(join(
            TEMPORARY_DIR,
            'parameters_file',
            'parameters_file_sub-fake_subject_id_run-fake_run_id.tsv')
        )

    @staticmethod
    @mark.unit_test
    def test_one_sample_t_test_regressors():
        """ Test the get_one_sample_t_test_regressors method """

        regressors = PipelineTeamT54A.get_one_sample_t_test_regressors(['001', '002'])
        assert regressors == {'group_mean': [1, 1]}

    @staticmethod
    @mark.unit_test
    def test_two_sample_t_test_regressors():
        """ Test the get_two_sample_t_test_regressors method """

        regressors, groups = PipelineTeamT54A.get_two_sample_t_test_regressors(
            ['001', '003'], # equalRange group
            ['002', '004'], # equalIndifference group
            ['001', '002', '003', '004'] # all subjects
            )
        assert regressors == dict(
                equalRange = [1, 0, 1, 0],
                equalIndifference = [0, 1, 0, 1]
            )
        assert groups == [1, 2, 1, 2]

    @staticmethod
    @mark.pipeline_test
    def test_execution():
        """ Test the execution of a PipelineTeamT54A and compare results """
        helpers.test_pipeline_evaluation('T54A')
